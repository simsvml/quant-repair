import dataclasses
from dataclasses import dataclass
import itertools
import json
import math
import os
import shutil
import sys
import time
import tomllib
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchtune.models.llama3 import llama3_tokenizer_transformers
from torchtune.modules import RotaryPositionalEmbeddings
from gguf import GGUFReader, GGUFWriter
from tqdm import tqdm
from quant_repair.architecture import Llama3Arch
from quant_repair import datasets
from quant_repair.forward import SuperbatchEmbeddings
from quant_repair import functional as QRF
import quant_repair.gguf
import quant_repair.loader
from quant_repair.loader import StateDictLoader, SafetensorsLoader, GGUFLoader
from quant_repair.memory_accounting import MEMORY_ACCOUNTING
from quant_repair import model_util as QRM
from quant_repair.model_util.llama3_lora import TrainableParams, LayerTrainableParams
from quant_repair.model_util.superbatch import run_forward_superbatch
from quant_repair.offload import TensorOffload
from quant_repair import quantized
from quant_repair.quantized import DequantizeParams
import torch_ggml_quant


def config_as_dict(x) -> dict:
    if dataclasses.is_dataclass(x):
        d = dataclasses.asdict(x)
        return {k: config_as_dict(v) for k, v in d.items()}
    else:
        return x

@dataclass(frozen = True)
class TrainConfig:
    total_tokens: int
    max_seq_len: int
    gradient_accumulation_steps: int

    save_checkpoint_interval: Optional[int]
    backup_checkpoint_interval: Optional[int]

    @classmethod
    def from_dict(cls, d):
        def opt_int(x):
            return int(x) if x is not None else None
        x = cls(
            total_tokens = int(d['total_tokens']),
            max_seq_len = d['max_seq_len'],
            gradient_accumulation_steps = d['gradient_accumulation_steps'],
            save_checkpoint_interval = opt_int(d.get('save_checkpoint_interval')),
            backup_checkpoint_interval = opt_int(d.get('backup_checkpoint_interval')),
        )
        extra_keys = [k for k in d.keys() if k not in cls.__dataclass_fields__]
        assert len(extra_keys) == 0, 'extra keys in config: %r' % (extra_keys,)
        return x

    def optimizer_steps(self) -> int:
        # TODO: This is not quite accurate, since there is a partial steps at
        # the end of each superbatch, resulting in more optimizer steps than
        # would usually be required.
        grad_acc = self.gradient_accumulation_steps
        return (self.total_samples + grad_acc - 1) // grad_acc

@dataclass(frozen = True)
class OptimizerConfig:
    lr: float
    weight_decay: float
    lr_schedule: Optional[str]
    lr_schedule_params: dict

    @classmethod
    def from_dict(cls, d):
        x = cls(
            lr = d['lr'],
            weight_decay = d.get('weight_decay', 0.0),
            lr_schedule = d.get('lr_schedule'),
            lr_schedule_params = d.get('lr_schedule_params') or {},
        )
        extra_keys = [k for k in d.keys() if k not in cls.__dataclass_fields__]
        assert len(extra_keys) == 0, 'extra keys in config: %r' % (extra_keys,)
        return x

@dataclass(frozen = True)
class DatasetConfig:
    name: str
    shuffle_seed: int
    skip: int

    @classmethod
    def from_dict(cls, d):
        x = cls(
            name = d['name'],
            shuffle_seed = d.get('shuffle_seed', 0),
            skip = d.get('skip', 0),
        )
        extra_keys = [k for k in d.keys() if k not in cls.__dataclass_fields__]
        assert len(extra_keys) == 0, 'extra keys in config: %r' % (extra_keys,)
        return x

@dataclass(frozen = True)
class MemoryConfig:
    weights_vram_gb: float
    superbatch_ram_gb: float

    @classmethod
    def from_dict(cls, d):
        x = cls(
            weights_vram_gb = d.get('weights_vram_gb', 10),
            superbatch_ram_gb = d.get('superbatch_ram_gb', 8),
        )
        extra_keys = [k for k in d.keys() if k not in cls.__dataclass_fields__]
        assert len(extra_keys) == 0, 'extra keys in config: %r' % (extra_keys,)
        return x

@dataclass(frozen = True)
class ModelConfig:
    lora_rank: int
    lora_alpha: float
    lora_dropout: float

    @classmethod
    def from_dict(cls, d):
        x = cls(
            lora_rank = d.get('lora_rank', 32),
            lora_alpha = d.get('lora_alpha', 8.0),
            lora_dropout = d.get('lora_dropout', 0.0),
        )
        extra_keys = [k for k in d.keys() if k not in cls.__dataclass_fields__]
        assert len(extra_keys) == 0, 'extra keys in config: %r' % (extra_keys,)
        return x

@dataclass(frozen = True)
class Config:
    model_arch: str
    orig_weights_safetensors_dir: str
    quant_weights_gguf_path: str

    train: TrainConfig
    optimizer: OptimizerConfig
    dataset: DatasetConfig
    memory: MemoryConfig
    model: ModelConfig

    @classmethod
    def from_dict(cls, d):
        x = cls(
            model_arch = d['model_arch'],
            orig_weights_safetensors_dir = d['orig_weights_safetensors_dir'],
            quant_weights_gguf_path = d['quant_weights_gguf_path'],
            train = TrainConfig.from_dict(d.get('train', {})),
            optimizer = OptimizerConfig.from_dict(d.get('optimizer', {})),
            dataset = DatasetConfig.from_dict(d.get('dataset', {})),
            memory = MemoryConfig.from_dict(d.get('memory', {})),
            model = ModelConfig.from_dict(d.get('model', {})),
        )
        extra_keys = [k for k in d.keys() if k not in cls.__dataclass_fields__]
        assert len(extra_keys) == 0, 'extra keys in config: %r' % (extra_keys,)
        return x


def get_model_arch(name):
    if name == 'llama3_8b':
        return Llama3Arch.llama3_8b()
    elif name == 'llama3_70b':
        return Llama3Arch.llama3_70b()
    else:
        raise ValueError('unknown model name %r' % (name,))

def build_optimizer(cfg, params):
    return torch.optim.AdamW(
        params,
        lr = cfg.optimizer.lr,
        weight_decay = cfg.optimizer.weight_decay,
    )

def build_lr_schedule(cfg, optimizer):
    """
    Return a closure that can be called as `lr_schedule(step)` to update
    `optimizer`'s learning rates for the given `step`.  For these schedules,
    `step` should be the number of tokens processed so far.
    """
    kind = cfg.optimizer.lr_schedule
    if kind is None:
        lr_lambda = lambda _: 1
    elif kind == 'cosine':
        warmup = cfg.optimizer.lr_schedule_params['warmup_factor']
        # `lr_lambda` copied from `torchtune/modules/lr_schedulers.py`,
        # function `get_cosine_schedule_with_warmup`.
        num_training_steps = cfg.train.total_tokens
        num_warmup_steps = int(num_training_steps * warmup)
        num_cycles = 0.5
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return current_step / max(1, num_warmup_steps)
            progress = (current_step - num_warmup_steps) / max(
                1, num_training_steps - num_warmup_steps
            )
            cosine_lr_multiple = 0.5 * (
                1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
            )
            return max(0.0, cosine_lr_multiple)
    else:
        raise ValueError('unknown lr schedule %r' % (kind,))

    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    def apply(step):
        factor = lr_lambda(step)
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            param_group['lr'] = base_lr * factor

    return apply

def build_dataset(cfg, tokenizer):
    name = cfg.dataset.name
    if name == 'slimorca':
        dataset = datasets.load_slimorca_dataset(tokenizer, cfg.train.max_seq_len)
    elif name == 'wikitext':
        dataset = datasets.load_wikitext_dataset(tokenizer, cfg.train.max_seq_len)
    else:
        raise ValueError('unknown dataset %r' % (name,))

    return dataset.shuffle(seed = cfg.dataset.shuffle_seed)


def run_backward_step(
    x: Tensor,
    grad_y: Tensor,
    f,
) -> Tensor:
    """
    Given input `x` and output gradient `grad_y`, where `y = f(x)`, compute the
    input gradient `grad_x`.
    """
    x.requires_grad_(True)
    MEMORY_ACCOUNTING.register(x, 'run_backward_step x')
    MEMORY_ACCOUNTING.register(grad_y, 'run_backward_step grad_y')
    y = f(x)
    MEMORY_ACCOUNTING.register(y, 'run_backward_step y')
    y.backward(grad_y)
    MEMORY_ACCOUNTING.register(x.grad, 'run_backward_step x.grad')
    return x.grad

def run_initial_backward_step(
    x: Tensor,
    f,
) -> Tuple[Tensor, Tensor]:
    """
    Given input `x`, compute `y = f(x)` and the gradient of `x`.
    """
    x.requires_grad_(True)
    MEMORY_ACCOUNTING.register(x, 'run_initial_backward_step x')
    y = f(x)
    MEMORY_ACCOUNTING.register(y, 'run_initial_backward_step y')
    y.backward()
    MEMORY_ACCOUNTING.register(x.grad, 'run_initial_backward_step x.grad')
    return y, x.grad

def run_final_backward_step(
    x: Tensor,
    grad_y: Tensor,
    f,
):
    """
    Given input `x` and output gradient `grad_y`, where `y = f(x)`, update
    parameter graidents.  The gradient of `x` is not returned.
    """
    MEMORY_ACCOUNTING.register(x, 'run_initial_backward_step x')
    MEMORY_ACCOUNTING.register(grad_y, 'run_backward_step grad_y')
    y = f(x)
    MEMORY_ACCOUNTING.register(y, 'run_backward_step y')
    y.backward(grad_y)
    return x.grad


def run_init(config_path, checkpoint_path):
    """
    Create an initial checkpoint based on the configuration.
    """
    print('loading config from %r' % config_path)
    cfg_dict = tomllib.load(open(config_path, 'rb'))
    cfg = Config.from_dict(cfg_dict)
    cfg_dict = config_as_dict(cfg)

    if os.path.exists(checkpoint_path):
        print('Output file %r already exists' % checkpoint_path)
        answer = input('Overwrite? ')
        if answer.lower() not in ('y', 'yes'):
            print('Operation cancelled')
            return


    arch = get_model_arch(cfg.model_arch)
    device = torch.device('cuda')
    torch.set_default_dtype(torch.bfloat16)


    quant_loader = GGUFLoader(cfg.quant_weights_gguf_path,
        convert = quant_repair.loader.llama_cpp_conversion())


    # Initialize trainable params

    lora_alpha = cfg.model.lora_alpha

    dims = QRM.llama3_lora.linear_dimensions(arch)

    def init_lora_params(
        key: str,
        # TODO: Try float16 for easier llama.cpp compat
        dtype: torch.dtype = torch.bfloat16,
    ) -> QRF.LowRankAdapterParams:
        n, m = dims.get(key)
        #rank = rank_map[key]
        rank = cfg.model.lora_rank
        return QRF.LowRankAdapterParams(
            lora_a = QRF.TensorParams.from_unquantized_tensor(
                torch.empty((rank, n), dtype=dtype, device=device).normal_()),
            lora_b = QRF.TensorParams.from_unquantized_tensor(
                torch.zeros((m, rank), dtype=dtype, device=device)),
            lora_alpha = lora_alpha / rank,
        )

    def init_layer_params(layer_index: int) -> LayerTrainableParams:
        embed_dim = arch.embed_dim
        num_heads = arch.num_heads
        num_kv_heads = arch.num_kv_heads
        head_dim = arch.head_dim()
        hidden_dim = arch.hidden_dim()

        return LayerTrainableParams(
            q_proj = init_lora_params('layers.%d.attn.q_proj' % layer_index),
            k_proj = init_lora_params('layers.%d.attn.k_proj' % layer_index),
            v_proj = init_lora_params('layers.%d.attn.v_proj' % layer_index),
            output_proj = init_lora_params('layers.%d.attn.output_proj' % layer_index),
            gate_proj = init_lora_params('layers.%d.mlp.w1' % layer_index),
            down_proj = init_lora_params('layers.%d.mlp.w2' % layer_index),
            up_proj = init_lora_params('layers.%d.mlp.w3' % layer_index),
            sa_norm = QRF.RMSNormParams(
                quant_loader.get('layers.%d.sa_norm.scale' % layer_index)),
            mlp_norm = QRF.RMSNormParams(
                quant_loader.get('layers.%d.mlp_norm.scale' % layer_index)),
        )

    def init_train_params() -> TrainableParams:
        return TrainableParams(
            tok_embeddings = init_lora_params('tok_embeddings'),
            layers = [init_layer_params(i) for i in range(arch.num_layers)],
            norm = QRF.RMSNormParams(quant_loader.get('norm.scale')),
            output = init_lora_params('output'),
        )

    train_params = init_train_params()

    for tensor in train_params.tensors():
        tensor.requires_grad_(True)


    # Initialize optimizer and LR schedule
    optimizer = build_optimizer(cfg, list(train_params.tensors()))


    checkpoint_dict = {
        'cfg': cfg_dict,
        'params': QRM.llama3_lora.save_trainable_params(train_params),
        'optimizer': optimizer.state_dict(),
        'progress': {
            'samples': 0,
            'optimizer_steps': 0,
            'tokens': 0,
        },
    }
    torch.save(checkpoint_dict, checkpoint_path)


def run_train(checkpoint_path):
    print('loading checkpoint from %r' % checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, weights_only = True)
    cfg_dict = checkpoint_dict['cfg']
    cfg = Config.from_dict(cfg_dict)
    cfg_dict = config_as_dict(cfg)

    arch = get_model_arch(cfg.model_arch)
    device = torch.device('cuda')
    torch.set_default_dtype(torch.bfloat16)
    #MEMORY_ACCOUNTING.disable()
    MEMORY_ACCOUNTING.disable_print()


    print('loading original weights from %r' % (cfg.orig_weights_safetensors_dir,))
    orig_loader = SafetensorsLoader(cfg.orig_weights_safetensors_dir,
        convert = quant_repair.loader.hf_conversion(arch))


    # Offload setup
    print('loading quantized weights from %r' % (cfg.quant_weights_gguf_path,))
    quant_loader_raw = GGUFLoader(cfg.quant_weights_gguf_path,
        convert = quant_repair.loader.llama_cpp_conversion())

    offload = TensorOffload(device, vram_limit_gb=cfg.memory.weights_vram_gb)

    offload.add_group('tok_embeddings')
    for i in range(arch.num_layers):
        offload.add_group('layers.%d' % i)
    offload.add_group('norm')
    offload.add_group('output')

    def offload_init_tensors():
        def add_weight(group, name):
            full_name = '%s.%s' % (group, name)
            tensor = quant_loader_raw.get(full_name, device = 'cpu')
            offload.add_tensor(group, full_name, tensor, read_only=True)

        add_weight('tok_embeddings', 'weight')
        for i in range(arch.num_layers):
            add_weight('layers.%d' % i, 'attn.q_proj.weight')
            add_weight('layers.%d' % i, 'attn.k_proj.weight')
            add_weight('layers.%d' % i, 'attn.v_proj.weight')
            add_weight('layers.%d' % i, 'attn.output_proj.weight')
            add_weight('layers.%d' % i, 'mlp.w1.weight')
            add_weight('layers.%d' % i, 'mlp.w2.weight')
            add_weight('layers.%d' % i, 'mlp.w3.weight')
            add_weight('layers.%d' % i, 'sa_norm.scale')
            add_weight('layers.%d' % i, 'mlp_norm.scale')
        add_weight('norm', 'scale')
        add_weight('output', 'weight')
    offload_init_tensors()

    offload.build_partitions()

    # Use `offload` for fetching base model weights.
    quant_loader = offload


    if os.path.isdir(cfg.orig_weights_safetensors_dir):
        orig_dir = cfg.orig_weights_safetensors_dir
    else:
        orig_dir = os.path.dirname(cfg.orig_weights_safetensors_dir)
    tokenizer_json_path = os.path.join(orig_dir, 'tokenizer.json')
    print('loading tokenizer from %s' % tokenizer_json_path)
    tokenizer = llama3_tokenizer_transformers(tokenizer_json_path)


    print('loading params from checkpoint')
    param_loader = StateDictLoader(checkpoint_dict['params'])
    with torch.no_grad():
        train_params = QRM.llama3_lora.load_trainable_params(
            param_loader, arch.num_layers, device)
    MEMORY_ACCOUNTING.register_params(train_params, 'trainable params')


    # Initialize optimizer and LR schedule
    optimizer = build_optimizer(cfg, list(train_params.tensors()))
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    # TODO: If `lr_schedule` is changed to have internal state, be sure to
    # update checkpoint saving/loading and `register_optimizer_tensors` to
    # process its state_dict.
    lr_schedule = build_lr_schedule(cfg, optimizer)
    kl_div_loss_fn = nn.KLDivLoss(log_target=True, reduction='batchmean')

    def register_optimizer_tensors():
        MEMORY_ACCOUNTING.register_all(
            (x for x in optimizer.state_dict().values() if isinstance(x, Tensor)),
            'optimizer state')
    register_optimizer_tensors()


    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file_path = checkpoint_path + '.train_%s.log' % timestamp
    print('writing log to %r' % log_file_path)
    assert not os.path.exists(log_file_path)
    log_file = open(log_file_path, 'w')
    def write_log(obj):
        json.dump(obj, log_file)
        log_file.write('\n')
        log_file.flush()
    write_log(cfg_dict)


    progress = checkpoint_dict['progress'].copy()
    write_log(progress)


    del checkpoint_dict


    # Set up dataset and loader
    print('loading dataset')
    dataset = build_dataset(cfg, tokenizer)
    if progress['samples'] > 0:
        print('skipping %d samples that were already processed' % progress['samples'])
        dataset = dataset.skip(progress['samples'])
    dataloader = DataLoader(dataset)


    model_common = QRM.llama3.ModelCommon(
        rope = RotaryPositionalEmbeddings(
            dim = arch.head_dim(),
            max_seq_len = arch.max_seq_len,
            base = arch.rope_base,
        ).to(device),
    )
    model = QRM.llama3.make_model(arch, model_common)

    lora_dropout = cfg.model.lora_dropout

    def embedding_with_lora():
        base = QRF.Embedding()
        adapter = QRF.EmbeddingLowRankAdapter(dropout = lora_dropout)
        return QRF.WithAdapter(base, adapter)

    def linear_with_lora():
        base = QRF.Linear()
        adapter = QRF.LowRankAdapter(dropout = lora_dropout)
        return QRF.WithAdapter(base, adapter)

    model_with_lora = QRM.llama3.make_model(
        arch, model_common, linear_with_lora, embedding_with_lora)


    # Activation checkpointing and offloading
    activation_checkpoint_buffer = None
    activation_shapes = []

    def init_activation_checkpoint_buffer(samples):
        nonlocal activation_checkpoint_buffer
        num_samples = len(samples)
        max_seq_len = max(s.numel() for s in samples)
        if activation_checkpoint_buffer is not None \
                and activation_checkpoint_buffer.shape[1] >= num_samples \
                and activation_checkpoint_buffer.shape[2] >= max_seq_len:
            return

        del activation_checkpoint_buffer
        activation_checkpoint_buffer = torch.empty(
            (arch.num_layers + 1, num_samples, max_seq_len, arch.embed_dim))

    def save_activation_checkpoint(layer_index, activations):
        for i, act in enumerate(activations):
            if i >= len(activation_shapes):
                activation_shapes.append(None)
            activation_shapes[i] = act.shape
            act = act.view(-1, arch.embed_dim)
            activation_checkpoint_buffer[layer_index, i, :act.shape[0], :].copy_(act)

    def load_activation_checkpoint(layer_index, sample_index):
        out = torch.empty(activation_shapes[sample_index], device=device)
        out_view = out.view(-1, arch.embed_dim)
        out_view.copy_(
            activation_checkpoint_buffer[layer_index, sample_index, :out_view.shape[0], :])
        return out


    def save_checkpoint():
        checkpoint_dict = {
            'cfg': cfg_dict,
            'params': QRM.llama3_lora.save_trainable_params(train_params),
            'optimizer': optimizer.state_dict(),
            'progress': progress,
        }
        torch.save(checkpoint_dict, checkpoint_path)
        tqdm.write('[%s] saved checkpoint to %s (after %d tokens)'
            % (time.asctime(), checkpoint_path, progress['tokens']))

    def backup_checkpoint():
        base, ext = os.path.splitext(checkpoint_path)
        index = progress['tokens'] // cfg.train.backup_checkpoint_interval
        path = '%s_ckpt%d%s' % (base, index, ext)
        tqdm.write('backed up checkpoint to %s' % path)
        shutil.copyfile(checkpoint_path, path)


    samples_iter = iter(dataloader)

    embeds_orig = SuperbatchEmbeddings(arch, ram_gb = cfg.memory.superbatch_ram_gb)
    superbatch_limit = embeds_orig.tokens_free()

    gradient_accumulation_steps = cfg.train.gradient_accumulation_steps

    # Overall training progress, counted by number of tokens processed.
    pbar_tokens = tqdm(
        desc = 'tokens',
        total = cfg.train.total_tokens,
        initial = progress['tokens'],
        smoothing = 0,
        unit_scale = True,
    )
    # Progress through the current superbatch, in tokens.
    pbar_super_tokens = tqdm(
        desc = 'superbatch tokens',
        total = 1,
        unit_scale = True,
    )
    # Superbatch forward pass progress, in layers finished.
    pbar_super_forward = tqdm(desc = 'superbatch forward', total = 2 + arch.num_layers)
    # Superbatch layer progress, in samples processed.
    pbar_super_layer = tqdm(desc = 'superbatch layer', total = 1)
    # Training forward pass progress, in layers.
    pbar_train_forward = tqdm(desc = 'train forward', total = 1)
    # Training backward pass progress, in layers.
    pbar_train_backward = tqdm(desc = 'train backward', total = 1)

    def collect_batch_coroutine(samples):
        limit = yield
        batch = []
        size = 0
        for sample in samples:
            sample_size = sample.numel()
            while sample_size > limit - size:
                limit = yield batch
                batch = []
                size = 0
            batch.append(sample)
            size += sample_size
            assert size <= limit
        yield batch
        while True:
            yield []
    collect_batch = collect_batch_coroutine(samples_iter)
    collect_batch.send(None)

    while True:
        assert progress['tokens'] <= cfg.train.total_tokens
        limit = min(superbatch_limit, cfg.train.total_tokens - progress['tokens'])
        superbatch_samples = collect_batch.send(limit)
        if len(superbatch_samples) == 0:
            # Either we're close to the `total_tokens` limit and the next
            # sample would put us over, or we've exhausted the dataset.
            break

        pbar_super_tokens.reset(sum(s.numel() for s in superbatch_samples))
        run_forward_superbatch(
            model,
            orig_loader,
            superbatch_samples,
            embeds_orig,
            device,
            pbar_forward = pbar_super_forward,
            pbar_layer = pbar_super_layer,
        )

        # Reset a second time so the estimated time remaining only considers
        # actual training steps, not the superbatch forward pass.
        pbar_super_tokens.reset(sum(s.numel() for s in superbatch_samples))

        MEMORY_ACCOUNTING.report('after orig forward pass')

        # Train using the collected embeddings.
        for samples_start in range(0, len(superbatch_samples), gradient_accumulation_steps):
            samples = superbatch_samples[samples_start :
                samples_start + gradient_accumulation_steps]
            samples = [t.to(device).requires_grad_(False) for t in samples]
            batch_tokens = sum(s.numel() for s in samples)
            MEMORY_ACCOUNTING.register_all(samples, 'samples')

            init_activation_checkpoint_buffer(samples)
            activations = []

            pbar_train_forward.reset(2 + arch.num_layers)
            pbar_train_backward.reset(1 + 2 + arch.num_layers)

            with torch.no_grad():
                m = QRM.llama3_lora.build_trainable_tok_embeddings(
                    model_with_lora, quant_loader, train_params, device)
                activations = [m(sample) for sample in samples]
                MEMORY_ACCOUNTING.register_all(activations, 'activations')
                save_activation_checkpoint(0, activations)
                del m
                pbar_train_forward.update()

                for i in range(arch.num_layers):
                    m = QRM.llama3_lora.build_trainable_layer(
                        model_with_lora, quant_loader, train_params, i, device)
                    # Explicit loop instead of list comprehension ensures only one
                    # extra activation tensor is in memory at a time.
                    #activations = [m(act) for act in activations]
                    for j in range(len(activations)):
                        activations[j] = m(activations[j])
                    MEMORY_ACCOUNTING.register_all(activations, 'activations')
                    save_activation_checkpoint(i + 1, activations)
                    del m
                    pbar_train_forward.update()

                m = QRM.llama3_lora.build_trainable_norm(
                    model_with_lora, quant_loader, train_params, device)
                for j in range(len(activations)):
                    activations[j] = m(activations[j])
                MEMORY_ACCOUNTING.register_all(activations, 'activations')
                # Norm output is not recorded as a checkpoint.
                del m
                pbar_train_forward.update()

                # Final activations are kept around for later use in
                # calc_loss.

            MEMORY_ACCOUNTING.report('after train forward pass')

            # Run loss forward and backward
            m_orig = QRM.llama3.build_forward_output(model, orig_loader, device)
            m_train = QRM.llama3_lora.build_trainable_output(
                model_with_lora, quant_loader, train_params, device)

            # Sum of loss values (used only for logging).
            loss_sum = 0.0
            gradients = []
            for i in range(len(samples)):
                orig_logits = m_orig(embeds_orig[samples_start + i].to(device))
                MEMORY_ACCOUNTING.register(orig_logits, 'orig_logits')
                orig_log_prob = F.log_softmax(orig_logits, dim=-1)
                MEMORY_ACCOUNTING.register(orig_log_prob, 'orig_log_prob')
                orig_log_prob = orig_log_prob.view(-1, arch.vocab_size)
                del orig_logits

                def calc_loss(train_embeds: Tensor) -> Tensor:
                    train_logits = m_train(train_embeds)
                    MEMORY_ACCOUNTING.register(train_logits, 'train_logits')
                    train_log_prob = F.log_softmax(train_logits, dim=-1)
                    MEMORY_ACCOUNTING.register(train_log_prob, 'train_log_prob')
                    train_log_prob = train_log_prob.view(-1, arch.vocab_size)
                    del train_logits

                    loss = kl_div_loss_fn(train_log_prob, orig_log_prob)
                    MEMORY_ACCOUNTING.register(loss, 'loss')
                    loss = loss / len(samples)
                    return loss

                train_embeds = activations[i]
                loss, grad = run_initial_backward_step(train_embeds, calc_loss)
                MEMORY_ACCOUNTING.register(grad, 'backward gradients')
                activations[i] = None

                gradients.append(grad)
                loss_sum += loss.item()

            pbar_train_backward.update()
            MEMORY_ACCOUNTING.report('after initial backward pass')

            # Loss is already divided by `len(samples)` (which is the number of
            # gradient accumulation steps) inside `calc_loss`.
            loss_avg = loss_sum
            #loss_avg = loss_sum / len(samples)
            del m_orig, m_train

            m = QRM.llama3_lora.build_trainable_norm(
                model_with_lora, quant_loader, train_params, device)
            for j in range(len(samples)):
                act = load_activation_checkpoint(arch.num_layers, j)
                gradients[j] = run_backward_step(act, gradients[j], m)
            MEMORY_ACCOUNTING.register_all(gradients, 'backward gradients')
            del m, act
            pbar_train_backward.update()

            for i in reversed(range(arch.num_layers)):
                m = QRM.llama3_lora.build_trainable_layer(
                    model_with_lora, quant_loader, train_params, i, device)
                for j in range(len(samples)):
                    act = load_activation_checkpoint(i, j)
                    gradients[j] = run_backward_step(act, gradients[j], m)
                MEMORY_ACCOUNTING.register_all(gradients, 'backward gradients')
                del m, act
                pbar_train_backward.update()

            m = QRM.llama3_lora.build_trainable_tok_embeddings(
                model_with_lora, quant_loader, train_params, device)
            for j, sample in enumerate(samples):
                run_final_backward_step(sample, gradients[j], m)
            del m, sample, gradients
            pbar_train_backward.update()

            MEMORY_ACCOUNTING.report('after full backward pass')

            progress['samples'] += len(samples)
            old_tokens = progress['tokens']
            progress['tokens'] += batch_tokens

            lr_schedule(progress['tokens'])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            progress['optimizer_steps'] += 1

            register_optimizer_tensors()
            MEMORY_ACCOUNTING.report('after optimizer')

            pbar_tokens.update(batch_tokens)
            pbar_super_tokens.update(batch_tokens)


            write_log(dict(
                loss = loss_avg,
                lr = optimizer.param_groups[0]["lr"],
                gpu_resources = torch.cuda.memory_allocated(),
                progress = progress,
                time = time.time(),
            ))

            if old_tokens // cfg.train.save_checkpoint_interval \
                    != progress['tokens'] // cfg.train.save_checkpoint_interval:
                save_checkpoint()

            if old_tokens // cfg.train.backup_checkpoint_interval \
                    != progress['tokens'] // cfg.train.backup_checkpoint_interval:
                backup_checkpoint()


    save_checkpoint()


def run_extract_config(checkpoint_path):
    # Disable printing the final memory report.  The report normally goes to
    # stdout, which would interfere with redirecting the config toml to a file.
    MEMORY_ACCOUNTING.disable()

    print('loading checkpoint from %r' % checkpoint_path, file=sys.stderr)
    checkpoint_dict = torch.load(checkpoint_path, weights_only = True, map_location = 'meta')
    cfg_dict = checkpoint_dict['cfg']
    cfg = Config.from_dict(cfg_dict)

    lines = []
    def emit(line):
        lines.append(line)

    def emit_toml_key_value(key, value, prefix=''):
        if value is None:
            emit('# %s = UNSET' % key)
        elif isinstance(value, (int, float, str)):
            emit('%s = %r' % (key, value))
        elif isinstance(value, dict):
            emit('\n[%s%s]' % (prefix, key))
            emit_dict_toml(value, prefix = '%s%s.' % (prefix, key))
        elif dataclasses.is_dataclass(value):
            emit('\n[%s%s]' % (prefix, key))
            emit_cfg_toml(value, prefix = '%s%s.' % (prefix, key))
        else:
            raise TypeError('emit_cfg_toml: unknown type %r' % type(value))

    def emit_toml_table(items, prefix=''):
        subtables = []
        for key, value in items:
            if isinstance(value, dict) or dataclasses.is_dataclass(value):
                subtables.append((key, value))
            else:
                emit_toml_key_value(key, value, prefix = prefix)
        for key, value in subtables:
            emit_toml_key_value(key, value, prefix = prefix)

    def emit_cfg_toml(c, prefix=''):
        emit_toml_table(
            ((field.name, getattr(c, field.name)) for field in dataclasses.fields(type(c))),
            prefix = prefix)

    def emit_dict_toml(d, prefix=''):
        emit_toml_table(d.items(), prefix = prefix)

    emit_cfg_toml(cfg)

    toml_str = '\n'.join(lines)
    cfg_dict_orig = config_as_dict(cfg)
    cfg_dict_reparsed = tomllib.loads(toml_str)
    assert cfg_dict_orig == cfg_dict_reparsed, \
            "sanity check failed - printed TOML doesn't match original"

    print(toml_str)

def run_update_config(checkpoint_path, config_path):
    print('loading checkpoint from %r' % checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, weights_only = True, map_location = 'cpu')
    old_cfg_dict = checkpoint_dict['cfg']
    old_cfg = Config.from_dict(old_cfg_dict)
    old_cfg_dict = config_as_dict(old_cfg)

    print('loading config from %r' % config_path)
    new_cfg_dict = tomllib.load(open(config_path, 'rb'))
    new_cfg = Config.from_dict(new_cfg_dict)
    new_cfg_dict = config_as_dict(new_cfg)

    if old_cfg_dict == new_cfg_dict:
        print('Configs already match; nothing to do')
        return

    def flatten_dict(d, out = None, prefix = ''):
        if out is None:
            out = {}

        for key, value in d.items():
            if isinstance(value, dict):
                flatten_dict(value, out = out, prefix = '%s%s.' % (prefix, key))
            else:
                key_ext = prefix + key
                assert key_ext not in out, 'duplicate key %r' % key_ext
                out[key_ext] = value

        return out

    old_flat = flatten_dict(old_cfg_dict)
    new_flat = flatten_dict(new_cfg_dict)

    print('\nChanges:')
    all_keys = set(old_flat.keys()) | set(new_flat.keys())
    for key in sorted(all_keys):
        # `key` can't be missing from both dicts, since in that case it
        # wouldn't be present in `all_keys`.
        if key not in old_flat:
            print('  %s: UNSET -> %r' % (key, new_flat[key]))
        elif key not in new_flat:
            print('  %s: %r -> UNSET' % (key, old_flat[key]))
        else:
            old_value = old_flat[key]
            new_value = new_flat[key]
            if old_value != new_value:
                print('  %s: %r -> %r' % (key, old_value, new_value))

    print()
    answer = input('Apply these changes? ')
    if answer.lower() not in ('y', 'yes'):
        print('Operation cancelled')
        return

    checkpoint_dict['cfg'] = new_cfg_dict
    torch.save(checkpoint_dict, checkpoint_path)
    print('updated %r' % checkpoint_path)


def run_export_gguf(checkpoint_path, gguf_path, quantize_lora = False):
    if os.path.exists(gguf_path):
        print('Refusing to overwrite existing file %r' % gguf_path)
        return

    MEMORY_ACCOUNTING.disable()

    print('loading checkpoint from %r' % checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, weights_only = True, map_location = 'cpu')
    cfg = Config.from_dict(checkpoint_dict['cfg'])

    arch = get_model_arch(cfg.model_arch)
    device = torch.device('cpu')

    print('loading params from checkpoint')
    param_loader = StateDictLoader(checkpoint_dict['params'])
    with torch.no_grad():
        train_params = QRM.llama3_lora.load_trainable_params(
            param_loader, arch.num_layers, device)

    print('loading original gguf from %r' % cfg.quant_weights_gguf_path)
    reader = quant_repair.gguf.GGUFReader2(cfg.quant_weights_gguf_path)
    #reader = quant_repair.gguf.GGUFReader2(cfg.quant_weights_gguf_path
    #    .replace('_XXS.', '_XS.'))

    writer = quant_repair.gguf.GGUFWriter2(gguf_path)

    # Copy KV entries.  We adjust the architecture from `llama` to
    # `llamawithlora`, which requires updating key names that are prefixed with
    # the old architecture name.
    new_arch = reader.architecture + 'withlora'
    writer.write_kv_str('general.architecture', new_arch)
    arch_dot = reader.architecture + '.'
    for kv in reader.kvs:
        if kv.key == 'general.architecture':
            continue
        if kv.key.startswith(arch_dot):
            new_key = '%s.%s' % (new_arch, kv.key[len(arch_dot):])
            writer.copy_kv_with_key(reader, new_key, kv)
        else:
            writer.copy_kv(reader, kv)

    # Add tensors.

    # Copy tensor descriptions from the quantized model GGUF.  Norm weights
    # are excluded in favor of the trained versions from the checkpoint.
    copied_tensors = [t for t in reader.tensors if 'norm.weight' not in t.name]
    writer.copy_tensors(reader, copied_tensors)

    tensor_data = {}
    def add_tensor(
        name: str,
        tensor: QRF.TensorParams,
        transpose: bool = False,
        quantize: bool = False,
    ):
        if not transpose:
            dimensions = tuple(reversed(tensor.dequant.shape))
        else:
            assert len(tensor.dequant.shape) == 2
            dimensions = tuple(tensor.dequant.shape)

        if not quantize:
            type_ = quant_repair.gguf.GGMLQuantizationType(int(tensor.dequant.quant))
            data = tensor.data.detach()
            if transpose:
                assert tensor.dequant.quant in quantized.UNQUANTIZED_TYPES, \
                    "can't transpose a quantized tensor (type %s)" % tensor.dequant.quant.name
                data = data.t().contiguous()
        else:
            data_f32 = DequantizeParams(
                quant = tensor.dequant.quant,
                shape = tensor.dequant.shape,
                dtype = torch.float32,
            ).apply(tensor.data)
            if transpose:
                data_f32 = data_f32.t()
            data_f32 = data_f32.contiguous()

            quant = quant_repair.gguf.GGMLQuantizationType.Q8_0
            assert (
                data_f32.shape[-1] % torch_ggml_quant.quant_format_values_per_block(quant) == 0
            ), "can't quantize lora - not a multiple of quant block size"
            print('quantize %s to format %s' % (name, quant.name))
            data_f32 = data_f32.view(-1, torch_ggml_quant.quant_format_values_per_block(quant))
            data = torch.empty(
                (data_f32.shape[0], torch_ggml_quant.quant_format_block_size(quant)),
                dtype = torch.uint8,
                device = 'cpu',
            )
            torch_ggml_quant.quantize_fp32_cpu(data_f32, data, quant)
            data = data.view(-1)

            type_ = quant

        writer.add_tensor(name, dimensions, type_)
        tensor_data[name] = data

    def add_lora(
        name: str,
        params: QRF.LowRankAdapterParams,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ):
        add_tensor(name + '.lora_a', params.lora_a.map(lambda t: t * params.lora_alpha),
            transpose = transpose_a, quantize = quantize_lora)
        add_tensor(name + '.lora_b', params.lora_b,
            transpose = transpose_b, quantize = quantize_lora)

    add_lora('token_embd.weight', train_params.tok_embeddings, transpose_a = True)
    for i, layer in enumerate(train_params.layers):
        add_lora('blk.%d.attn_q.weight' % i, layer.q_proj)
        add_lora('blk.%d.attn_k.weight' % i, layer.k_proj)
        add_lora('blk.%d.attn_v.weight' % i, layer.v_proj)
        add_lora('blk.%d.attn_output.weight' % i, layer.output_proj)
        add_lora('blk.%d.ffn_gate.weight' % i, layer.gate_proj)
        add_lora('blk.%d.ffn_down.weight' % i, layer.down_proj)
        add_lora('blk.%d.ffn_up.weight' % i, layer.up_proj)
        add_tensor('blk.%d.attn_norm.weight' % i, layer.sa_norm.scale)
        add_tensor('blk.%d.ffn_norm.weight' % i, layer.mlp_norm.scale)
    add_tensor('output_norm.weight', train_params.norm.scale)
    add_lora('output.weight', train_params.output)


    writer.finish_header()


    # Write tensor data

    # Copy all the tensor data from the quantized GGUF.  We copy the norm
    # weights even though they're unused because they're small and finding
    # the end of each tensor's data requires additional logic.
    writer.copy_tensor_data(reader, copied_tensors)

    for name, data in tensor_data.items():
        writer.write_tensor_data(name, data.view(torch.uint8).numpy())


    writer.finish_data()




def main():
    assert len(sys.argv) >= 2
    mode = sys.argv[1]

    try:
        if mode == 'init':
            assert len(sys.argv) == 4
            config_path = sys.argv[2]
            checkpoint_path = sys.argv[3]
            run_init(config_path, checkpoint_path)
        elif mode == 'train':
            assert len(sys.argv) == 3
            checkpoint_path = sys.argv[2]
            run_train(checkpoint_path)
        elif mode == 'init_and_train':
            assert len(sys.argv) == 4
            config_path = sys.argv[2]
            checkpoint_path = sys.argv[3]
            run_init(config_path, checkpoint_path)
            run_train(checkpoint_path)
        elif mode == 'extract_config':
            assert len(sys.argv) == 3
            checkpoint_path = sys.argv[2]
            run_extract_config(checkpoint_path)
        elif mode == 'update_config':
            assert len(sys.argv) == 4
            checkpoint_path = sys.argv[2]
            config_path = sys.argv[3]
            run_update_config(checkpoint_path, config_path)
        elif mode == 'export_gguf':
            assert len(sys.argv) == 4
            checkpoint_path = sys.argv[2]
            gguf_path = sys.argv[3]
            run_export_gguf(checkpoint_path, gguf_path)
        elif mode == 'export_gguf_quantized':
            assert len(sys.argv) == 4
            checkpoint_path = sys.argv[2]
            gguf_path = sys.argv[3]
            run_export_gguf(checkpoint_path, gguf_path, quantize_lora = True)
        else:
            raise ValueError('unknown mode %r' % mode)
    finally:
        MEMORY_ACCOUNTING.report('on exit/crash')


if __name__ == '__main__':
    main()
