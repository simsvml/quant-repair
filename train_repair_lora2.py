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
from torchtune.models.llama3 import llama3_tokenizer_transformers
from torchtune.modules import RotaryPositionalEmbeddings
from gguf import GGUFReader
from tqdm import tqdm
from quant_repair.architecture import Llama3Arch
from quant_repair.datasets import load_slimorca_dataset, load_wikitext_dataset
from quant_repair.forward import SuperbatchEmbeddings
from quant_repair import functional as QRF
from quant_repair.memory_accounting import MEMORY_ACCOUNTING
from quant_repair import model_util as QRM
from quant_repair.model_util.llama3_lora import TrainableParams, LayerTrainableParams
from quant_repair.model_util.superbatch import run_forward_superbatch
from quant_repair.offload import TensorOffload
from quant_repair.quantized import DequantizeParams
from quant_repair.weights import load_weights_safetensors_hf, CheckpointStateDict


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
        return load_slimorca_dataset(
            tokenizer=tokenizer,
            max_seq_len=cfg.train.max_seq_len,
            train_on_input=True,
            seed=cfg.dataset.shuffle_seed,
            batch_size=1,
        )
    elif name == 'wikitext':
        return load_wikitext_dataset(
            tokenizer = tokenizer,
            max_seq_len = cfg.train.max_seq_len,
            seed = cfg.dataset.shuffle_seed,
            batch_size = 1,
        )
    else:
        raise ValueError('unknown dataset %r' % (name,))


@dataclass
class ModelCommon:
    rope: RotaryPositionalEmbeddings

def make_model(
    arch,
    common: ModelCommon,
    make_linear = QRF.Linear,
    make_embedding = QRF.Embedding,
) -> QRF.TransformerDecoder:
    return QRF.TransformerDecoder(
        tok_embeddings = make_embedding(),
        layers = [
            QRF.TransformerDecoderLayer(
                attn = QRF.CausalSelfAttention(
                    embed_dim = arch.embed_dim,
                    num_heads = arch.num_heads,
                    num_kv_heads = arch.num_kv_heads,
                    head_dim = arch.head_dim(),
                    q_proj = make_linear(),
                    k_proj = make_linear(),
                    v_proj = make_linear(),
                    output_proj = make_linear(),
                    pos_embeddings = common.rope,
                ),
                mlp = QRF.FeedForward(
                    gate_proj = make_linear(),
                    down_proj = make_linear(),
                    up_proj = make_linear(),
                    activation = nn.SiLU(),
                ),
                sa_norm = QRF.RMSNorm(eps = arch.norm_eps),
                mlp_norm = QRF.RMSNorm(eps = arch.norm_eps),
            ) for i in range(arch.num_layers)
        ],
        norm = QRF.RMSNorm(eps = arch.norm_eps),
        output = make_linear(),
    )


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

    if os.path.exists(checkpoint_path):
        print('Output file %r already exists' % checkpoint_path)
        answer = input('Overwrite? ')
        if answer.lower() not in ('y', 'yes'):
            print('Operation cancelled')
            sys.exit(1)


    arch = get_model_arch(cfg.model_arch)
    device = torch.device('cuda')
    torch.set_default_dtype(torch.bfloat16)


    gguf_reader = GGUFReader(cfg.quant_weights_gguf_path)
    gguf_tensors = {t.name: t for t in gguf_reader.tensors}

    def gguf_get(key):
        t = gguf_tensors[key]

        shape_length = max((j + 1 for j, dim in enumerate(t.shape) if dim != 1),
            default=len(t.shape))
        shape = tuple(int(x) for x in reversed(t.shape[:shape_length]))

        dq = DequantizeParams(
            quant = t.tensor_type,
            shape = shape,
        )

        x = torch.from_numpy(t.data).to(device)
        return dq.apply(x)


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
            lora_a = torch.empty((rank, n), dtype=dtype, device=device).normal_(),
            lora_b = torch.zeros((m, rank), dtype=dtype, device=device),
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
                gguf_get('blk.%d.attn_norm.weight' % layer_index)),
            mlp_norm = QRF.RMSNormParams(
                gguf_get('blk.%d.ffn_norm.weight' % layer_index)),
        )

    def init_train_params() -> TrainableParams:
        return TrainableParams(
            tok_embeddings = init_lora_params('tok_embeddings'),
            layers = [init_layer_params(i) for i in range(arch.num_layers)],
            norm = QRF.RMSNormParams(gguf_get('output_norm.weight')),
            output = init_lora_params('output'),
        )

    train_params = init_train_params()

    for tensor in train_params.tensors():
        tensor.requires_grad_(True)


    # Initialize optimizer and LR schedule
    optimizer = build_optimizer(cfg, list(train_params.tensors()))


    checkpoint_dict = {
        # TODO: Instead of saving `cfg_dict`, convert `cfg` to a dict and save
        # that.  `cfg` has default values filled in for all fields, and saving
        # those would ensure that changing the defaults in the script doesn't
        # break old checkpoints.
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
    checkpoint_dict = torch.load(checkpoint_path)
    cfg_dict = checkpoint_dict['cfg']
    cfg = Config.from_dict(cfg_dict)

    arch = get_model_arch(cfg.model_arch)
    device = torch.device('cuda')
    torch.set_default_dtype(torch.bfloat16)
    #MEMORY_ACCOUNTING.disable()
    MEMORY_ACCOUNTING.disable_print()


    print('loading original weights from %r' % (cfg.orig_weights_safetensors_dir,))
    orig_state_dict = load_weights_safetensors_hf(cfg.orig_weights_safetensors_dir, arch)
    orig_weights = CheckpointStateDict(orig_state_dict)


    # Offload setup
    print('loading quantized weights from %r' % (cfg.quant_weights_gguf_path,))
    gguf_reader = GGUFReader(cfg.quant_weights_gguf_path)
    gguf_tensors = {t.name: t for t in gguf_reader.tensors}
    gguf_quant_map = {}

    offload = TensorOffload(device, vram_limit_gb=cfg.memory.weights_vram_gb)

    offload.add_group('tok_embeddings')
    for i in range(arch.num_layers):
        offload.add_group('layers.%d' % i)
    offload.add_group('norm')
    offload.add_group('output')

    def offload_init_tensors():
        def add_weight(group, name, gguf_name):
            full_name = '%s.%s' % (group, name)

            tensor = gguf_tensors[gguf_name]
            shape_length = max((j + 1 for j, dim in enumerate(tensor.shape) if dim != 1),
                default=len(tensor.shape))
            shape = tuple(int(x) for x in reversed(tensor.shape[:shape_length]))
            gguf_quant_map[full_name] = DequantizeParams(
                quant = tensor.tensor_type,
                shape = shape,
            )

            torch_tensor = torch.from_numpy(tensor.data)
            offload.add_tensor(group, full_name, torch_tensor, read_only=True)

        add_weight('tok_embeddings', 'weight', 'token_embd.weight')
        for i in range(arch.num_layers):
            add_weight('layers.%d' % i, 'attn.q_proj.weight', 'blk.%d.attn_q.weight' % i)
            add_weight('layers.%d' % i, 'attn.k_proj.weight', 'blk.%d.attn_k.weight' % i)
            add_weight('layers.%d' % i, 'attn.v_proj.weight', 'blk.%d.attn_v.weight' % i)
            add_weight('layers.%d' % i, 'attn.output_proj.weight', 'blk.%d.attn_output.weight' % i)
            add_weight('layers.%d' % i, 'mlp.w1.weight', 'blk.%d.ffn_gate.weight' % i)
            add_weight('layers.%d' % i, 'mlp.w2.weight', 'blk.%d.ffn_down.weight' % i)
            add_weight('layers.%d' % i, 'mlp.w3.weight', 'blk.%d.ffn_up.weight' % i)
            add_weight('layers.%d' % i, 'sa_norm.scale', 'blk.%d.attn_norm.weight' % i)
            add_weight('layers.%d' % i, 'mlp_norm.scale', 'blk.%d.ffn_norm.weight' % i)
        add_weight('norm', 'scale', 'output_norm.weight')
        add_weight('output', 'weight', 'output.weight')
    offload_init_tensors()

    offload.build_partitions()

    # Use `offload` for fetching base model weights.
    quant_weights = offload


    tokenizer_json_path = os.path.join(cfg.orig_weights_safetensors_dir, 'tokenizer.json')
    print('loading tokenizer from %s' % tokenizer_json_path)
    tokenizer = llama3_tokenizer_transformers(tokenizer_json_path)


    print('loading params from checkpoint')
    param_weights = CheckpointStateDict(checkpoint_dict['params'])
    train_params = QRM.llama3_lora.load_trainable_params(param_weights, arch.num_layers, device)
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


    # Set up dataset and loader
    print('loading dataset')
    sampler, dataloader = build_dataset(cfg, tokenizer)


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


    model_common = ModelCommon(
        rope = RotaryPositionalEmbeddings(
            dim = arch.head_dim(),
            max_seq_len = arch.max_seq_len,
            base = arch.rope_base,
        ).to(device),
    )
    model = make_model(arch, model_common, QRF.Linear, QRF.Embedding)

    lora_dropout = cfg.model.lora_dropout

    def embedding_with_lora():
        base = QRF.QuantEmbedding()
        adapter = QRF.EmbeddingLowRankAdapter(dropout = lora_dropout)
        return QRF.WithAdapter(base, adapter)

    def linear_with_lora():
        base = QRF.QuantLinear()
        adapter = QRF.LowRankAdapter(dropout = lora_dropout)
        return QRF.WithAdapter(base, adapter)

    model_with_lora = make_model(arch, model_common, linear_with_lora, embedding_with_lora)


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


    sampler.set_epoch(0)
    samples_iter = (tokens for tokens, labels in dataloader)
    # TODO: better strategy for skipping initial steps
    samples_iter = itertools.islice(samples_iter, cfg.dataset.skip + progress['samples'], None)

    embeds_orig = SuperbatchEmbeddings(arch, ram_gb = cfg.memory.superbatch_ram_gb)
    superbatch_limit = embeds_orig.tokens_free()

    gradient_accumulation_steps = cfg.train.gradient_accumulation_steps

    pbar_tokens = tqdm(
        desc = 'tokens',
        total = cfg.train.total_tokens,
        initial = progress['tokens'],
        smoothing = 0,
        unit_scale = True,
    )
    pbar_super_tokens = tqdm(
        desc = 'superbatch tokens',
        total = cfg.train.total_tokens,
        initial = progress['tokens'],
        unit_scale = True,
    )
    pbar_super_forward = tqdm(desc = 'superbatch forward', total = 2 + arch.num_layers)
    pbar_super_layer = tqdm(desc = 'superbatch layer', total = 1)
    pbar_train_forward = tqdm(desc = 'train forward', total = 1)
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

        run_forward_superbatch(
            model,
            orig_weights,
            superbatch_samples,
            embeds_orig,
            device,
            pbar_forward = pbar_super_forward,
            pbar_layer = pbar_super_layer,
        )
        pbar_super_tokens.update(sum(s.numel() for s in superbatch_samples))

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
                    model_with_lora, quant_weights, train_params, device, gguf_quant_map)
                activations = [m(sample) for sample in samples]
                MEMORY_ACCOUNTING.register_all(activations, 'activations')
                save_activation_checkpoint(0, activations)
                del m
                pbar_train_forward.update()

                for i in range(arch.num_layers):
                    m = QRM.llama3_lora.build_trainable_layer(
                        model_with_lora, quant_weights, train_params, i, device, gguf_quant_map)
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
                    model_with_lora, quant_weights, train_params, device, gguf_quant_map)
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
            m_orig = QRM.llama3.build_forward_output(model, orig_weights, device)
            m_train = QRM.llama3_lora.build_trainable_output(
                model_with_lora, quant_weights, train_params, device, gguf_quant_map)

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
                model_with_lora, quant_weights, train_params, device, gguf_quant_map)
            for j in range(len(samples)):
                act = load_activation_checkpoint(arch.num_layers, j)
                gradients[j] = run_backward_step(act, gradients[j], m)
            MEMORY_ACCOUNTING.register_all(gradients, 'backward gradients')
            del m, act
            pbar_train_backward.update()

            for i in reversed(range(arch.num_layers)):
                m = QRM.llama3_lora.build_trainable_layer(
                    model_with_lora, quant_weights, train_params, i, device, gguf_quant_map)
                for j in range(len(samples)):
                    act = load_activation_checkpoint(i, j)
                    gradients[j] = run_backward_step(act, gradients[j], m)
                MEMORY_ACCOUNTING.register_all(gradients, 'backward gradients')
                del m, act
                pbar_train_backward.update()

            m = QRM.llama3_lora.build_trainable_tok_embeddings(
                model_with_lora, quant_weights, train_params, device, gguf_quant_map)
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
        else:
            raise ValueError('unknown mode %r' % mode)
    finally:
        MEMORY_ACCOUNTING.report('on exit/crash')


if __name__ == '__main__':
    main()
