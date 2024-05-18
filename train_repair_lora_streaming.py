from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import json
import math
import os
from pprint import pprint
import sys
import time
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Set, Any, Union, Iterable, Callable
import weakref
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchtune.models.llama3 import llama3_tokenizer_transformers
from torchtune.modules import quantized, lr_schedulers
from torchtune.modules import RotaryPositionalEmbeddings
from torchtune.utils import set_default_dtype
from gguf import GGMLQuantizationType
from quant_repair.architecture import Llama3Arch
from quant_repair.common import build_module, load_weights, init_lora_weights
from quant_repair.datasets import load_slimorca_dataset
from quant_repair.forward import SuperbatchEmbeddings, sized_chunks
from quant_repair import functional as QRF
from quant_repair.memory_accounting import MEMORY_ACCOUNTING
from quant_repair import model_util as QRM
from quant_repair.model_util.misc import weights_getter
from quant_repair.model_util.llama3_lora import TrainableParams, LayerTrainableParams
from quant_repair.model_util.superbatch import run_forward_superbatch
from quant_repair.modules import LowRankAdapter, QuantLowRankAdapter, WithAdapter
from quant_repair.offload import TensorOffload
from quant_repair.weights import load_weights_safetensors_hf, \
    CheckpointStateDict, QuantizedCheckpointLoader


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


@contextmanager
def record_time(dest, key):
    start = time.time()
    yield
    end = time.time()
    dest[key] += end - start

def main1():
    with set_default_dtype(torch.bfloat16):
        #with torch.no_grad():
            run(None)

def main2():
    configs = []

    configs.append({
        'lora_rank': 32,
        'layer_lora_rank': [
            16 if i < 14 else 64 if i >= 32 - 7 else 32
            for i in range(32)
        ],
    })

    for r in (64, 16, 8, 128):
        configs.append({
            'lora_rank': r,
            'layer_lora_rank': [r] * 32,
        })

    with set_default_dtype(torch.bfloat16):
        #with torch.no_grad():
            for cfg in configs:
                try:
                    run(cfg)
                except Exception as e:
                    print('error: %s' % (e,))
                    print('config = %r' % (cfg,))
                    # Ensure the next run has a distinct timestamp, even if the
                    # current run failed in under 1 second.
                    time.sleep(2)

def main3():
    configs = []

    def add(override_rank, lr_exponent):
        configs.append(dict(
            lr = 1e-6 * 2 ** lr_exponent,
            override_rank = override_rank,
        ))

    add(None, 2)
    add(None, 2.5)
    add(None, 1.5)

    add(32, 2.5)
    add(32, 1.5)

    add(None, 1)
    add(None, 3)

    #add(None, -1)
    #add(None, 0)
    #add(None, 1)
    #add(None, 2)
    #add(None, 3)
    #add(None, 4)
    #add(None, 5)

    #add(32, -1)
    #add(32, 0)
    #add(32, 1)
    #add(32, 2)
    #add(32, 3)
    #add(32, 4)
    #add(32, 5)

    #for override_rank in (None, 32):
    #    for i in (-1, 0, 1, 2, 4, 5):

    with set_default_dtype(torch.bfloat16):
        #with torch.no_grad():
            for cfg in configs:
                try:
                    run(cfg)
                except Exception as e:
                    print('error: %s' % (e,))
                    print('config = %r' % (cfg,))
                    # Ensure the next run has a distinct timestamp, even if the
                    # current run failed in under 1 second.
                    time.sleep(2)

main = main3

def run(cfg):
    assert len(sys.argv) == 4
    orig_dir = sys.argv[1]
    quant_dir = sys.argv[2]
    rank_map_path = sys.argv[3]

    MEMORY_ACCOUNTING.disable()

    device = torch.device('cuda')
    arch = Llama3Arch.llama3_8b()

    print('loading original weights from %r' % (orig_dir,))
    orig_state_dict = load_weights_safetensors_hf(orig_dir, arch)
    orig_weights = CheckpointStateDict(orig_state_dict)

    print('loading quantized weights from %r' % (quant_dir,))
    quant_weights = QuantizedCheckpointLoader(quant_dir, dequant_device=device)

    tokenizer_json_path = os.path.join(orig_dir, 'tokenizer.json')
    print('loading tokenizer from %s' % tokenizer_json_path)
    tokenizer = llama3_tokenizer_transformers(tokenizer_json_path)


    # Build llama3 model
    rope = RotaryPositionalEmbeddings(
        dim = arch.head_dim(),
        max_seq_len = arch.max_seq_len,
        base = arch.rope_base,
    ).to(device)
    def make_model(make_linear, make_embedding):
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
                        pos_embeddings = rope,
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

    model = make_model(QRF.Linear, QRF.Embedding)

    def embedding_with_lora():
        base = QRF.Embedding()
        adapter = QRF.EmbeddingLowRankAdapter()
        return QRF.WithAdapter(base, adapter)

    def linear_with_lora():
        base = QRF.Linear()
        adapter = QRF.LowRankAdapter()
        return QRF.WithAdapter(base, adapter)

    model_with_lora = make_model(linear_with_lora, embedding_with_lora)


    print('initializing trainable parameters')

    # Norm scales are initialized from the base model's weights.  LoRA
    # parameters are initialized to default values.

    #lora_rank = 64
    lora_alpha = 32

    #layer_lora_rank = []
    #for i in range(arch.num_layers):
    #    #if i < 14:
    #    #    layer_lora_rank.append(lora_rank // 2)
    #    #elif i >= arch.num_layers - 7:
    #    #    layer_lora_rank.append(lora_rank * 2)
    #    #else:
    #    #    layer_lora_rank.append(lora_rank)
    #    layer_lora_rank.append(lora_rank)

    #lora_rank = cfg['lora_rank']
    #layer_lora_rank = cfg['layer_lora_rank']

    dims = QRM.llama3_lora.linear_dimensions(arch)
    rank_map = json.load(open(rank_map_path))

    if cfg['override_rank'] is not None:
        for key in rank_map:
            rank_map[key] = cfg['override_rank']

    def init_lora_params(
        key: str,
        # TODO: Try float16 for easier llama.cpp compat
        dtype: torch.dtype = torch.bfloat16,
    ) -> QRF.LowRankAdapterParams:
        n, m = dims.get(key)
        rank = rank_map[key]
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

        get1 = weights_getter(quant_weights, device)

        return LayerTrainableParams(
            q_proj = init_lora_params('layers.%d.attn.q_proj' % layer_index),
            k_proj = init_lora_params('layers.%d.attn.k_proj' % layer_index),
            v_proj = init_lora_params('layers.%d.attn.v_proj' % layer_index),
            output_proj = init_lora_params('layers.%d.attn.output_proj' % layer_index),
            gate_proj = init_lora_params('layers.%d.mlp.w1' % layer_index),
            down_proj = init_lora_params('layers.%d.mlp.w2' % layer_index),
            up_proj = init_lora_params('layers.%d.mlp.w3' % layer_index),
            sa_norm = QRF.RMSNormParams(
                get1('layers.%d.sa_norm.scale' % layer_index).to(torch.bfloat16)),
            mlp_norm = QRF.RMSNormParams(
                get1('layers.%d.mlp_norm.scale' % layer_index).to(torch.bfloat16)),
        )

    def init_train_params() -> TrainableParams:
        get1 = weights_getter(quant_weights, device)
        return TrainableParams(
            tok_embeddings = init_lora_params('tok_embeddings'),
            layers = [init_layer_params(i) for i in range(arch.num_layers)],
            norm = QRF.RMSNormParams(get1('norm.scale').to(torch.bfloat16)),
            output = init_lora_params('output'),
        )

    train_params = init_train_params()
    MEMORY_ACCOUNTING.register_params(train_params, 'trainable params')

    for tensor in train_params.tensors():
        tensor.requires_grad_(True)


    # Training config
    max_seq_len = 1024
    batch_size = 1
    total_epochs = 1
    max_steps_per_epoch = 1000
    #max_steps_per_epoch = 2500
    #max_steps_per_epoch = 1000
    #max_steps_per_epoch = 2500
    gradient_accumulation_steps = 16

    max_samples = gradient_accumulation_steps * max_steps_per_epoch

    #superbatch_mem_gb = 32
    superbatch_mem_gb = 8

    # Set up dataset and loader
    print('loading dataset')
    sampler, dataloader = load_slimorca_dataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        train_on_input=True,
        seed=0,
        batch_size=batch_size,
    )

    # Optimizer, learning rate schedule, and loss function

#    optimizer = torch.optim.AdamW(
#        list(train_params.tensors()),
#        lr = 2.5e-6,
#    )
#    lr_scheduler = lr_schedulers.get_exponential_schedule(
#        optimizer,
#        start_factor = 1.0,
#        end_factor = 0.1,
#        num_training_steps = total_epochs * max_steps_per_epoch,
#    )

    optimizer = torch.optim.AdamW(
        list(train_params.tensors()),
        lr = cfg['lr'],
        #weight_decay = 0.01,
        weight_decay = 0,
    )
#    lr_scheduler = lr_schedulers.get_cosine_schedule_with_warmup(
#        optimizer,
#        num_warmup_steps = 100,
#        num_training_steps = total_epochs * max_steps_per_epoch,
#    )
    lr_scheduler = lr_schedulers.get_linear_schedule(
        optimizer,
        start_factor = 1.0,
        end_factor = 0.1,
        num_training_steps = total_epochs * max_steps_per_epoch,
    )

    loss_fn = nn.MSELoss()
    kl_div_loss_fn = nn.KLDivLoss(log_target=True, reduction='batchmean')


    # Offload setup
    print('loading quantized weights for offload')
    offload = TensorOffload(device, vram_limit_gb=8)

    offload.add_group('tok_embeddings')
    for i in range(arch.num_layers):
        offload.add_group('layers.%d' % i)
    offload.add_group('norm')
    offload.add_group('output')

    def offload_init_tensors():
        get1 = weights_getter(quant_weights, 'cpu')
        def add_weight(group, name):
            full_name = '%s.%s' % (group, name)
            offload.add_tensor(group, full_name, get1(full_name), read_only=True)

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
    quant_weights = offload


    MEMORY_ACCOUNTING.report('before training loop')


    # Training loop
    print('training %d parameter tensors' % (len(list(train_params.tensors()))))

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    log_file = open('train_%s.log' % timestamp, 'w')
    def write_log(obj):
        json.dump(obj, log_file)
        log_file.write('\n')
        log_file.flush()
    config_dict = {
        'max_seq_len': max_seq_len,
        'batch_size': batch_size,
        'total_epochs': total_epochs,
        'max_steps_per_epoch': max_steps_per_epoch,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'lr': optimizer.defaults['lr'],
        'weight_decay': optimizer.defaults['weight_decay'],
        'scheduler': str(lr_scheduler),
        'lora_alpha': lora_alpha,
        'lora_rank_map': rank_map,
    }
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.LambdaLR):
        config_dict['lr_lambdas'] = [str(l) for l in lr_scheduler.lr_lambdas]
    write_log(config_dict)

    metrics = {
        'quant_time': 0.,
        'orig_time': 0.,
        'train_time': 0.,
        'opt_time': 0.,
        'loss': None,
        'lr': None,
        'gpu_resources': None,
    }
    for curr_epoch in range(total_epochs):
        # Update the sampler to ensure data is correctly shuffled across epochs
        # in case shuffle is True
        sampler.set_epoch(curr_epoch)

        samples_iter = (tokens for tokens, labels in dataloader)
        samples_iter = itertools.islice(samples_iter, max_samples)

        embeds_orig = SuperbatchEmbeddings(arch, ram_gb=superbatch_mem_gb)
        superbatch_limit = embeds_orig.tokens_free()


        pbar_samples = tqdm(desc='samples', total=max_samples, smoothing=0)
        pbar_superbatch = tqdm(desc='superbatch', total=max_samples)
        pbar_superbatch_forward = tqdm(desc='superbatch forward', total=2 + arch.num_layers)
        pbar_superbatch_layer = tqdm(desc='superbatch layer', total=1)
        pbar_train_forward = tqdm(desc='train forward', total=1)
        pbar_train_backward = tqdm(desc='train backward', total=1)

        total_tokens = 0

        for superbatch_samples in sized_chunks(samples_iter, superbatch_limit,
                lambda t: t.numel()):
            run_forward_superbatch(
                model,
                orig_weights,
                superbatch_samples,
                embeds_orig,
                device,
                pbar_forward = pbar_superbatch_forward,
                pbar_layer = pbar_superbatch_layer,
            )
            pbar_superbatch.update(len(superbatch_samples))

            MEMORY_ACCOUNTING.report('after forward superbatch')

            # Train using the collected embeddings.
            with record_time(metrics, 'train_time'):
                for samples_start in range(0, len(superbatch_samples), gradient_accumulation_steps):
                    samples = superbatch_samples[samples_start :
                        samples_start + gradient_accumulation_steps]
                    samples = [t.to(device).requires_grad_(False) for t in samples]
                    MEMORY_ACCOUNTING.register_all(samples, 'samples')

                    pbar_train_forward.reset(2 + arch.num_layers)
                    pbar_train_backward.reset(1 + 2 + arch.num_layers)

                    activation_checkpoints = []
                    with torch.no_grad():
                        m = QRM.llama3_lora.build_trainable_tok_embeddings(
                            model_with_lora, quant_weights, train_params, device)
                        activations = [m(sample) for sample in samples]
                        MEMORY_ACCOUNTING.register_all(activations, 'activations')
                        activation_checkpoints.append(activations)
                        pbar_train_forward.update()
                        del m

                        for i in range(arch.num_layers):
                            m = QRM.llama3_lora.build_trainable_layer(
                                model_with_lora, quant_weights, train_params, i, device)
                            activations = [m(act) for act in activations]
                            MEMORY_ACCOUNTING.register(activations, 'activations')
                            activation_checkpoints.append(activations)
                            pbar_train_forward.update()
                            del m

                        m = QRM.llama3_lora.build_trainable_norm(
                            model_with_lora, quant_weights, train_params, device)
                        activations = [m(act) for act in activations]
                        MEMORY_ACCOUNTING.register_all(activations, 'activations')
                        # Norm output is not recorded as a checkpoint.
                        pbar_train_forward.update()
                        del m

                        # Final activations are kept around for later use in
                        # calc_loss.

                    MEMORY_ACCOUNTING.report('after train forward pass')

                    # Run loss forward and backward
                    m_orig = QRM.llama3.build_forward_output(model, orig_weights, device)
                    m_train = QRM.llama3_lora.build_trainable_output(
                        model_with_lora, quant_weights, train_params, device)

                    # Sum of loss values (used only for logging).
                    loss_sum = 0.0
                    gradients = []
                    for i in range(len(samples)):
                        def calc_loss(train_embeds: Tensor) -> Tensor:
                            orig_logits = m_orig(embeds_orig[samples_start + i].to(device))
                            MEMORY_ACCOUNTING.register(orig_logits, 'orig_logits')
                            orig_log_prob = F.log_softmax(orig_logits, dim=-1)
                            MEMORY_ACCOUNTING.register(orig_log_prob, 'orig_log_prob')
                            orig_log_prob = orig_log_prob.view(-1, arch.vocab_size)
                            del orig_logits

                            train_logits = m_train(train_embeds)
                            MEMORY_ACCOUNTING.register(train_logits, 'train_logits')
                            train_log_prob = F.log_softmax(train_logits, dim=-1)
                            MEMORY_ACCOUNTING.register(train_log_prob, 'train_log_prob')
                            train_log_prob = train_log_prob.view(-1, arch.vocab_size)
                            del train_logits

                            loss = kl_div_loss_fn(train_log_prob, orig_log_prob)
                            MEMORY_ACCOUNTING.register(loss, 'loss')
                            return loss

                        train_embeds = activations[i]
                        loss, grad = run_initial_backward_step(train_embeds, calc_loss)
                        MEMORY_ACCOUNTING.register(grad, 'backward gradients')

                        gradients.append(grad)
                        loss_sum += loss.item()

                    loss_avg = loss_sum / len(samples)
                    del m_orig, m_train
                    pbar_train_backward.update()

                    # Run remaining backward steps.

                    m = QRM.llama3_lora.build_trainable_norm(
                        model_with_lora, quant_weights, train_params, device)
                    for i, act in enumerate(activation_checkpoints.pop()):
                        gradients[i] = run_backward_step(act, gradients[i], m)
                    MEMORY_ACCOUNTING.register_all(gradients, 'backward gradients')
                    del m, act
                    pbar_train_backward.update()

                    for i in reversed(range(arch.num_layers)):
                        m = QRM.llama3_lora.build_trainable_layer(
                            model_with_lora, quant_weights, train_params, i, device)
                        for j, act in enumerate(activation_checkpoints.pop()):
                            gradients[j] = run_backward_step(act, gradients[j], m)
                        MEMORY_ACCOUNTING.register_all(gradients, 'backward gradients')
                        del m, act
                        pbar_train_backward.update()

                    m = QRM.llama3_lora.build_trainable_tok_embeddings(
                        model_with_lora, quant_weights, train_params, device)
                    for i, sample in enumerate(samples):
                        run_final_backward_step(sample, gradients[i], m)
                    del m, sample, gradients
                    pbar_train_backward.update()

                    MEMORY_ACCOUNTING.report('after train backward pass')

                    with record_time(metrics, 'opt_time'):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        lr_scheduler.step()

                    pbar_samples.set_description(
                        f"Loss: {loss_avg:.6e}"
                    )
                    pbar_samples.update(len(samples))
                    total_tokens += sum(x.numel() for x in samples)

                    metrics['loss'] = loss_avg
                    metrics['lr'] = optimizer.param_groups[0]["lr"]
                    metrics['gpu_resources'] = torch.cuda.memory_allocated()
                    write_log(metrics)

        pbar_samples.close()
        pbar_superbatch.close()
        pbar_superbatch_forward.close()
        pbar_superbatch_layer.close()
        pbar_train_forward.close()
        pbar_train_backward.close()

        print('trained on %d tokens' % total_tokens)

    MEMORY_ACCOUNTING.report('after training loop')


    state_dict = {}

    def save_low_rank_adapter_params(name, params):
        state_dict[name + '.lora_a'] = params.lora_a
        state_dict[name + '.lora_b'] = params.lora_b
        state_dict[name + '.lora_alpha'] = params.lora_alpha

    def save_rms_norm_params(name ,params):
        state_dict[name + '.scale'] = params.scale

    def save_layer_trainable_params(name, params):
        save_low_rank_adapter_params(name + '.q_proj', params.q_proj)
        save_low_rank_adapter_params(name + '.k_proj', params.k_proj)
        save_low_rank_adapter_params(name + '.v_proj', params.v_proj)
        save_low_rank_adapter_params(name + '.output_proj', params.output_proj)
        save_low_rank_adapter_params(name + '.gate_proj', params.gate_proj)
        save_low_rank_adapter_params(name + '.down_proj', params.down_proj)
        save_low_rank_adapter_params(name + '.up_proj', params.up_proj)
        save_rms_norm_params(name + '.sa_norm', params.sa_norm)
        save_rms_norm_params(name + '.mlp_norm', params.mlp_norm)

    def save_trainable_params(name, params):
        save_low_rank_adapter_params(name + '.tok_embeddings', params.tok_embeddings)
        for i, layer_params in enumerate(params.layers):
            save_layer_trainable_params('%s.layers.%d' % (name, i), layer_params)
        save_rms_norm_params(name + '.norm', params.norm)
        save_low_rank_adapter_params(name + '.output', params.output)

    save_trainable_params('params', train_params)
    checkpoint_path = os.path.join(quant_dir, 'repair_ckpt_%s.pt' % timestamp)
    torch.save(state_dict, checkpoint_path)
    print('\n\nsaved %s' % checkpoint_path)


if __name__ == '__main__':
    main()
