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
from typing import Optional, Dict, Set
import torch
from torch import Tensor
from torch import nn
from torchtune.models.llama3 import llama3_tokenizer_transformers
from torchtune.modules import quantized, lr_schedulers
from torchtune.utils import set_default_dtype
from gguf import GGMLQuantizationType
from quant_repair.architecture import Llama3Arch
from quant_repair.common import build_module, load_weights
from quant_repair.datasets import load_slimorca_dataset
from quant_repair.forward import SuperbatchEmbeddings, sized_chunks
from quant_repair.modules import LowRankAdapter, QuantLowRankAdapter, WithAdapter
from quant_repair.weights import load_weights_safetensors_hf, \
    CheckpointStateDict, QuantizedCheckpointLoader

LAYER_MODULES_LINEAR = (
    'attn.q_proj',
    'attn.k_proj',
    'attn.v_proj',
    'attn.output_proj',
    'mlp.w1',
    'mlp.w2',
    'mlp.w3',
)

LAYER_MODULES_NORM = (
    'sa_norm',
    'mlp_norm',
)

LAYER_PARAMETERS = tuple('%s.weight' % name for name in LAYER_MODULES_LINEAR) + \
    tuple('%s.scale' % name for name in LAYER_MODULES_NORM)


@contextmanager
def record_time(dest, key):
    start = time.time()
    yield
    end = time.time()
    dest[key] += end - start

def main():
    with set_default_dtype(torch.bfloat16):
        #with torch.no_grad():
            run()

def run():
    assert len(sys.argv) == 4
    orig_dir = sys.argv[1]
    quant_dir = sys.argv[2]
    train_layer_index = int(sys.argv[3])

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

    print('creating forward pass modules')
    fwd_tok_embeddings = build_module(arch, 'tok_embeddings', device=device)
    fwd_tok_embeddings.requires_grad_(False)
    fwd_layer = build_module(arch, 'layer', device=device)
    fwd_layer.requires_grad_(False)

    def init_fwd_tok_embeddings(loader):
        load_weights(loader, fwd_tok_embeddings, 'tok_embeddings')

    def init_fwd_layer(loader, layer_index):
        load_weights(loader, fwd_layer, 'layers.%d' % layer_index)


    print('creating training module')

    lora_rank = 32
    #lora_quant = GGMLQuantizationType.Q6_K

    train_module = build_module(
        arch,
        'layer',
        layer_index = 0,
        loader = quant_weights,
        base_quant = False,
        lora_rank = lora_rank,
        lora_quant = False,
        device = device,
    )

    train_lora_only = True
    if train_lora_only:
        train_module.requires_grad_(False)
        train_params = []
        for name, param in train_module.named_parameters():
            if '.adapter.' in name:
                param.requires_grad_(True)
                train_params.append(param)
    else:
        train_module.requires_grad_(True)
        train_params = list(train_module.parameters())

    load_weights(quant_weights, train_module, 'layers.%d' % train_layer_index)


    # Training config
    max_seq_len = 1024
    batch_size = 4
    total_epochs = 1
    max_steps_per_epoch = 2000
    gradient_accumulation_steps = 2

    max_samples = gradient_accumulation_steps * max_steps_per_epoch

    superbatch_mem_gb = 32

    # Set up dataset and loader
    sampler, dataloader = load_slimorca_dataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        train_on_input=True,
        seed=0,
        batch_size=batch_size,
    )

    # Optimizer, learning rate schedule, and loss function
    optimizer = torch.optim.AdamW(
        train_params,
        lr = 1.0e-6 * math.sqrt(1 + train_layer_index),
    )
    lr_scheduler = lr_schedulers.get_exponential_schedule(
        optimizer,
        start_factor = 1.0,
        end_factor = 0.5,
        num_training_steps = total_epochs * max_steps_per_epoch,
    )
    loss_fn = nn.MSELoss()


    # Training loop
    print('training layer %d' % train_layer_index)
    log_file = open('train_%d.log' % time.time(), 'w')
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
        samples_iter = iter(pbar := tqdm(samples_iter,
            total=max_samples, desc='samples', position=0, leave=True))
        samples_processed = 0

        embeds_orig = SuperbatchEmbeddings(arch, ram_gb=superbatch_mem_gb // 2)
        embeds_quant = SuperbatchEmbeddings(arch, ram_gb=superbatch_mem_gb // 2)
        superbatch_limit = embeds_orig.tokens_free()

        for superbatch_samples in sized_chunks(samples_iter, superbatch_limit,
                lambda t: t.numel()):
            embeds_orig.clear()
            embeds_quant.clear()

            orig_layers_iter = tqdm(range(train_layer_index + 1),
                desc='orig layers', position=1, leave=False)
            quant_layers_iter = tqdm(range(train_layer_index),
                desc='quant layers', position=2, leave=False)
            superbatch_tqdm_kwargs = dict(desc='superbatch', position=3, leave=False)

            with record_time(metrics, 'orig_time'):
                init_fwd_tok_embeddings(orig_weights)
                for sample in tqdm(superbatch_samples, **superbatch_tqdm_kwargs):
                    y = fwd_tok_embeddings(sample.to(device))
                    embeds_orig.append(y)
                for layer_index in orig_layers_iter:
                    init_fwd_layer(orig_weights, layer_index)
                    embeds_orig.apply(fwd_layer, device=device, tqdm_kwargs=superbatch_tqdm_kwargs)

            with record_time(metrics, 'quant_time'):
                init_fwd_tok_embeddings(quant_weights)
                for sample in tqdm(superbatch_samples, **superbatch_tqdm_kwargs):
                    y = fwd_tok_embeddings(sample.to(device))
                    embeds_quant.append(y)
                for layer_index in quant_layers_iter:
                    init_fwd_layer(quant_weights, layer_index)
                    embeds_quant.apply(fwd_layer, device=device, tqdm_kwargs=superbatch_tqdm_kwargs)


            # Train using the collected embeddings.

            with record_time(metrics, 'train_time'):
                for i in tqdm(range(len(superbatch_samples)), desc='train',
                        position=3, leave=False):
                    quant_input = embeds_quant[i].to(device)
                    orig_output = embeds_orig[i].to(device)
                    train_output = train_module(quant_input)

                    loss = loss_fn(train_output, orig_output)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    samples_processed += 1
                    if samples_processed % gradient_accumulation_steps == 0:
                        with record_time(metrics, 'opt_time'):
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            lr_scheduler.step()

                    pbar.set_description(
                        f"{curr_epoch+1}|{samples_processed}|Loss: {loss.item():.6e}"
                    )

                    metrics['loss'] = loss.item()
                    metrics['lr'] = optimizer.param_groups[0]["lr"]
                    metrics['gpu_resources'] = torch.cuda.memory_allocated()
                    json.dump(metrics, log_file)
                    log_file.write('\n')
                    log_file.flush()


    train_module_key = 'layer%d' % train_layer_index
    checkpoint_file_name = quant_weights.next_checkpoint_file(train_module_key)
    checkpoint_path = os.path.join(quant_dir, checkpoint_file_name)
    state_dict = train_module.state_dict()
    if os.path.exists(checkpoint_path):
        old_checkpoint_path = '%s.old.%d' % (checkpoint_path, time.time())
        os.rename(checkpoint_path, old_checkpoint_path)
        tqdm.write('renamed %s -> %s to avoid overwrite' %
            (checkpoint_path, old_checkpoint_path))
    torch.save(state_dict, checkpoint_path)
    tqdm.write('saved %s' % checkpoint_path)


if __name__ == '__main__':
    main()
