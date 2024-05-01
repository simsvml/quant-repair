"""
Sequential training, in which we train layer 0, then use the new layer 0 to
train layer 1, and so on.  This is not parallelizable, but should produce
better results.
"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import itertools
import json
import math
import os
from pprint import pprint
import re
import sys
from tempfile import TemporaryDirectory
import time
from typing import Optional, Any, Dict
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchtune.models import convert_weights
from torchtune.models.llama3 import llama3_8b, llama3_tokenizer_transformers
from torchtune.modules import quantized, lr_schedulers
from torchtune.modules.module_cache import ModuleCache, Llama3Arch
from torchtune.datasets import slimorca_dataset
from torchtune.utils import FullModelGGUFCheckpointer, set_default_dtype
from gguf import GGUFReader, GGMLQuantizationType
from torchtune.utils import gguf_quant
from torchtune.utils import padded_collate
import safetensors.torch
from tqdm import tqdm
from quant_repair.forward import SuperbatchEmbeddings, sized_chunks


LAYER_PARAMETERS = {
    'attn.k_proj.weight',
    'attn.output_proj.weight',
    'attn.q_proj.weight',
    'attn.v_proj.weight',
    'mlp.w1.weight',
    'mlp.w2.weight',
    'mlp.w3.weight',
    'mlp_norm.scale',
    'sa_norm.scale',
}

CHECKPOINT_FILE_RE = re.compile(r'([a-z0-9_]+?)(_ckpt([0-9]+))?\.pt$')


@contextmanager
def record_time(dest, key):
    start = time.time()
    yield
    end = time.time()
    dest[key] += end - start


def main():
    with set_default_dtype(torch.bfloat16):
        run()

def run():
    assert len(sys.argv) == 4
    orig_dir = sys.argv[1]
    quant_dir = sys.argv[2]
    train_layer_index = int(sys.argv[3])

    device = torch.device('cuda')
    arch = Llama3Arch.llama3_8b()

    print('loading original weights from %r' % (orig_dir,))
    orig_state_dict = {}
    for name in os.listdir(orig_dir):
        if not (name.startswith('model') and name.endswith('.safetensors')):
            continue
        # This will mmap the file.  Data will be paged in and out on
        # demand, so this effectively consumes no RAM.
        path = os.path.join(orig_dir, name)
        chunk = safetensors.torch.load_file(path, device='cpu')
        for key in chunk.keys():
            assert key not in orig_state_dict, \
                'duplicate tensor %s found in %s' % (key, path)
        orig_state_dict.update(chunk)
    orig_state_dict = convert_weights.hf_to_tune(
        orig_state_dict,
        num_heads=arch.num_heads,
        num_kv_heads=arch.num_kv_heads,
        dim=arch.embed_dim,
    )

    print('loading quantized weights from %r' % (quant_dir,))
    # Find the newest checkpoint for each module.  We don't load these files
    # until the data is needed.
    quant_module_files = {}
    for name in os.listdir(quant_dir):
        match = CHECKPOINT_FILE_RE.match(name)
        if match is None:
            continue
        key = match.group(1)
        version = match.group(3)
        if version is None:
            version = -1
        else:
            version = int(version)
        if key not in quant_module_files:
            insert = True
        else:
            old_version = quant_module_files[key][1]
            insert = version > old_version
        if insert:
            quant_module_files[key] = (name, version)

    quant_map = torch.load(os.path.join(quant_dir, 'quant_map.pt'))

    tokenizer_json_path = os.path.join(orig_dir, 'tokenizer.json')
    print('loading tokenizer from %s' % tokenizer_json_path)
    tokenizer = llama3_tokenizer_transformers(tokenizer_json_path)


    # Build layer info lists for running the two models.

    # These two lists store `(key, weights_desc)` pairs for use with the
    # `ModuleCache`.  For `key`, we use `(kind, fmt)` as expected by
    # `Llama3Arch.make_module`.  All quant modes in `fmt` are always `F16`,
    # since we dequantize weights when loaded to improve performance of the
    # forward pass.  For `weights_desc`, `orig_layer_info` has a parameter name
    # prefix for looking up in `orig_state_dict`, and `quant_layer_info` has
    # the filename from `quant_module_files` where the weights can be read
    # along with the quantized format (used to load and process the quantized
    # weights).
    #
    # Note that the two lists use the same `key`s, so they must use distinct
    # sets of `weights_desc`s to avoid collisions within the cache.
    orig_layer_info = []
    quant_layer_info = []
    def add_layer_info(kind, param_names, param_name_prefix=None, quant_name=None):
        if param_name_prefix is None:
            param_name_prefix = kind + '.'
        if quant_name is None:
            quant_name = kind

        orig_layer_quant_map = {name: GGMLQuantizationType.F16 for name in param_names}
        orig_fmt = tuple(sorted(orig_layer_quant_map.items()))
        key = (kind, orig_fmt)

        layer_quant_map = {name: quant_map[param_name_prefix + name] for name in param_names}
        quant_fmt = tuple(sorted(layer_quant_map.items()))
        quant_file_name, _ = quant_module_files[quant_name]

        orig_layer_info.append((key, param_name_prefix))
        quant_layer_info.append((key, (quant_file_name, quant_fmt)))
    add_layer_info('tok_embeddings', ('weight',))
    for i in range(arch.num_layers):
        add_layer_info(
            'layer',
            LAYER_PARAMETERS,
            param_name_prefix='layers.%d.' % i,
            quant_name='layer%d' % i,
        )
    add_layer_info('norm', ('scale',))
    add_layer_info('output', ('weight',))

    print('quantized module files:')
    for info in quant_layer_info:
        key, weights_desc = info
        quant_file_name, quant_fmt = weights_desc
        print('  ' + quant_file_name)


    # Helpers for obtaining forward-pass modules

    module_cache = ModuleCache(cache_size=8)

    def orig_module_state_dict(key, weights_desc):
        kind, fmt = key
        param_name_prefix = weights_desc
        state_dict = {}
        for name, _ in fmt:
            state_dict[name] = orig_state_dict[param_name_prefix + name]
        return state_dict

    def orig_layer_module(info) -> nn.Module:
        key, weights_desc = info
        return module_cache.get_module(
            key,
            weights_desc,
            lambda key: arch.make_module(key).requires_grad_(False).to(device),
            orig_module_state_dict,
        )

    def run_orig(num_modules: int, x: Tensor) -> Tensor:
        for info in orig_layer_info[:num_modules]:
            m = orig_layer_module(info)
            x = m(x)
        return x

    def load_quantized_state_dict(key, weights_desc):
        quant_file_name = weights_desc
        return torch.load(os.path.join(quant_dir, quant_file_name))

    def load_quantized_module(kind, quant_fmt, quant_file_name):
        return module_cache.get_module(
            (kind, quant_fmt),
            quant_file_name,
            lambda key: arch.make_module(key).requires_grad_(False).to(device),
            load_quantized_state_dict,
        )

    def quant_module_state_dict(key, weights_desc):
        kind, fmt = key
        quant_file_name, quant_fmt = weights_desc
        if all(quant in quantized.UNQUANTIZED_TYPES for _, quant in quant_fmt):
            # For modules that are already unquantized, just load the weights.
            return load_quantized_state_dict(key, quant_file_name)

        # Load the quantized version of the module, then dequantize it.
        quant_module = load_quantized_module(kind, quant_fmt, quant_file_name)

        state_dict = {name: None for name, _ in fmt}
        for name, param in quant_module.named_parameters():
            if name not in state_dict:
                continue
            state_dict[name] = param
        for name, value in state_dict.items():
            if value is not None:
                continue
            state_dict[name] = quant_module.get_submodule(name + '_quant').forward()
        assert not any(v is None for v in state_dict.values()), \
                'missing parameters: %s' % [k for k,v in state_dict.items() if v is None]
        return state_dict

    def quant_layer_module(info) -> nn.Module:
        key, weights_desc = info
        return module_cache.get_module(
            key,
            weights_desc,
            lambda key: arch.make_module(key).requires_grad_(False).to(device),
            quant_module_state_dict,
        )

    def run_quant(num_modules: int, x: Tensor) -> Tensor:
        for info in quant_layer_info[:num_modules]:
            m = quant_layer_module(info)
            x = m(x)
        return x


    # Build trainable layer
    def build_train_module():
        key, weights_desc = quant_layer_info[1 + train_layer_index]
        kind, fmt = key
        quant_file_name, quant_fmt = weights_desc
        quant_key = (kind, quant_fmt)
        m = arch.make_module(quant_key).to(device)
        m.load_state_dict(load_quantized_state_dict(quant_key, quant_file_name))
        m.requires_grad_(True)
        return m

    train_module = build_train_module()

    train_checkpoint_index = quant_module_files['layer%d' % train_layer_index][1]


    # Training config
    max_seq_len = 1024
    batch_size = 8
    total_epochs = 1
    max_steps_per_epoch = 100
    gradient_accumulation_steps = 8

    max_samples = gradient_accumulation_steps * max_steps_per_epoch

    superbatch_mem_gb = 32


    # Set up dataset and loader
    dataset = slimorca_dataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        train_on_input=True,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=0,
    )
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=partial(
            padded_collate,
            padding_idx=tokenizer.pad_id,
            #ignore_idx=loss_fn.ignore_index,
            ignore_idx=-100,
        ),
    )


    # Optimizer, learning rate schedule, and loss function
    optimizer = torch.optim.AdamW(
        train_module.parameters(),
        #lr = 1e-6 * math.sqrt(1 + train_layer_index),
        # Changed after layer 9
        lr = 1e-6 * math.sqrt(1 + train_layer_index),
    )
    lr_scheduler = lr_schedulers.get_exponential_schedule(
        optimizer,
        start_factor = 1.0,
        end_factor = 0.1,
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
    for curr_epoch in range(0, total_epochs):
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

            orig_infos = orig_layer_info[:1 + train_layer_index + 1]
            orig_infos_iter = tqdm(orig_infos, desc='orig layers', position=1, leave=False)
            quant_infos = quant_layer_info[:1 + train_layer_index]
            quant_infos_iter = tqdm(quant_infos, desc='quant layers', position=2, leave=False)
            superbatch_tqdm_kwargs = dict(desc='superbatch', position=3, leave=False)

            with record_time(metrics, 'orig_time'):
                for info in orig_infos_iter:
                    (kind, _), _ = info
                    m = orig_layer_module(info)
                    if kind == 'tok_embeddings':
                        for sample in tqdm(superbatch_samples, **superbatch_tqdm_kwargs):
                            y = m(sample.to(device))
                            embeds_orig.append(y)
                    else:
                        embeds_orig.apply(m, device=device, tqdm_kwargs=superbatch_tqdm_kwargs)

            with record_time(metrics, 'quant_time'):
                for info in quant_infos_iter:
                    (kind, _), _ = info
                    m = quant_layer_module(info)
                    if kind == 'tok_embeddings':
                        for sample in tqdm(superbatch_samples, **superbatch_tqdm_kwargs):
                            y = m(sample.to(device))
                            embeds_quant.append(y)
                    else:
                        embeds_quant.apply(m, device=device, tqdm_kwargs=superbatch_tqdm_kwargs)


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

        train_checkpoint_index += 1
        state_dict = train_module.state_dict()
        checkpoint_file_name = 'layer%d_ckpt%d.pt' % (train_layer_index, train_checkpoint_index)
        checkpoint_path = os.path.join(quant_dir, checkpoint_file_name)
        if os.path.exists(checkpoint_path):
            old_checkpoint_path = '%s.old.%d' % (checkpoint_path, time.time())
            os.rename(checkpoint_path, old_checkpoint_path)
            tqdm.write('renamed %s -> %s to avoid overwrite' %
                (checkpoint_path, old_checkpoint_path))
        torch.save(state_dict, checkpoint_path)
        tqdm.write('saved %s' % checkpoint_path)


if __name__ == '__main__':
    main()
