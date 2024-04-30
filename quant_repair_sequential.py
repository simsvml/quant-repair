"""
Sequential training, in which we train layer 0, then use the new layer 0 to
train layer 1, and so on.  This is not parallelizable, but should produce
better results.
"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
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
    # `Llama3Arch.make_module`, where `fmt` is the layer's `quant_map` sorted
    # and flattened into a tuple.  (For the original model, all the quant modes
    # are `F16`.)  For `weights_desc`, `orig_layer_info` has a parameter name
    # prefix for looking up in `orig_state_dict`, and `quant_layer_info` has
    # the filename from `quant_module_files` where the weights can be read.
    orig_layer_info = []
    quant_layer_info = []
    def add_layer_info(kind, param_names, param_name_prefix=None, quant_name=None):
        if param_name_prefix is None:
            param_name_prefix = kind + '.'
        if quant_name is None:
            quant_name = kind

        layer_quant_map = {name: quant_map[param_name_prefix + name] for name in param_names}
        quant_fmt = tuple(sorted(layer_quant_map.items()))
        quant_key = (kind, quant_fmt)
        quant_file_name, _ = quant_module_files[quant_name]

        orig_layer_quant_map = {name: GGMLQuantizationType.F16 for name in param_names}
        orig_fmt = tuple(sorted(orig_layer_quant_map.items()))
        orig_key = (kind, orig_fmt)

        orig_layer_info.append((orig_key, param_name_prefix))
        quant_layer_info.append((quant_key, quant_file_name))
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
        quant_file_name = weights_desc
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

    def quant_module_state_dict(key, weights_desc):
        quant_file_name = weights_desc
        return torch.load(os.path.join(quant_dir, quant_file_name))

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
    train_key, train_weights_desc = quant_layer_info[1 + train_layer_index]
    train_module = arch.make_module(train_key).to(device)
    train_module.load_state_dict(quant_module_state_dict(train_key, train_weights_desc))
    train_module.requires_grad_(True)

    train_checkpoint_index = quant_module_files['layer%d' % train_layer_index][1]


    # Training config
    max_seq_len = 1024
    batch_size = 8
    total_epochs = 1
    max_steps_per_epoch = 100
    gradient_accumulation_steps = 16

#    TODO
#    superbatch_mem_gb = 2
#    # When processing a superbatch, we have two output tensors and one
#    # temporary tensor, each containing one embedding per token.
#    superbatch_bytes_per_token = 3 * arch.embed_dim * torch.get_default_dtype().itemsize
#    superbatch_tokens = superbatch_mem_gb * 1024 ** 3 // superbatch_bytes_per_token
#    superbatch_size = superbatch_tokens // max_seq_len
#    print('forward pass superbatch size: %d' % superbatch_size)


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

        max_samples = gradient_accumulation_steps * max_steps_per_epoch
        for idx, batch in enumerate(pbar := tqdm(dataloader, total=max_samples)):
            if (
                max_steps_per_epoch is not None
                and (idx // gradient_accumulation_steps) == max_steps_per_epoch
            ):
                break

            input_ids, labels = batch
            input_ids = input_ids.to(device)

            with record_time(metrics, 'quant_time'):
                quant_input = run_quant(1 + train_layer_index, input_ids)
            with record_time(metrics, 'orig_time'):
                orig_output = run_orig(1 + train_layer_index + 1, input_ids)

            with record_time(metrics, 'train_time'):
                train_output = train_module(quant_input)

                loss = loss_fn(train_output, orig_output)
                loss = loss / gradient_accumulation_steps
                loss.backward()

            pbar.set_description(
                f"{curr_epoch+1}|{idx+1}|Loss: {loss.item():.6e}"
            )

            if (idx + 1) % gradient_accumulation_steps == 0:
                with record_time(metrics, 'opt_time'):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()

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
            print('renamed %s -> %s to avoid overwrite' % (checkpoint_path, old_checkpoint_path))
        torch.save(state_dict, checkpoint_path)
        print('saved %s' % checkpoint_path)

    return



    quant_reader = GGUFReader(gguf_path)


    # Build the list of layer keys.
    quant_map = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
    quant_map = convert_weights.gguf_to_tune(quant_map)

    # `layer_info` stores `(key, weights_desc)` pairs for use with the
    # `ModuleCache`.  For `key`, we use `(kind, fmt)` as expected by
    # `Llama3Arch.make_module`, where `fmt` is the layer's `quant_map` sorted
    # and flattened into a tuple.  For `weights_desc`, we store the string that
    # should be prefixed to the parameter names in `fmt` to get the names of
    # the weights to read from the model file.
    layer_info = []
    def add_layer_info(kind, param_names, weights_desc=None):
        if weights_desc is None:
            weights_desc = kind + '.'
        layer_quant_map = {name: quant_map[weights_desc + name] for name in param_names}
        fmt = tuple(sorted(layer_quant_map.items()))
        key = (kind, fmt)
        layer_info.append((key, weights_desc))
    add_layer_info('tok_embeddings', ('weight',))
    for i in range(arch.num_layers):
        add_layer_info('layer', LAYER_PARAMETERS, 'layers.%d.' % i)
    add_layer_info('norm', ('scale',))
    add_layer_info('output', ('weight',))

    pprint(layer_info)


    # Helpers for obtaining module layers
    module_cache = ModuleCache()

    def get_state_dict(key, weights_desc):
        kind, fmt = key
        # After conversion, this will map the full name as used in the GGUF to
        # the short name (scoped to the current layer) expected in the output.
        layer_name_map = convert_weights.tune_to_gguf(
            dict((weights_desc + name, name) for name, quant in fmt))
        state_dict = {}
        for gguf_name, tune_name in layer_name_map.items():
            state_dict.update(gguf_load_tensor_unpacked(reader, gguf_name, tune_name))
        print('state dict keys for %s, %s = %s' % (key, weights_desc, list(state_dict.keys())))
        return state_dict

    def layer_module(info) -> nn.Module:
        key, weights_desc = info
        return module_cache.get_module(
            key,
            weights_desc,
            lambda key: arch.make_module(key).to(device),
            get_state_dict,
        )

    #print('loading', layer_info[3])
    #m = layer_module(layer_info[3])

    print('test run')
    tokens = tokenizer.encode('Hello, my name is', add_eos=False)
    print(tokens)

    x = torch.tensor([tokens], device=device, dtype=torch.int)
    for info in layer_info:
        print('\nrunning %s' % (info,))
        m = layer_module(info)
        x = m(x)

    print('\ndone')
    print(x[0, -1])
    print(top_tokens(tokenizer, x[0, -2], top_k=10))
    print(top_tokens(tokenizer, x[0, -1], top_k=10))


if __name__ == '__main__':
    main()
