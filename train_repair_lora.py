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

def init_lora_weights(module: LowRankAdapter, module_name: str) -> Dict[str, Tensor]:
    # Original LoRA paper initializes A to be normally distributed and
    # initializes B to zero.
    return {
        '%s.lora_a' % module_name: torch.normal(0, 1, module.lora_a.shape), 
        '%s.lora_b' % module_name: torch.zeros(module.lora_b.shape), 
    }


def _join_name(*args) -> str:
    """
    Join together parts of a module or parameter name with dots.  Empty
    components are omitted.
    """
    return '.'.join(x for x in args if x != '')

def load_module(
    loader,
    arch,
    kind: str,
    layer_index: Optional[int] = None,
    base_quant: bool = False,
    lora_rank: Optional[int] = None,
    lora_quant: bool = False,
    lora_quant_format: Optional[GGMLQuantizationType] = None,
    device = None,
) -> nn.Module:
    """
    Build a module for training and load its weights.

    Args:
        loader: used to load weights from a checkpoint.
        arch: model architecture to create a module for.
        kind: which module to create from `arch`.
        layer_index: when building a `layer` module, this gives the layer index
            whose weights should be loaded.
        base_quant: if set, use a quantized representation for the base weights.
        lora_rank: if set, add a LoRA of this rank.
        lora_quant: if set, use a quantized representation for the LoRA weights.
        lora_quant_format: if `lora_quant` is set but the `loader` doesn't have
            a quant format for a LoRA, this format is used instead.
    """

    lora = lora_rank is not None

    def convert_module_name(name: str) -> str:
        """
        Convert generic `layer.foo` to a specific name `layers.0.foo` (using
        `layer_index` for the index) as it will appear in the checkpoint.
        """
        if name.startswith('layer.'):
            _, rest = name.split('.', 1)
            return 'layers.%d.%s' % (layer_index, rest)
        else:
            return name

    def get_quant_type(
        module_name: str,
        param_name: str,
        for_lora: bool = False,
    ) -> Optional[GGMLQuantizationType]:
        if not for_lora:
            quant = loader.get_quant_type(_join_name(module_name, 'base', param_name))
            if quant is None:
                # Try again with the non-LoRA name.
                quant = loader.get_quant_type(_join_name(module_name, param_name))
            return quant
        else:
            quant = loader.get_quant_type(_join_name(module_name, 'adapter', param_name))
            if quant is None:
                quant = lora_quant_format
            return quant

    def quant_type(
        module_name: str,
        param_name: str,
        for_lora: bool = False,
    ) -> GGMLQuantizationType:
        quant = get_quant_type(module_name, param_name, for_lora)
        assert quant is not None, 'missing quant type for %s.%s (for_lora = %s)' % \
                (module_name, param_name, for_lora)
        return quant

    def embedding(name: str, num_embeddings, embedding_dim, device=None):
        name = convert_module_name(name)
        if not base_quant:
            base = nn.Embedding(num_embeddings, embedding_dim, device=device)
        else:
            base = quantized.QuantEmbedding(num_embeddings, embedding_dim,
                weight_quant=quant_type(name, 'weight'),
                device=device)
        if not lora:
            return base
        else:
            if not lora_quant:
                adapter = EmbeddingLowRankAdapter(num_embeddings, embedding_dim, lora_rank,
                    device=device)
            else:
                adapter = QuantEmbeddingLowRankAdapter(num_embeddings, embedding_dim, lora_rank,
                    lora_quant=quant_type(name, 'lora_a', for_lora=True),
                    device=device)
            return WithAdapter(base, adapter)

    def linear(name: str, in_features, out_features, bias=True, device=None):
        name = convert_module_name(name)
        if not base_quant:
            base = nn.Linear(in_features, out_features, bias=bias, device=device)
        else:
            base = quantized.QuantLinear(in_features, out_features, bias=bias,
                weight_quant=quant_type(name, 'weight'),
                bias_quant=get_quant_type(name, 'bias'),
                device=device)
        if not lora:
            return base
        else:
            if not lora_quant:
                adapter = LowRankAdapter(in_features, out_features, lora_rank,
                    device=device)
            else:
                adapter = QuantLowRankAdapter(in_features, out_features, lora_rank,
                    lora_quant=quant_type(name, 'lora_a', for_lora=True),
                    device=device)
            return WithAdapter(base, adapter)

    module = arch.make_module2(kind, linear=linear, embedding=embedding, device='meta') \
            .to_empty(device=device)
    return module


def load_weights(
    loader,
    module: nn.Module,
    prefix: str,
):
    # Examine module parameters to decide which weights to load.

    @dataclass
    class NeedWeights:
        # Original parameter names that should be loaded for this module.
        params: Set[str]
        # Whether the base weights should be quantized.
        quant_base: Optional[bool]
        # Whether the LoRA weights (if any) should be quantized.
        quant_lora: Optional[bool]
        # Whether the module has separate base and LoRA weights, or only base.
        has_lora: bool
    # Map from original module name (like `layers.0.mlp.w1`) to info about the
    # weights required for that module.
    need_weights = {}

    for param_name, _ in module.named_parameters():
        module_name, _, param_base_name = param_name.rpartition('.')

        need_quant = module_name.endswith('_quant')
        if need_quant:
            # `param_name` was originally like `foo.bar_quant.k_qs`, derived
            # from parameter name `foo.bar`.  Now `module_name` is something
            # like `foo.bar_quant`.
            module_name, _, param_base_name = module_name.removesuffix('_quant').rpartition('.')

        has_lora = False
        is_lora = False
        if module_name.endswith('.base'):
            module_name = module_name.removesuffix('.base')
            has_lora = True
        elif module_name.endswith('.adapter'):
            has_lora = True
            is_lora = True
            module_name = module_name.removesuffix('.adapter')
            # Don't include LoRA param names in `params`.
            assert param_base_name in ('lora_a', 'lora_b')
            param_base_name = None

        if module_name not in need_weights:
            need_weights[module_name] = NeedWeights(
                params = set(),
                quant_base = None,
                quant_lora = None,
                has_lora = has_lora,
            )

        nw = need_weights[module_name]
        if param_base_name is not None:
            nw.params.add(param_base_name)
        if not is_lora:
            if nw.quant_base is None:
                nw.quant_base = need_quant
            else:
                assert nw.quant_base == need_quant, \
                    'saw mix of quant and unquant base params for %r' % module_name
        else:
            if nw.quant_lora is None:
                nw.quant_lora = need_quant
            else:
                assert nw.quant_lora == need_quant, \
                    'saw mix of quant and unquant lora params for %r' % module_name
        assert nw.has_lora == has_lora, \
                'saw mix of lora and non-lora params for %r' % module_name

    #pprint(need_weights)


    # Load parameters
    state_dict = {}
    for module_name, nw in need_weights.items():
        for param_name in nw.params:
            assert nw.quant_base is not None, \
                    'got params, but quant_base is unset for %r' % module_name
            key_without_lora = _join_name(prefix, module_name, param_name)
            key_with_lora = _join_name(prefix, module_name, 'base', param_name)
            key = key_with_lora if loader.has(key_with_lora) else key_without_lora
            if not nw.has_lora:
                result_key = _join_name(module_name, param_name)
            else:
                result_key = _join_name(module_name, 'base', param_name)
            dequant = not nw.quant_base
            state_dict.update(loader.get(key, result_key, dequant))

        checkpoint_has_lora = loader.has(_join_name(prefix, module_name, 'adapter.lora_a'))
        if nw.has_lora:
            assert nw.quant_lora is not None, \
                    'module has_lora, but quant_lora is unset for %r' % module_name
            if checkpoint_has_lora:
                state_dict.update(loader.get_multi(
                    (
                        _join_name(prefix, module_name, 'adapter.lora_a'),
                        _join_name(prefix, module_name, 'adapter.lora_b'),
                    ),
                    (
                        _join_name(module_name, 'adapter.lora_a'),
                        _join_name(module_name, 'adapter.lora_b'),
                    ),
                    dequant = not nw.quant_lora,
                ))
            else:
                # There are no LoRA weights in the checkpoint, so initialize to
                # some defaults.
                assert not nw.quant_lora, \
                        'default initialization of quantized LoRA is not supported yet'
                lora_module_name = module_name + '.adapter'
                lora_module = module.get_submodule(lora_module_name)
                state_dict.update(init_lora_weights(lora_module, lora_module_name))
        else:
            if checkpoint_has_lora:
                # The checkpoint has separate base and LoRA weights, but the
                # model has only base weights.  Flatten the LoRA into the base.
                target_params = set(x for x in nw.params if x != 'bias')
                assert len(target_params) == 1, \
                    'multiple possible target params for merging %r lora: %r' % \
                        (module_name, target_params)
                target_param = target_params.pop()

                assert not nw.quant_base, \
                    "can't merge LoRA into quantized base for %r" % module_name

                lora_params = loader.get_multi(
                    (
                        _join_name(prefix, module_name, 'adapter.lora_a'),
                        _join_name(prefix, module_name, 'adapter.lora_b'),
                    ),
                    ('a', 'b'),
                    dequant = True,
                )
                with torch.no_grad():
                    weights = lora_params['b'] @ lora_params['a']
                    state_dict[_join_name(module_name, target_param)] += weights

    #pprint(list(state_dict.keys()))
    module.load_state_dict(state_dict)


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
    fwd_tok_embeddings = arch.make_module2('tok_embeddings', device='meta') \
            .requires_grad_(False).to_empty(device=device)
    fwd_layer = arch.make_module2('layer', device='meta') \
            .requires_grad_(False).to_empty(device=device)

    def init_fwd_tok_embeddings(loader):
        load_weights(loader, fwd_tok_embeddings, 'tok_embeddings')

    def init_fwd_layer(loader, layer_index):
        load_weights(loader, fwd_layer, 'layers.%d' % layer_index)


    print('creating training module')

    lora_rank = 32
    #lora_quant = GGMLQuantizationType.Q6_K

    train_module = load_module(
        quant_weights,
        arch,
        'layer',
        train_layer_index,
        base_quant = False,
        lora_rank = lora_rank,
        lora_quant = False,
        device = device,
    )

    load_weights(quant_weights, train_module, 'layers.0')


    # Training config
    max_seq_len = 1024
    batch_size = 4
    total_epochs = 1
    max_steps_per_epoch = 1000
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
        list(param for name, param in train_module.named_parameters() if '.adapter.' in name),
        lr = 1.0e-6 * math.sqrt(1 + train_layer_index),
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
