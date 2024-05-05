"""
Misc functionality common to various scripts.
"""
from dataclasses import dataclass
from pprint import pprint
from typing import Optional, Dict, Set
import torch
from torch import Tensor
from torch import nn
from torchtune.modules import quantized
from gguf import GGMLQuantizationType
from quant_repair.modules import LowRankAdapter, EmbeddingLowRankAdapter, \
        QuantLowRankAdapter, QuantEmbeddingLowRankAdapter, WithAdapter

def init_lora_weights(module: LowRankAdapter, module_name: str) -> Dict[str, Tensor]:
    # Original LoRA paper initializes A to be normally distributed and
    # initializes B to zero.
    print('initialize lora %r' % module_name)
    return {
        '%s.lora_a' % module_name: torch.normal(0, 1, module.lora_a.shape), 
        '%s.lora_b' % module_name: torch.zeros(module.lora_b.shape), 
    }


def move_to_device(device):
    """
    Return a function that moves all parameters and buffers of a module to
    `device`.  Example usage: `my_module.apply(move_to_device(device))`.
    Unlike `my_module.to(device)`, this can "move" tensors from the `meta`
    device, by creating a new empty tensor of the correct shape.
    """
    def f(m):
        meta_device = torch.device('meta')
        for name, param in m.named_parameters(recurse=False):
            if param is None:
                pass
            elif param.device == meta_device:
                m.register_parameter(name, nn.Parameter(
                    torch.empty_like(param, device=device),
                    requires_grad = param.requires_grad,
                ))
            else:
                m.register_parameter(name, nn.Parameter(
                    param.to(device=device),
                    requires_grad = param.requires_grad,
                ))
        for name, buffer in m.named_buffers(recurse=False):
            if buffer is None:
                pass
            elif buffer.device == meta_device:
                setattr(m, name, torch.empty_like(buffer, device=device))
                #m.register_buffer(name, )
            else:
                setattr(m, name, buffer.to(device=device))
                #m.register_buffer(name, )
    return f


def _join_name(*args) -> str:
    """
    Join together parts of a module or parameter name with dots.  Empty
    components are omitted.
    """
    return '.'.join(x for x in args if x != '')

def build_module(
    arch,
    kind: str,
    loader = None,
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
        arch: model architecture to create a module for.
        kind: which module to create from `arch`.
        loader: used to look up quant formats for weights.  Required if
            `base_quant` or `lora_quant` is set.
        layer_index: when building a layer, this gives the layer index.
            Required if `base_quant` or `lora_quant` is set and `kind` is
            "layer".
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
        if not base_quant:
            base = nn.Embedding(num_embeddings, embedding_dim, device=device)
        else:
            base = quantized.QuantEmbedding(num_embeddings, embedding_dim,
                weight_quant=quant_type(convert_module_name(name), 'weight'),
                device=device)
        if not lora:
            return base
        else:
            if not lora_quant:
                adapter = EmbeddingLowRankAdapter(num_embeddings, embedding_dim, lora_rank,
                    device=device)
            else:
                adapter = QuantEmbeddingLowRankAdapter(num_embeddings, embedding_dim, lora_rank,
                    lora_quant=quant_type(convert_module_name(name), 'lora_a', for_lora=True),
                    device=device)
            return WithAdapter(base, adapter)

    def linear(name: str, in_features, out_features, bias=True, device=None):
        if not base_quant:
            base = nn.Linear(in_features, out_features, bias=bias, device=device)
        else:
            base = quantized.QuantLinear(in_features, out_features, bias=bias,
                weight_quant=quant_type(convert_module_name(name), 'weight'),
                bias_quant=get_quant_type(convert_module_name(name), 'bias'),
                device=device)
        if not lora:
            return base
        else:
            if not lora_quant:
                adapter = LowRankAdapter(in_features, out_features, lora_rank,
                    device=device)
            else:
                adapter = QuantLowRankAdapter(in_features, out_features, lora_rank,
                    lora_quant=quant_type(convert_module_name(name), 'lora_a', for_lora=True),
                    device=device)
            return WithAdapter(base, adapter)

    # Allocate on the `meta` device to skip unneeded initialization of weights.
    # The real weights will be loaded later and would overwrite any defaults.
    module = arch.make_module2(kind, linear=linear, embedding=embedding, device='meta') \
            .apply(move_to_device(device))
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
        if module_name.endswith('.base') or module_name == 'base':
            module_name = module_name.removesuffix('base').removesuffix('.')
            has_lora = True
        elif module_name.endswith('.adapter') or module_name == 'adapter':
            has_lora = True
            is_lora = True
            module_name = module_name.removesuffix('adapter').removesuffix('.')
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
                lora_module_name = _join_name(module_name, 'adapter')
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
                    target_key = _join_name(module_name, target_param)
                    state_dict[target_key] = state_dict[target_key] + weights

    #pprint(list(state_dict.keys()))
    # TODO: Remove strict=False.  Currently it gives an error about _quant and
    # _shape when loading quantized layers.  Probably the top-level module's
    # load_state_dict is checking and seeing that no parameters by that name
    # exist (those keys are processed specially in QuantizedTensor).
    module.load_state_dict(state_dict, strict=False)
