from dataclasses import dataclass
import os
from pprint import pprint
import sys
from tempfile import TemporaryDirectory
from typing import Optional, Any, Dict
import torch
from torch import nn
from torch import Tensor
from torchtune.models import convert_weights
from torchtune.models.llama3 import llama3_8b, llama3_tokenizer_transformers
from torchtune.modules import quantized
from torchtune.modules.module_cache import ModuleCache, Llama3Arch
from torchtune.utils import FullModelGGUFCheckpointer, set_default_dtype
from gguf import GGUFReader, GGMLQuantizationType
from torchtune.utils import gguf_quant


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

def get_layer_format(quant_map, i):
    """
    Get the format of layer `i`, which is a tuple indicating which quantized
    format is used for each weight matrix in the layer.  The order of the tuple
    is unspecified; it should be used only to determine whether two layers have
    the same format or different ones.
    """
    return tuple(quant_map['layers.%d.%s' % (i, name)] for name in sorted(LAYER_PARAMETERS))




def top_tokens(tokenizer, logits, top_k=3, temperature=1.0):
    # https://medium.com/@pashashaik/natural-language-generation-from-scratch-in-large-language-models-with-pytorch-4d9379635316
    print('logits', logits.shape, logits)
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    top_k_probs = torch.nn.functional.softmax(top_k_logits / temperature, dim=-1)
    print('top_k_logits', top_k_logits)
    print('top_k_indices', top_k_indices)
    print('top_k_probs', top_k_probs)
    return [(prob.item(), tokenizer.decode(token)) for prob, token in zip(top_k_probs, top_k_indices)]


def gguf_load_tensor_unpacked(
    reader: GGUFReader,
    name: str,
    output_name: Optional[str] = None,
) -> Dict[str, Tensor]:
    if output_name is None:
        output_name = name

    tensor = None
    for tensor in reader.tensors:
        if tensor.name == name:
            break
    # The last value of `tensor` from the loop should be the one where the
    # names match.
    assert tensor.name == name, 'tensor %r not found' % name

    shape_length = max((j + 1 for j, dim in enumerate(tensor.shape) if dim != 1),
        default=len(tensor.shape))
    shape = tuple(reversed(tensor.shape[:shape_length]))

    quant = GGMLQuantizationType(tensor.tensor_type)
    print(quant.name, shape, tensor.name)

    # Quantization is current unsupported for these tensors, so dequantize them
    # and return a normal floating-point representation.
    dequantize = tensor.name == 'token_embd.weight' or 'norm' in tensor.name

    state_dict = {}

    if quant in quantized.UNQUANTIZED_TYPES:
        state_dict[output_name] = torch.from_numpy(tensor.data).view(*shape)
    elif quant == GGMLQuantizationType.Q6_K:
        qs, scales, d = gguf_quant.unpack_q6_k(tensor.data)

        if dequantize:
            data = gguf_quant.dequant_q6_k(qs, scales, d)
            state_dict[output_name] = torch.from_numpy(data).view(*shape)
            return state_dict

        state_dict.update({
            '%s_quant.k_qs' % output_name: torch.from_numpy(qs),
            '%s_quant.k_scales' % output_name: torch.from_numpy(scales),
            '%s_quant.k_d' % output_name: torch.from_numpy(d),
        })
    elif quant == GGMLQuantizationType.Q5_K:
        qs, sc, m, d, dmin = gguf_quant.test_unpack_q5_k(tensor.data)

        if dequantize:
            data = gguf_quant.dequant_q5_k(qs, sc, m, d, dmin)
            state_dict[output_name] = torch.from_numpy(data).view(*shape)
            return state_dict

        state_dict.update({
            '%s_quant.k_qs' % output_name: torch.from_numpy(qs),
            '%s_quant.k_sc' % output_name: torch.from_numpy(sc),
            '%s_quant.k_m' % output_name: torch.from_numpy(m),
            '%s_quant.k_d' % output_name: torch.from_numpy(d),
            '%s_quant.k_dmin' % output_name: torch.from_numpy(dmin),
        })
    elif quant == GGMLQuantizationType.Q4_K:
        qs, sc, m, d, dmin = gguf_quant.test_unpack_q4_k(tensor.data)

        if dequantize:
            data = gguf_quant.dequant_q4_k(qs, sc, m, d, dmin)
            state_dict[output_name] = torch.from_numpy(data).view(*shape)
            return state_dict

        state_dict.update({
            '%s_quant.k_qs' % output_name: torch.from_numpy(qs),
            '%s_quant.k_sc' % output_name: torch.from_numpy(sc),
            '%s_quant.k_m' % output_name: torch.from_numpy(m),
            '%s_quant.k_d' % output_name: torch.from_numpy(d),
            '%s_quant.k_dmin' % output_name: torch.from_numpy(dmin),
        })
    elif quant == GGMLQuantizationType.Q3_K:
        qs, scales, d = gguf_quant.test_unpack_q3_k(tensor.data)

        if dequantize:
            data = gguf_quant.dequant_q3_k(qs, scales, d)
            state_dict[output_name] = torch.from_numpy(data).view(*shape)
            return state_dict

        state_dict.update({
            '%s_quant.k_qs' % output_name: torch.from_numpy(qs),
            '%s_quant.k_scales' % output_name: torch.from_numpy(scales),
            '%s_quant.k_d' % output_name: torch.from_numpy(d),
        })
    elif quant == GGMLQuantizationType.Q2_K:
        qs, sc, m, d, dmin = gguf_quant.test_unpack_q2_k(tensor.data)

        if dequantize:
            data = gguf_quant.dequant_q2_k(qs, sc, m, d, dmin)
            state_dict[output_name] = torch.from_numpy(data).view(*shape)
            return state_dict

        state_dict.update({
            '%s_quant.k_qs' % output_name: torch.from_numpy(qs),
            '%s_quant.k_sc' % output_name: torch.from_numpy(sc),
            '%s_quant.k_m' % output_name: torch.from_numpy(m),
            '%s_quant.k_d' % output_name: torch.from_numpy(d),
            '%s_quant.k_dmin' % output_name: torch.from_numpy(dmin),
        })
    elif quant == GGMLQuantizationType.IQ2_XS:
        grids, signs, scales, d = gguf_quant.test_unpack_iq2_xs(tensor.data)

        if dequantize:
            data = gguf_quant.dequant_iq2_xs(qs, sc, m, d, dmin)
            state_dict[output_name] = torch.from_numpy(data).view(*shape)
            return state_dict

        state_dict.update({
            # TODO: Convert grids and signs from index to one-hot format
            #'%s_quant.k_grids' % output_name: torch.from_numpy(grids),
            #'%s_quant.k_signs' % output_name: torch.from_numpy(signs),
            '%s_quant.k_scales' % output_name: torch.from_numpy(scales),
            '%s_quant.k_d' % output_name: torch.from_numpy(d),
        })
    else:
        raise AssertionError('quant %s not implemented for tensor %s' % (
            quant, tensor.name))

    return state_dict


def main():
    with set_default_dtype(torch.bfloat16):
        with torch.no_grad():
            run()

def run():
    assert len(sys.argv) == 2
    gguf_path = sys.argv[1]
    dir_path = os.path.dirname(gguf_path)

    device = torch.device('cuda')
    arch = Llama3Arch.llama3_8b()

    print('load weights from %r' % (gguf_path,))

    reader = GGUFReader(gguf_path)


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

    print('load tokenizer')
    tokenizer = llama3_tokenizer_transformers(os.path.join(dir_path, 'tokenizer.json'))

    print('test run')
    tokens = tokenizer.encode('Hello, my name is', add_eos=False)
    print(tokens)

    x = torch.tensor([tokens], device=device, dtype=torch.int)
    for info in layer_info:
        print('\nrunning %s' % (info,))
        m = layer_module(info)
        x = m(x)
        print('layer %s: %s' % (info[1], x))

    print('\ndone')
    print(x[0, -1])
    print(top_tokens(tokenizer, x[0, -2], top_k=10))
    print(top_tokens(tokenizer, x[0, -1], top_k=10))


if __name__ == '__main__':
    main()
