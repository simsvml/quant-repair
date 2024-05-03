from collections import defaultdict
import os
from pprint import pprint
import sys
from typing import Optional, Any, Dict
import torch
from torch import Tensor
from torch.nn import functional as F
from torchtune.models import convert_weights
from torchtune.modules import quantized
from torchtune.utils import gguf_quant
import gguf
from gguf import GGUFReader, GGMLQuantizationType


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

def gguf_unpack_tensor(
    tensor: gguf.ReaderTensor,
    output_name: Optional[str] = None,
    device = None,
) -> Dict[str, Tensor]:
    if output_name is None:
        output_name = tensor.name

    shape_length = max((j + 1 for j, dim in enumerate(tensor.shape) if dim != 1),
        default=len(tensor.shape))
    shape = tuple(reversed(tensor.shape[:shape_length]))

    quant = GGMLQuantizationType(tensor.tensor_type)
    print(quant.name, shape, tensor.name)

    # Quantization is current unsupported for these tensors, so dequantize them
    # and return a normal floating-point representation.
    dequantize = tensor.name == 'token_embd.weight' or 'norm' in tensor.name

    state_dict = {}

    def conv_qs(x) -> Tensor:
        x = torch.from_numpy(x)
        if device is not None:
            x = x.to(device)
        x = x.to(quantized.QUANTIZED_DTYPE[quant])
        latent = quantized.quantized_to_latent(x, quantized.QS_BOUNDS[quant])
        #y = quantized.latent_to_quantized(latent, quantized.QS_BOUNDS[quant])
        #print('roundtrip loss: %s' % (F.mse_loss(y, x),))
        latent = latent.to('cpu')
        return latent

    def conv_scales(x) -> Tensor:
        x = torch.from_numpy(x)
        if device is not None:
            x = x.to(device)
        x = x.to(quantized.QUANTIZED_DTYPE[quant])
        latent = quantized.quantized_to_latent(x, quantized.SCALES_BOUNDS[quant])
        #y = quantized.latent_to_quantized(latent, quantized.SCALES_BOUNDS[quant])
        #print('roundtrip loss: %s' % (F.mse_loss(y, x),))
        latent = latent.to('cpu')
        return latent


    if quant in quantized.UNQUANTIZED_TYPES:
        state_dict[output_name] = torch.from_numpy(tensor.data).view(*shape)
    elif quant == GGMLQuantizationType.Q6_K:
        qs, scales, d = gguf_quant.unpack_q6_k(tensor.data)

        if dequantize:
            data = gguf_quant.dequant_q6_k(qs, scales, d)
            state_dict[output_name] = torch.from_numpy(data).view(*shape)
            return state_dict

        state_dict.update({
            '%s_quant.k_qs' % output_name: conv_qs(qs),
            '%s_quant.k_scales' % output_name: conv_scales(scales),
            '%s_quant.k_d' % output_name: torch.from_numpy(d),
        })
    elif quant == GGMLQuantizationType.Q5_K:
        qs, sc, m, d, dmin = gguf_quant.test_unpack_q5_k(tensor.data)

        if dequantize:
            data = gguf_quant.dequant_q5_k(qs, sc, m, d, dmin)
            state_dict[output_name] = torch.from_numpy(data).view(*shape)
            return state_dict

        state_dict.update({
            '%s_quant.k_qs' % output_name: conv_qs(qs),
            '%s_quant.k_sc' % output_name: conv_scales(sc),
            '%s_quant.k_m' % output_name: conv_scales(m),
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
            '%s_quant.k_qs' % output_name: conv_qs(qs),
            '%s_quant.k_sc' % output_name: conv_scales(sc),
            '%s_quant.k_m' % output_name: conv_scales(m),
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
            '%s_quant.k_qs' % output_name: conv_qs(qs),
            '%s_quant.k_scales' % output_name: conv_scales(scales),
            '%s_quant.k_d' % output_name: torch.from_numpy(d),
        })
    elif quant == GGMLQuantizationType.Q2_K:
        qs, sc, m, d, dmin = gguf_quant.test_unpack_q2_k(tensor.data)

        if dequantize:
            data = gguf_quant.dequant_q2_k(qs, sc, m, d, dmin)
            state_dict[output_name] = torch.from_numpy(data).view(*shape)
            return state_dict

        state_dict.update({
            '%s_quant.k_qs' % output_name: conv_qs(qs),
            '%s_quant.k_sc' % output_name: conv_scales(sc),
            '%s_quant.k_m' % output_name: conv_scales(m),
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
            '%s_quant.k_scales' % output_name: conv_scales(scales),
            '%s_quant.k_d' % output_name: torch.from_numpy(d),
        })
    else:
        raise AssertionError('quant %s not implemented for tensor %s' % (
            quant, tensor.name))

    return state_dict

def main():
    assert len(sys.argv) == 3
    gguf_path = sys.argv[1]
    output_dir = sys.argv[2]

    # Use CUDA for converting quantized weights to latent weights.
    device = 'cuda'

    print('load weights from %r' % (gguf_path,))
    reader = GGUFReader(gguf_path)

    # Build the list of layer keys.
    quant_map = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
    quant_map = convert_weights.gguf_to_tune(quant_map)

    quant_shape_map = {}
    for tensor in reader.tensors:
        name = tensor.name
        quant = tensor.tensor_type
        shape_length = max((j + 1 for j, dim in enumerate(tensor.shape) if dim != 1),
            default=len(tensor.shape))
        shape = tuple(int(x) for x in reversed(tensor.shape[:shape_length]))
        quant_shape_map[name] = (quant, shape)
    quant_shape_map = convert_weights.gguf_to_tune(quant_shape_map)

    # Collect layers
    named_tensor_map = {tensor.name: tensor for tensor in reader.tensors}
    named_tensor_map = convert_weights.gguf_to_tune(named_tensor_map)

    layer_tensors = defaultdict(list)
    for tune_name in named_tensor_map.keys():
        if tune_name.startswith('layers.'):
            layer_index = int(tune_name.split('.')[1])
            layer_tensors[layer_index].append(tune_name)

    # Convert tensors and save to files

    os.makedirs(output_dir, exist_ok=True)

    # Avoid holding multple unpacked layers in memory at the same time.
    for layer_index, tensor_names in layer_tensors.items():
        print('layers.%d' % layer_index)
        state_dict = {}
        for name in tensor_names:
            tensor = named_tensor_map[name]
            short_name = '.'.join(name.split('.')[2:])
            state_dict.update(gguf_unpack_tensor(tensor, short_name, device=device))
        output_path = os.path.join(output_dir, 'layer%d.pt' % layer_index)
        torch.save(state_dict, output_path)
        del state_dict
        print('saved %s' % output_path)

    def save_single_tensor(module, param_name):
        print(module)
        tensor = named_tensor_map['%s.%s' % (module, param_name)]
        state_dict = gguf_unpack_tensor(tensor, param_name, device=device)
        output_path = os.path.join(output_dir, '%s.pt' % module)
        torch.save(state_dict, output_path)
        print('saved %s' % output_path)

    save_single_tensor('tok_embeddings', 'weight')
    save_single_tensor('norm', 'scale')
    save_single_tensor('output', 'weight')

    # Save the quant map so the reader doesn't have to consult the GGUF.
    print('quant map')
    output_path = os.path.join(output_dir, 'quant_map.pt')
    torch.save(quant_map, output_path)
    print('saved %s' % output_path)

    print('quant and shape map ')
    output_path = os.path.join(output_dir, 'quant_shape_map.pt')
    torch.save(quant_shape_map, output_path)
    print('saved %s' % output_path)



if __name__ == '__main__':
    main()
