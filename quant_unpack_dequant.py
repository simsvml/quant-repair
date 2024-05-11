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


def gguf_load_tensor(
    tensor: gguf.ReaderTensor,
    output_name: Optional[str] = None,
    device = None,
) -> Tensor:
    if output_name is None:
        output_name = tensor.name

    shape_length = max((j + 1 for j, dim in enumerate(tensor.shape) if dim != 1),
        default=len(tensor.shape))
    # Cast to int explicitly, since `ndarray.shape` is a special numpy type.
    shape = tuple(int(x) for x in reversed(tensor.shape[:shape_length]))

    quant = GGMLQuantizationType(tensor.tensor_type)
    print(quant.name, shape, tensor.name)


    if quant in quantized.UNQUANTIZED_TYPES:
        x = tensor.data
    elif quant == GGMLQuantizationType.Q6_K:
        qs, scales, d = gguf_quant.unpack_q6_k(tensor.data)
        x = gguf_quant.dequant_q6_k(qs, scales, d)
    elif quant == GGMLQuantizationType.Q5_K:
        qs, sc, m, d, dmin = gguf_quant.test_unpack_q5_k(tensor.data)
        x = gguf_quant.dequant_q5_k(qs, sc, m, d, dmin)
    elif quant == GGMLQuantizationType.Q4_K:
        qs, sc, m, d, dmin = gguf_quant.test_unpack_q4_k(tensor.data)
        x = gguf_quant.dequant_q4_k(qs, sc, m, d, dmin)
    elif quant == GGMLQuantizationType.Q3_K:
        qs, scales, d = gguf_quant.test_unpack_q3_k(tensor.data)
        x = gguf_quant.dequant_q3_k(qs, scales, d)
    elif quant == GGMLQuantizationType.Q2_K:
        qs, sc, m, d, dmin = gguf_quant.test_unpack_q2_k(tensor.data)
        x = gguf_quant.dequant_q2_k(qs, sc, m, d, dmin)
    elif quant == GGMLQuantizationType.IQ2_XS:
        grids, signs, scales, d = gguf_quant.test_unpack_iq2_xs(tensor.data)
        x = gguf_quant.dequant_iq2_xs(grids, signs, scales, d)
    else:
        raise AssertionError('quant %s not implemented for tensor %s' % (
            quant, tensor.name))

    num_elems = 1
    for d in shape:
        num_elems *= d

    return torch.from_numpy(x)[:num_elems].view(*shape)

def main():
    assert len(sys.argv) == 3
    gguf_path = sys.argv[1]
    output_dir = sys.argv[2]

    # Use CUDA for converting quantized weights to latent weights.
    device = 'cuda'

    print('load weights from %r' % (gguf_path,))
    reader = GGUFReader(gguf_path)

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
            # TODO: Fix hardcoded dtype
            state_dict[short_name] = \
                    gguf_load_tensor(tensor, short_name, device=device).to(torch.bfloat16)
        output_path = os.path.join(output_dir, 'layer%d.pt' % layer_index)
        torch.save(state_dict, output_path)
        del state_dict
        print('saved %s' % output_path)

    def save_single_tensor(module, param_name):
        print(module)
        tensor = named_tensor_map['%s.%s' % (module, param_name)]
        state_dict = {
            # TODO: Fix hardcoded dtype
            param_name: gguf_load_tensor(tensor, param_name, device=device).to(torch.bfloat16)
        }
        output_path = os.path.join(output_dir, '%s.pt' % module)
        torch.save(state_dict, output_path)
        print('saved %s' % output_path)

    save_single_tensor('tok_embeddings', 'weight')
    save_single_tensor('norm', 'scale')
    save_single_tensor('output', 'weight')


if __name__ == '__main__':
    main()
