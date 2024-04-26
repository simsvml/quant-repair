import os
import sys
import torch

import gguf
from gguf import GGMLQuantizationType
from torchtune.models import convert_weights
from torchtune.modules import quantized
from torchtune.utils._checkpointing._checkpointer_utils import _gguf_pack_q6_k

def main():
    assert len(sys.argv) == 3
    gguf_path = sys.argv[1]
    layers_dir = sys.argv[2]

    print('opening %s' % gguf_path)
    reader = gguf.GGUFReader(gguf_path, 'r+')

    prev_layer = None
    prev_layer_dict = None

    def get_layer_tensor(layer, name):
        nonlocal prev_layer, prev_layer_dict
        if layer != prev_layer:
            layer_dir = os.path.join(layers_dir, str(layer))
            num_checkpoints = sum(
                1 for f in os.listdir(layer_dir) if f.startswith('torchtune_model_'))
            layer_path = os.path.join(layer_dir, 'torchtune_model_%d.pt' % (num_checkpoints - 1))
            print('loading layer %d from %s' % (layer, layer_path))
            state_dict = torch.load(layer_path)
            state_dict = convert_weights.tune_to_gguf(state_dict)
            prev_layer = layer
            prev_layer_dict = state_dict

        return prev_layer_dict[name]


    for tensor in reader.tensors:
        if tensor.name.startswith('blk.'):
            layer = int(tensor.name.split('.')[1])
            if tensor.tensor_type in quantized.UNQUANTIZED_TYPES:
                layer_tensor = get_layer_tensor(layer, tensor.name)
                tensor.data[...] = layer_tensor.numpy(force=True)
            elif tensor.tensor_type == GGMLQuantizationType.Q6_K:
                quant = tensor.tensor_type
                qs = get_layer_tensor(layer, '%s_quant.k_qs' % tensor.name)
                qs = qs.round().clamp(*quantized.QS_BOUNDS[quant]).to(torch.int8)
                qs = qs.numpy(force=True)
                scales = get_layer_tensor(layer, '%s_quant.k_scales' % tensor.name)
                scales = scales.round().clamp(*quantized.SCALES_BOUNDS[quant]).to(torch.int8)
                scales = scales.numpy(force=True)
                d = get_layer_tensor(layer, '%s_quant.k_d' % tensor.name)
                d = d.numpy(force=True)
                tensor.data[...] = _gguf_pack_q6_k(qs, scales, d)
            else:
                assert False, 'unsupported tensor type %s' % tensor.tensor_type
        else:
            # TODO
            pass





if __name__ == "__main__":
    main()
