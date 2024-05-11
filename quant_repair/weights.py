import os
import re
import safetensors
from typing import Optional, Tuple, Dict, Iterable
import torch
from torch import Tensor
from torch import nn
from torchtune.models import convert_weights
from torchtune.modules import quantized
from gguf import GGMLQuantizationType

def load_weights_safetensors_hf(weights_dir, arch):
    """
    Load all `model*.safetensors` files from `weights_dir` and collect their
    tensors into a single `state_dict`.  Files are assumed to use HuggingFace
    tensor format.  The model architecture `arch` is required for converting
    the weights to TorchTune format.
    """
    state_dict = {}
    for name in os.listdir(weights_dir):
        if not (name.startswith('model') and name.endswith('.safetensors')):
            continue
        # This will mmap the file.  Data will be paged in and out on
        # demand, so this effectively consumes no RAM.
        path = os.path.join(weights_dir, name)
        chunk = safetensors.torch.load_file(path, device='cpu')
        for key in chunk.keys():
            assert key not in state_dict, \
                'duplicate tensor %s found in %s' % (key, path)
        state_dict.update(chunk)
    state_dict = convert_weights.hf_to_tune(
        state_dict,
        num_heads=arch.num_heads,
        num_kv_heads=arch.num_kv_heads,
        dim=arch.embed_dim,
    )
    return state_dict

class CheckpointStateDict:
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def quant_type(self, key: str) -> GGMLQuantizationType:
        return GGMLQuantizationType.F16

    def get_quant_type(self, key: str) -> Optional[GGMLQuantizationType]:
        if key in self.state_dict:
            return GGMLQuantizationType.F16
        else:
            return None

    def has(self, key: str) -> bool:
        return key in self.state_dict

    def get(
        self,
        key: str,
        result_key: Optional[str] = None,
        dequant: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Retrieve a tensor.  Returns a dict mapping `key` plus various suffixes
        to the pieces of the quantized tensor named `key`.  If `result_key` is
        set, it will be used in place of `key` when building the output dict.

        `dequant` is ignored; this class assumes all weights in the
        `state_dict` are already unquantized.
        """
        if result_key is None:
            result_key = key
        return {result_key: self.state_dict[key]}

    def get_multi(
        self,
        keys: Iterable[str],
        result_keys: Optional[str] = None,
        dequant: bool = False,
    ) -> Dict[str, Tensor]:
        if result_keys is not None:
            keys_iter = zip(keys, result_keys, strict=True)
        else:
            keys_iter = ((k, k) for k in keys)

        return {
            result_key: self.state_dict[key] for key, result_key in keys_iter
        }

CHECKPOINT_FILE_RE = re.compile(r'([a-z0-9_]+?)(_ckpt([0-9]+))?\.pt$')

class QuantizedCheckpointLoader:
    def __init__(self, quant_dir, dequant_device=None):
        self.quant_dir = quant_dir
        self.dequant_device = dequant_device
        self.last_state_dict = None
        self.last_state_dict_file = None

        # Dict containing the most recent checkpoint file and counter for each
        # module.  Valuse will be pairs like `("layer1_ckpt3.pt", 3)`.  The
        # counter is used to decide where to write the next checkpoint.
        self.checkpoint_files = {}
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
            if key not in self.checkpoint_files:
                insert = True
            else:
                old_version = self.checkpoint_files[key][1]
                insert = version > old_version
            if insert:
                self.checkpoint_files[key] = (name, version)

    def _load_state_dict(self, module):
        file_name, _ = self.checkpoint_files[module]
        file_path = os.path.join(self.quant_dir, file_name)
        if self.last_state_dict_file != file_path:
            self.last_state_dict = torch.load(file_path, weights_only=True)
            self.last_state_dict_file = file_path
        return self.last_state_dict

    def _split_key(self, key: str) -> Tuple[str, str]:
        """
        Split a parameter name into `module` and `rel_key` components.
        `module` is a module name suitable for passing to
        `self._load_state_dict`.
        """
        if key.startswith('layers.'):
            _, layer_index_str, rel_key = key.split('.', 2)
            layer_index = int(layer_index_str)
            module = 'layer%d' % layer_index
        else:
            module, rel_key = key.split('.', 1)
        return module, rel_key

    def next_checkpoint_file(self, module: str) -> str:
        _, checkpoint_index = self.checkpoint_files[module]
        checkpoint_index += 1
        return '%s_ckpt%d.pt' % (module, checkpoint_index)

    def quant_type(self, key: str) -> GGMLQuantizationType:
        module, rel_key = self._split_key(key)
        state_dict = self._load_state_dict(module)
        return state_dict[rel_key + '_quant._quant']

    def get_quant_type(self, key: str) -> Optional[GGMLQuantizationType]:
        module, rel_key = self._split_key(key)
        state_dict = self._load_state_dict(module)
        return state_dict.get(rel_key + '_quant._quant')

    def has(self, key: str) -> bool:
        module, rel_key = self._split_key(key)
        state_dict = self._load_state_dict(module)
        unquant_key = rel_key
        quant_key = rel_key + '_quant._quant'
        return unquant_key in state_dict or quant_key in state_dict

    def get(
        self,
        key: str,
        result_key: Optional[str] = None,
        dequant: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Retrieve a tensor in quantized or dequantized form.  Returns a dict
        mapping `key` plus various suffixes to the pieces of the quantized
        tensor named `key`.  If `dequant` is set, it instead returns a
        single-element dict mapping `key` to the dequantized version of the
        tensor.  If `result_key` is set, it will be used in place of `key` when
        building the output dict.
        """
        return self.get_multi(
            (key,),
            result_keys = (result_key,) if result_key is not None else None,
            dequant = dequant, 
        )

    def get_multi(
        self,
        keys: Iterable[str],
        result_keys: Optional[str] = None,
        dequant: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Retrieve multiple tensors in quantized or dequantized form.  Returns a
        dict with several entries for each key in `keys`, mapping the key plus
        various suffixes to the pieces of the quantized tensor identified by
        that key.  If `dequant` is set, it instead returns a dict mapping each
        key to the dequantized version of the tensor.  If `result_keys` are
        set, they will be used in place of the corresponding `keys` when
        building the output dict (in which case `result_keys` must have the
        same length as `keys`).
        """
        if result_keys is not None:
            keys_iter = zip(keys, result_keys, strict=True)
        else:
            keys_iter = ((k, k) for k in keys)

        result = {}
        for key, result_key in keys_iter:
            module, rel_key = self._split_key(key)
            state_dict = self._load_state_dict(module)

            if rel_key in state_dict:
                # Unquantized case
                result[result_key] = state_dict[rel_key]
            else:
                quant = state_dict['%s_quant._quant' % rel_key]
                shape = state_dict['%s_quant._shape' % rel_key]

                if dequant:
                    qt = quantized.make_quantized_tensor(shape, quant)
                    if self.dequant_device is not None:
                        qt = qt.to(self.dequant_device)
                    qt_state_dict = {
                        param: state_dict['%s_quant.%s' % (rel_key, param)]
                        for param, _ in qt.named_parameters()
                    }
                    qt_state_dict['_quant'] = quant
                    qt_state_dict['_shape'] = shape
                    qt.load_state_dict(qt_state_dict)
                    result[result_key] = qt.forward().cpu()
                else:
                    params = quantized.QUANTIZED_TENSOR_TYPE[quant].PARAMETER_NAMES
                    for param in params:
                        result['%s_quant.%s' % (result_key, param)] = \
                                state_dict['%s_quant.%s' % (rel_key, param)]
                    result['%s_quant._quant' % result_key] = quant
                    result['%s_quant._shape' % result_key] = shape

        return result
