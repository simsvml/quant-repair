from dataclasses import dataclass
import os
from typing import Any, Optional, Tuple, Callable
import torch
from torch import Tensor
from gguf import GGUFReader
import safetensors.torch
from . import functional as QRF
from .functional import TensorParams
from .quantized import DequantizeParams

@dataclass(frozen = True)
class WeightConversion:
    rename_to_torchtune: Callable[[str], str]
    rename_from_torchtune: Callable[[str], str]

    transform_to_torchtune: Optional[Callable[[str, Tensor], Tensor]] = None
    transform_from_torchtune: Optional[Callable[[str, Tensor], Tensor]] = None

    def key_to_torchtune(self, key: str) -> str:
        return self.rename_to_torchtune(key)

    def tensor_to_torchtune(self, new_key: str, tensor: Tensor) -> Tensor:
        if self.transform_to_torchtune is None:
            return tensor
        return self.transform_to_torchtune(new_key, tensor)

    def key_from_torchtune(self, key: str) -> str:
        return self.rename_from_torchtune(key)

    def to_torchtune(self, key: str, tensor: Tensor) -> Tuple[str, Tensor]:
        new_key = self.key_to_torchtune(key)
        new_tensor = self.tensor_to_torchtune(new_key, tensor)
        return new_key, new_tensor


_HF_TO_TORCHTUNE = {
    'model.embed_tokens.weight': 'tok_embeddings.weight',
    'model.norm.weight': 'norm.scale',
    'lm_head.weight': 'output.weight',
}
_TORCHTUNE_TO_HF = {v: k for k, v in _HF_TO_TORCHTUNE.items()}

_HF_TO_TORCHTUNE_LAYER = {
    'self_attn.q_proj.weight': 'attn.q_proj.weight',
    'self_attn.k_proj.weight': 'attn.k_proj.weight',
    'self_attn.v_proj.weight': 'attn.v_proj.weight',
    'self_attn.o_proj.weight': 'attn.output_proj.weight',
    'mlp.gate_proj.weight': 'mlp.w1.weight',
    'mlp.down_proj.weight': 'mlp.w2.weight',
    'mlp.up_proj.weight': 'mlp.w3.weight',
    'input_layernorm.weight': 'sa_norm.scale',
    'post_attention_layernorm.weight': 'mlp_norm.scale',
}
_TORCHTUNE_TO_HF_LAYER = {v: k for k, v in _HF_TO_TORCHTUNE_LAYER.items()}

def hf_conversion(arch) -> WeightConversion:
    def rename_hf_to_torchtune(key: str) -> str:
        if key.startswith('model.layers.'):
            _, _, layer_index_str, rel_key = key.split('.', 3)
            new_rel_key = _HF_TO_TORCHTUNE_LAYER[rel_key]
            return 'layers.%s.%s' % (layer_index_str, new_rel_key)
        else:
            return _HF_TO_TORCHTUNE[key]

    def rename_torchtune_to_hf(key: str) -> str:
        if key.startswith('layers.'):
            _, layer_index_str, rel_key = key.split('.', 2)
            new_rel_key = _TORCHTUNE_TO_HF_LAYER[rel_key]
            return 'model.layers.%s.%s' % (layer_index_str, new_rel_key)
        else:
            return _TORCHTUNE_TO_HF[key]

    dim = arch.embed_dim
    head_dim = arch.head_dim()
    # From torchtune/models/convert_weights.py
    def _permute(t, n_heads):
        return (
            t.view(n_heads, 2, head_dim // 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    def transform_hf_to_torchtune(key: str, tensor: Tensor) -> Tensor:
        if 'q_proj' in key:
            return _permute(tensor, arch.num_heads)
        elif 'k_proj' in key:
            return _permute(tensor, arch.num_kv_heads)
        else:
            return tensor

    def transform_torchtune_to_hf(key: str, tensor: Tensor) -> Tensor:
        assert False, 'TODO: implement transform_torchtune_to_hf'

    return WeightConversion(
        rename_to_torchtune = rename_hf_to_torchtune,
        rename_from_torchtune = rename_torchtune_to_hf,
        transform_to_torchtune = transform_hf_to_torchtune,
        transform_from_torchtune = transform_torchtune_to_hf,
    )


_LLAMA_CPP_TO_TORCHTUNE = {
    'token_embd.weight': 'tok_embeddings.weight',
    'output_norm.weight': 'norm.scale',
    'output.weight': 'output.weight',
}
_TORCHTUNE_TO_LLAMA_CPP = {v: k for k, v in _LLAMA_CPP_TO_TORCHTUNE.items()}

_LLAMA_CPP_TO_TORCHTUNE_LAYER = {
    'attn_q.weight': 'attn.q_proj.weight',
    'attn_k.weight': 'attn.k_proj.weight',
    'attn_v.weight': 'attn.v_proj.weight',
    'attn_output.weight': 'attn.output_proj.weight',
    'ffn_gate.weight': 'mlp.w1.weight',
    'ffn_down.weight': 'mlp.w2.weight',
    'ffn_up.weight': 'mlp.w3.weight',
    'attn_norm.weight': 'sa_norm.scale',
    'ffn_norm.weight': 'mlp_norm.scale',
}
_TORCHTUNE_TO_LLAMA_CPP_LAYER = {v: k for k, v in _LLAMA_CPP_TO_TORCHTUNE_LAYER.items()}

def llama_cpp_conversion() -> WeightConversion:
    def rename_llama_cpp_to_torchtune(key: str) -> str:
        if key.startswith('blk.'):
            _, layer_index_str, rel_key = key.split('.', 2)
            new_rel_key = _LLAMA_CPP_TO_TORCHTUNE_LAYER[rel_key]
            return 'layers.%s.%s' % (layer_index_str, new_rel_key)
        else:
            return _LLAMA_CPP_TO_TORCHTUNE[key]

    def rename_torchtune_to_llama_cpp(key: str) -> str:
        if key.startswith('layers.'):
            _, layer_index_str, rel_key = key.split('.', 2)
            new_rel_key = _TORCHTUNE_TO_LLAMA_CPP_LAYER[rel_key]
            return 'blk.%s.%s' % (layer_index_str, new_rel_key)
        else:
            return _TORCHTUNE_TO_LLAMA_CPP[key]

    return WeightConversion(
        rename_to_torchtune = rename_llama_cpp_to_torchtune,
        rename_from_torchtune = rename_torchtune_to_llama_cpp,
    )


NO_CONVERSION = WeightConversion(
    rename_to_torchtune = lambda k: k,
    rename_from_torchtune = lambda k: k,
)


class StateDictLoader:
    def __init__(self, state_dict, convert: Optional[WeightConversion] = NO_CONVERSION):
        self.tensors = {}
        self.metadata = {}
        for key, value in state_dict.items():
            # TODO: Detect TensorParams dict format, or combine self.tensors
            # and self.metadata
            if not isinstance(value, (Tensor, TensorParams, dict)):
                self.metadata[key] = value
                continue
            try:
                new_key = convert.key_to_torchtune(key)
            except KeyError:
                # Failed to convert.  `key` was missing from some `convert`
                # lookup table.
                new_key = key
            assert new_key not in self.tensors, 'duplicate key after renaming: %r' % new_key
            self.tensors[new_key] = value

        self.convert = convert

    @classmethod
    def from_file(
        cls,
        path: str,
        convert: Optional[WeightConversion] = NO_CONVERSION,
    ) -> 'StateDictLoader':
        return cls(torch.load(path, weights_only = True), convert = convert)

    def has(self, key: str) -> bool:
        return key in self.tensors

    def get(self, key: str, device: Optional[torch.device] = None) -> TensorParams:
        tensor = self.tensors[key]
        if isinstance(tensor, Tensor):
            new_tensor = self.convert.tensor_to_torchtune(key, tensor)
            return TensorParams.from_unquantized_tensor(new_tensor.to(device = device))
        else:
            if not isinstance(tensor, TensorParams):
                tensor = TensorParams.from_dict(tensor)
            assert self.convert.transform_to_torchtune is None
            return tensor.to(device)

    def get_meta(self, key: str) -> Any:
        return self.metadata[key]

class SafetensorsLoader:
    def __init__(self, path, convert: Optional[WeightConversion] = NO_CONVERSION):
        if os.path.isdir(path):
            paths = [os.path.join(path, name) for name in os.listdir(path)
                if not name.startswith('.') and name.endswith('.safetensors')]
        else:
            paths = [path]

        self.convert = convert

        self.tensors = {}
        for path in paths:
            print('opening %r' % path)
            st = safetensors.torch.load_file(path, device = 'cpu')
            for key, tensor in st.items():
                assert key not in self.tensors, 'duplicate tensor %r in %r' % (key, path)
                new_key = convert.key_to_torchtune(key)
                self.tensors[new_key] = tensor

    def has(self, key: str) -> bool:
        return key in self.tensors

    def get(self, key: str, device: Optional[torch.device] = None) -> TensorParams:
        tensor = self.tensors[key]
        new_tensor = self.convert.tensor_to_torchtune(key, tensor)
        return TensorParams.from_unquantized_tensor(new_tensor.to(device = device))

class GGUFLoader:
    def __init__(self, path, convert: Optional[WeightConversion] = NO_CONVERSION):
        self.reader = GGUFReader(path)
        self.tensors = {convert.key_to_torchtune(tensor.name): tensor
            for tensor in self.reader.tensors}

        # This class doesn't support applying transformations to the tensors
        assert convert.transform_to_torchtune is None

    def has(self, key: str) -> bool:
        return key in self.tensors

    def get(self, key: str, device: Optional[torch.device] = None) -> TensorParams:
        gguf_tensor = self.tensors[key]
        shape_length = max((j + 1 for j, dim in enumerate(gguf_tensor.shape) if dim != 1),
            default=len(gguf_tensor.shape))
        shape = tuple(int(x) for x in reversed(gguf_tensor.shape[:shape_length]))
        return TensorParams(
            data = torch.from_numpy(gguf_tensor.data).to(device = device),
            dequant = DequantizeParams(
                quant = gguf_tensor.tensor_type,
                shape = shape,
            )
        )

