# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors import safe_open
from gguf import GGUFReader, GGMLQuantizationType
import numpy as np
import numpy.typing as npt
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torchtune.utils._distributed import contains_fsdp
from torchtune.utils import gguf_quant


class ModelType(Enum):
    LLAMA2 = "llama2"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    LLAMA3 = "llama3"


def get_path(input_dir: Path, filename: str, missing_ok: bool = False) -> Path:
    """
    Utility to recover and validate the path for a given file within a given directory.

    Args:
        input_dir (Path): Directory containing the file
        filename (str): Name of the file
        missing_ok (bool): Whether to raise an error if the file is missing.

    Returns:
        Path: Path to the file

    Raises:
        ValueError: If the file is missing and missing_ok is False.
    """
    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a valid directory.")

    file_path = Path.joinpath(input_dir, filename)

    # If missing_ok is False, raise an error if the path is invalid
    if not missing_ok and not file_path.is_file():
        raise ValueError(f"No file with name: {filename} found in {input_dir}.")
    return file_path


def safe_torch_load(checkpoint_path: Path, weights_only: bool = True) -> Dict[str, Any]:
    """
    Utility to load a checkpoint file in a safe manner.
    """
    try:
        # convert the path into a string since pathlib Path and mmap don't work
        # well together
        is_safetensors_file = (
            True if str(checkpoint_path).endswith(".safetensors") else False
        )
        if is_safetensors_file:
            result = {}
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    result[k] = f.get_tensor(k)
            state_dict = result
        else:
            state_dict = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                mmap=True,
                weights_only=weights_only,
            )
    except Exception as e:
        raise ValueError(f"Unable to load checkpoint from {checkpoint_path}. ") from e
    return state_dict


def transform_opt_state_dict(
    opt_state_dict: Dict[str, Any], model: nn.Module, optimizer: optim.Optimizer
) -> Dict[str, Any]:
    """
    Transforms the optimizer state dict for FSDP using the ``optim_state_dict_to_load``
    from distributed library within PyTorch. If FSDP is not used, the optimizer state dict is returned as is.

    Args:
        opt_state_dict (Dict[str, Any]): Optimizer state dict extracted from the checkpoint
        model (nn.Module): Model that checkpoint will be loaded into.
        optimizer (optim.Optimizer): Optimizer that optimizer state checkpoints will be loaded into.

    Returns:
        ckpt_dict (Dict[str, Any]): Transformed optimizer state dict.
    """
    optim_state_dict_to_load = (
        FSDP.optim_state_dict_to_load(model, optimizer, opt_state_dict)
        if contains_fsdp(model)
        else opt_state_dict
    )

    return optim_state_dict_to_load

GGUF_SIMPLE_TYPES = (
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.I8,
    GGMLQuantizationType.I16,
    GGMLQuantizationType.I32,
    GGMLQuantizationType.I64,
    GGMLQuantizationType.F64,
)

KEEP_QUANTIZED_PARTS = {
    # nn.Linear instances
    'attn_q',
    'attn_k',
    'attn_v',
    'attn_output',
    'ffn_gate',
    'ffn_up',
    'ffn_down',
}


def _gguf_load_q6_k(
    data: npt.NDArray[np.uint8],
    name: str,
    shape: Tuple,
) -> Dict[str, Any]:
    qs, scales, d = gguf_quant.unpack_q6_k(data)

#    data2 = gguf_quant.pack_q6_k(qs, scales, d)
#    assert data2.shape == data.shape
#    if (data2 != data).any():
#        i = np.argwhere(data2 != data)[0, 0]
#        print(i, data.shape)
#        print(data[max(0, i - 10) : i + 10])
#        print(data2[max(0, i - 10) : i + 10])
#        assert False

    dequant = True
    if name.startswith('blk.') and name.split('.')[2] in KEEP_QUANTIZED_PARTS:
        dequant = False
    if name == 'output.weight':
        dequant = False

    if not dequant:
        return {
            '%s_quant.k_qs' % name: torch.from_numpy(qs),
            '%s_quant.k_scales' % name: torch.from_numpy(scales),
            '%s_quant.k_d' % name: torch.from_numpy(d),
            }
    else:
        x = gguf_quant.dequant_q6_k(qs, scales, d)
        x = torch.from_numpy(x).view(*shape)
        return {name: x}


def load_gguf(path: Path, filter_name_prefix: Optional[str] = None) -> Dict[str, Any]:
    reader = GGUFReader(path)
    state_dict: Dict[str, Any] = {}
    quant_map: Dict[str, GGMLQuantizationType] = {}
    for i, tensor in enumerate(reader.tensors):
        quant = GGMLQuantizationType(tensor.tensor_type)
        quant_map[tensor.name] = quant

        if filter_name_prefix is not None and not tensor.name.startswith(filter_name_prefix):
            continue

        shape_length = max((j + 1 for j, dim in enumerate(tensor.shape) if dim != 1),
            default=len(tensor.shape))
        shape = tuple(reversed(tensor.shape[:shape_length]))
        print(tensor.name, tensor.data.shape, shape, tensor.shape)

        if tensor.tensor_type in GGUF_SIMPLE_TYPES:
            state_dict[tensor.name] = torch.from_numpy(tensor.data).view(*shape)
        elif quant == GGMLQuantizationType.Q6_K:
            state_dict.update(_gguf_load_q6_k(tensor.data, tensor.name, shape))
        else:
            raise ValueError("unsupported quantization %s for %s" % (quant, tensor.name))
    state_dict['gguf_quant_map'] = quant_map
    return state_dict


def save_config(path: Path, config: Dict[str, Any]) -> None:
    """
    Save a configuration dictionary to a file.

    Args:
        path (Path): Path to save the configuration file.
        config (Dict[str, Any]): Configuration dictionary to save.
    """
    if not path.is_dir():
        path.mkdir(exist_ok=True)
    file_path = Path.joinpath(path, "config.json")
    if not file_path.exists():
        with open(file_path, "w") as f:
            json.dump(config, f)
