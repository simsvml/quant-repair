# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors import safe_open
from gguf import GGUFReader, GGMLQuantizationType
import numpy as np
import numpy.typing as npt
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torchtune.utils._distributed import contains_fsdp


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

QK_K = 256

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

def _split_bits(
    data: npt.NDArray[Any],
    ranges: List[Tuple[int, int]],
    dtype=None,
) -> npt.NDArray[Any]:
    """
    Extract the bits identified by `ranges` from each element of `data`.  Each
    element is expanded into a new dimension of size `len(ranges)`.
    """
    dtype = dtype if dtype is not None else data.dtype
    output = np.empty(data.shape + (len(ranges),), dtype)
    for i, (lo, hi) in enumerate(ranges):
        output_slice = output[..., i]
        np.right_shift(data, lo, out=output_slice, casting='same_kind')
        mask = (1 << (hi - lo)) - 1
        np.bitwise_and(output_slice, mask, out=output_slice)
    return output

def _gguf_unpack_q6_k(
    data: npt.NDArray[np.uint8],
    name: str,
    shape: Tuple,
) -> Dict[str, Any]:
    block_type = np.dtype([
        ('ql', np.uint8, QK_K // 2),
        ('qh', np.uint8, QK_K // 4),
        ('scales', np.int8, QK_K // 16),
        ('d', np.float16),
    ])
    blocks = data.view(block_type)

    scales = blocks['scales']
    d = blocks['d']

    # Unpack the `q` values from `ql` and `qh`.  Each block has 256 of these,
    # stored in a complicated order.  Looking at `dequantize_row_q6_K` in
    # `ggml-quants.c` (specifically, the `QK_K == 256` case), each block is
    # divided into two half-blocks, and the `q` values for each half-block are
    # assembled as follows:
    #
    # * Take the low bits `0:4` of each `ql[0:64]`, then take the high bits
    #   `4:8` of each `ql[0:64]`, to get the 128 `ql` values for the block.
    # * Take bits `0:2` of each `qh[0:32]`, then `2:4` of each, then `4:6`, and
    #   finally `6:8`, to get the 128 `qh` values.
    # * Combine each pair by computing `q = (ql | (qh << 4)) - 32` to obtain
    #   the final 6-bit value.

    ql = _split_bits(blocks['ql'], [(0, 4), (4, 8)])
    # [block, half_block, byte, bits]
    # Each half-block has 64 `ql` values, split into low bits and high bits.
    ql = ql.reshape(-1, 2, 64, 2)
    # Within each half-block, we use all the low bits first, then all the high
    # bits.
    ql = ql.swapaxes(-1, -2)
    # Final output is a sequence of 256 4-bit values per block.
    ql = ql.reshape(-1, 256)

    qh = _split_bits(blocks['qh'], [(0, 2), (2, 4), (4, 6), (6, 8)])
    # [block, half_block, byte, bits]
    # Each half-block has 32 `qh` values, split into four 2-bit values.
    qh = qh.reshape(-1, 2, 32, 4)
    qh = qh.swapaxes(-1, -2)
    qh = qh.reshape(-1, 256)

    np.left_shift(qh, 4, out=qh)
    np.bitwise_or(ql, qh, out=ql)
    qs = ql
    del ql, qh
    assert qs.dtype == np.uint8
    qs = qs.astype(np.int8, copy=False)
    qs -= 32

    qs = qs.reshape(-1, 16, 16)

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
        # Note the order of operations here.  Doing `q * shape` first would produce
        # a `uint8` output, causing issues with overflow.
        x = d.reshape(*d.shape, 1, 1) * scales.reshape(*scales.shape, 1) * qs
        x = torch.from_numpy(x).view(*shape)
        return {name: x}


def load_gguf(path: Path) -> Dict[str, Any]:

    reader = GGUFReader(path)
    state_dict: Dict[str, Any] = {}
    quant_map: Dict[str, GGMLQuantizationType] = {}
    for i, tensor in enumerate(reader.tensors):
        shape_length = max((j + 1 for j, dim in enumerate(tensor.shape) if dim != 1),
            default=len(tensor.shape))
        shape = tuple(reversed(tensor.shape[:shape_length]))
        print(tensor.name, tensor.data.shape, shape, tensor.shape)

        quant = GGMLQuantizationType(tensor.tensor_type)
        quant_map[tensor.name] = quant

        if tensor.tensor_type in GGUF_SIMPLE_TYPES:
            state_dict[tensor.name] = torch.from_numpy(tensor.data).view(*shape)
        elif quant == GGMLQuantizationType.Q6_K:
            state_dict.update(_gguf_unpack_q6_k(tensor.data, tensor.name, shape))
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
