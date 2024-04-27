from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import numpy.typing as npt

QK_K = 256


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

def _concat_bits(
    data: npt.NDArray[Any],
    ranges: List[Tuple[int, int]],
    dtype=None,
) -> npt.NDArray[Any]:
    """
    Concatenate bits along the last axis of `data` according to `ranges`.  This
    is the inverse of `_split_bits`, assuming the `ranges` cover all the
    nonzero bits in the original data.
    """
    dtype = dtype if dtype is not None else data.dtype
    output = np.zeros(data.shape[:-1], dtype)
    for i, (lo, hi) in enumerate(ranges):
        data_slice = data[..., i]
        mask = (1 << (hi - lo)) - 1
        tmp = np.bitwise_and(data_slice, mask, dtype=dtype, casting='same_kind')
        np.left_shift(tmp, lo, out=tmp)
        np.bitwise_or(output, tmp, out=output)
    return output


def _q6_k_dtype() -> np.dtype:
    return np.dtype([
        ('ql', np.uint8, QK_K // 2),
        ('qh', np.uint8, QK_K // 4),
        ('scales', np.int8, QK_K // 16),
        ('d', np.float16),
    ])

def unpack_q6_k(
    data: npt.NDArray[np.uint8],
) -> Tuple[npt.NDArray[np.int8], npt.NDArray[np.int8], npt.NDArray[np.float16]]:
    blocks = data.view(_q6_k_dtype())

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

    return qs, scales, d

def pack_q6_k(
    qs: npt.NDArray[np.int8],
    scales: npt.NDArray[np.int8],
    d: npt.NDArray[np.float16],
) -> npt.NDArray[np.uint8]:
    num_blocks = d.shape[0]
    assert d.shape == (num_blocks,)
    assert scales.shape == (num_blocks, 16)
    assert qs.shape == (num_blocks, 16, 16)
    blocks = np.empty((num_blocks,), _q6_k_dtype())

    blocks['d'][:] = d
    blocks['scales'][:, :] = scales

    qs = qs.reshape(-1)
    # Don't use += here to avoid mutating the argument.
    qs = qs + 32
    assert qs.dtype == np.int8
    qs = qs.astype(np.uint8, copy=False)

    qs = _split_bits(qs, [(0, 4), (4, 6)])
    ql = qs[..., 0]
    qh = qs[..., 1]

    # This is the reverse of the ql part of the unpacking procedure.
    ql = ql.reshape(-1, 2, 2, 64)
    ql = ql.swapaxes(-1, -2)
    ql = _concat_bits(ql, [(0, 4), (4, 8)])
    ql = ql.reshape(blocks['ql'].shape)
    blocks['ql'][:] = ql

    qh = qh.reshape(-1, 2, 4, 32)
    qh = qh.swapaxes(-1, -2)
    qh = _concat_bits(qh, [(0, 2), (2, 4), (4, 6), (6, 8)])
    qh = qh.reshape(blocks['qh'].shape)
    blocks['qh'][:] = qh

    return blocks.view(np.uint8)

def dequant_q6_k(
    qs: npt.NDArray[np.int8],
    scales: npt.NDArray[np.int8],
    d: npt.NDArray[np.float16],
) -> npt.NDArray[np.float16]:
    # Note the order of operations here.  Doing `q * shape` first would produce
    # a `uint8` output, causing issues with overflow.
    x = (d.reshape(*d.shape, 1, 1) * scales.reshape(*scales.shape, 1)) * qs
    return x
