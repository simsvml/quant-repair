from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import numpy.typing as npt


QK_K = 256
K_SCALE_SIZE = 12


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


def _unpack_k_qs_part(
    data: npt.NDArray[np.uint8],
    num_bits: int,
) -> npt.NDArray[np.uint8]:
    num_bytes = data.shape[-1]
    num_parts = 8 // num_bits
    # Each byte in `data` is divided into `num_parts` parts, each consisting of
    # `num_bits` bits.  The unpacked sequence consists of the lowest parts of
    # each byte, then the next-lowest part, and so on.
    ranges = [(i, i + num_bits) for i in range(0, 8, num_bits)]
    assert len(ranges) == num_parts
    qs = _split_bits(data, ranges)
    # The shape is currently `[..., byte, part]`.  We swap the last two axes to
    # produce `[..., part, byte]`, so that all the part-0 values come first,
    # then all the part-1 values, and so on.
    qs = qs.swapaxes(-1, -2)
    # Flatten the last two axes to produce a flat sequence of result values.
    qs = qs.reshape(*data.shape[:-1], num_bytes * num_parts)
    return qs

def _pack_k_qs_part(
    qs: npt.NDArray[np.uint8],
    num_bits: int,
) -> npt.NDArray[np.uint8]:
    num_parts = 8 // num_bits
    assert qs.shape[-1] % num_parts == 0
    num_bytes = qs.shape[-1] // num_parts
    # Reverse the operations from `_unpack_k_qs_part`.
    batch_shape = qs.shape[:-1]
    # Initial shape is `[..., num_bytes * num_parts]`
    qs = qs.reshape(*batch_shape, num_parts, num_bytes)
    qs = qs.swapaxes(-1, -2)
    # We now have `[..., num_bytes, num_parts]`, suitable for `_concat_bits`.
    ranges = [(i, i + num_bits) for i in range(0, 8, num_bits)]
    assert len(ranges) == num_parts
    data = _concat_bits(qs, ranges)
    return data

def _unpack_k_qs(
    ql: npt.NDArray[np.uint8],
    num_low_bits: int,
    num_low_sub_blocks: int,
    qh: Optional[npt.NDArray[np.uint8]] = None,
    num_high_bits: int = 0,
    num_high_sub_blocks: int = 0,
) -> npt.NDArray[np.uint8]:
    """
    Unpack K-quant `qs` values from low and high bits.
    """
    assert 8 % num_low_bits == 0
    block_size = ql.shape[-1] * (8 // num_low_bits)
    assert block_size % num_low_sub_blocks == 0
    if qh is not None:
        assert 8 % num_high_bits == 0
        assert block_size == qh.shape[-1] * (8 // num_high_bits)
        assert qh.shape[:-1] == ql.shape[:-1]
        assert block_size % num_high_sub_blocks == 0

    batch_shape = ql.shape[:-1]

    # Divide each block (the last axis) into sub-blocks.  Different K-quants
    # use different numbers of sub-blocks.
    elems_per_low_sub_block = block_size // num_low_sub_blocks
    bytes_per_low_sub_block = elems_per_low_sub_block * num_low_bits // 8
    ql = ql.reshape(*batch_shape, num_low_sub_blocks, bytes_per_low_sub_block)
    ql = _unpack_k_qs_part(ql, num_low_bits)
    ql = ql.reshape(*batch_shape, block_size)

    if qh is not None:
        elems_per_high_sub_block = block_size // num_high_sub_blocks
        bytes_per_high_sub_block = elems_per_high_sub_block * num_high_bits // 8
        qh = qh.reshape(*batch_shape, num_high_sub_blocks, bytes_per_high_sub_block)
        qh = _unpack_k_qs_part(qh, num_high_bits)
        qh = qh.reshape(*batch_shape, block_size)

        np.left_shift(qh, num_low_bits, out=qh)
        np.bitwise_or(ql, qh, out=ql)

    return ql

def _pack_k_qs(
    qs: npt.NDArray[np.uint8],
    num_low_bits: int,
    num_low_sub_blocks: int,
    num_high_bits: int = 0,
    num_high_sub_blocks: int = 0,
) -> Tuple[npt.NDArray[np.uint8], Optional[npt.NDArray[np.uint8]]]:
    """
    Pack K-quant `qs` values into separate low and high bits.
    """
    block_size = qs.shape[-1]
    assert 8 % num_low_bits == 0
    assert block_size % num_low_sub_blocks == 0
    if num_high_bits != 0:
        assert 8 % num_high_bits == 0
        assert block_size % num_high_sub_blocks == 0

    batch_shape = qs.shape[:-1]

    if num_high_bits != 0:
        ranges = [(0, num_low_bits), (num_low_bits, num_low_bits + num_high_bits)]
        qs = _split_bits(qs, ranges)
        ql = qs[..., 0]
        qh = qs[..., 1]
    else:
        ql = qs
        qh = None

    # Reverse the operations from `_unpack_k_qs`.
    elems_per_low_sub_block = block_size // num_low_sub_blocks
    ql = ql.reshape(*batch_shape, num_low_sub_blocks, elems_per_low_sub_block)
    ql = _pack_k_qs_part(ql, num_low_bits)
    bytes_per_low_block = block_size * num_low_bits // 8
    ql = ql.reshape(*batch_shape, bytes_per_low_block)

    if qh is not None:
        elems_per_high_sub_block = block_size // num_high_sub_blocks
        qh = qh.reshape(*batch_shape, num_high_sub_blocks, elems_per_high_sub_block)
        qh = _pack_k_qs_part(qh, num_high_bits)
        bytes_per_high_block = block_size * num_high_bits // 8
        qh = qh.reshape(*batch_shape, bytes_per_high_block)

    return ql, qh


# Q6_K

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
    qs = _unpack_k_qs(
        ql = blocks['ql'],
        num_low_bits = 4,
        num_low_sub_blocks = 2,
        qh = blocks['qh'],
        num_high_bits = 2,
        num_high_sub_blocks = 2,
    )
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

    qs = qs.reshape(num_blocks, QK_K)
    # Don't use += here to avoid mutating the argument.
    qs = qs + 32
    assert qs.dtype == np.int8
    qs = qs.astype(np.uint8, copy=False)

    ql, qh = _pack_k_qs(
        qs,
        num_low_bits = 4,
        num_low_sub_blocks = 2,
        num_high_bits = 2,
        num_high_sub_blocks = 2,
    )

    blocks['ql'][:] = ql
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


# Q5_K

def _unpack_scale_min_k4(
    data: npt.NDArray[np.uint8],
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """
    Unpack an array of 6-bit scales, as in `ggml-quants.c` `get_scale_min_k4`.
    """
    assert data.shape[-1] == K_SCALE_SIZE
    assert K_SCALE_SIZE == 12
    OUTPUT_SIZE = K_SCALE_SIZE * 8 // 6 // 2
    assert OUTPUT_SIZE == 8

    sc = np.zeros(data.shape[:-1] + (OUTPUT_SIZE,), dtype=np.uint8)
    m = np.zeros(data.shape[:-1] + (OUTPUT_SIZE,), dtype=np.uint8)

    # The 12 elements of the `data` array are divided up as follows:
    #
    #               |  high bits  <---------- low bits  |
    # `data[0:4]`:  | B (2b) |          A (6b)          |
    # `data[4:8]`:  | D (2b) |          C (6b)          |
    # `data[8:12]`: |      F (4b)     |      E (4b)     |
    #
    # The pieces are reassembled as follows to form `sc` and `m`:
    #
    # * `sc[0:4]`: `A`
    # * `sc[4:8]`: `E` (low bits) and `B` (high bits)
    # * `m[0:4]`: `C`
    # * `m[4:8]`: `F` (low bits) and `D` (high bits)

    # Extract `A` and `C` to initialize `sc[0:4]` and `m[0:4]`.
    mask6 = (1 << 6) - 1
    np.bitwise_and(data[..., 0:4], mask6, out=sc[..., 0:4])
    np.bitwise_and(data[..., 4:8], mask6, out=m[..., 0:4])

    # Extract `E` and `F` to set the low bits of `sc[4:8]` and `m[4:8]`.
    mask4 = (1 << 4) - 1
    np.bitwise_and(data[..., 8:12], mask4, out=sc[..., 4:8])
    np.right_shift(data[..., 8:12], 4, out=m[..., 4:8])

    # Gather `B` and `D`.  `high_bits[0:4]` is `B` and `high_bits[4:8]` is `D`.
    mask_high = ((1 << 2) - 1) << 6
    high_bits = data[..., 0:8] & mask_high
    high_bits >>= 2
    # Use `B` and `D` to set the high bits of `sc[4:8]` and `m[4:8]`.
    np.bitwise_or(sc[..., 4:8], high_bits[..., 0:4], out=sc[..., 4:8])
    np.bitwise_or(m[..., 4:8], high_bits[..., 4:8], out=m[..., 4:8])

    return sc, m

def _pack_scale_min_k4(
    sc: npt.NDArray[np.uint8],
    m: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    """
    Pack two arrays of 6-bit scales into a single array of bytes.
    """
    assert K_SCALE_SIZE == 12
    OUTPUT_SIZE = K_SCALE_SIZE * 8 // 6 // 2
    assert OUTPUT_SIZE == 8
    assert sc.shape[-1] == OUTPUT_SIZE
    assert sc.shape == m.shape

    data = np.zeros(sc.shape[:-1] + (K_SCALE_SIZE,), dtype=np.uint8)

    # Set `A` and `C`.
    mask6 = (1 << 6) - 1
    np.bitwise_and(sc[..., 0:4], mask6, out=data[..., 0:4])
    np.bitwise_and(m[..., 0:4], mask6, out=data[..., 4:8])

    # Set `E` and `F`.
    mask4 = (1 << 4) - 1
    np.bitwise_and(sc[..., 4:8], mask4, out=data[..., 8:12])
    tmp = m[..., 4:8] << 4
    np.bitwise_or(data[..., 8:12], tmp, out=data[..., 8:12])

    # Set `B` and `D`.
    tmp = sc[..., 4:8] >> 4
    tmp <<= 6
    np.bitwise_or(data[..., 0:4], tmp, out=data[..., 0:4])
    tmp = m[..., 4:8] >> 4
    tmp <<= 6
    np.bitwise_or(data[..., 4:8], tmp, out=data[..., 4:8])

    return data

def _q5_k_dtype() -> np.dtype:
    return np.dtype([
        ('d', np.float16),
        ('dmin', np.float16),
        ('scales', np.uint8, K_SCALE_SIZE),
        ('qh', np.uint8, QK_K // 8),
        ('qs', np.uint8, QK_K // 2),
    ])

def _test_unpack_scale_min_k4(
    data: npt.NDArray[np.uint8],
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    sc, m = _unpack_scale_min_k4(data)
    data2 = _pack_scale_min_k4(sc, m)
    if (data2 != data).any():
        for i in range(data.shape[0]):
            if (data2[i] != data[i]).any():
                print('mismatch at index %d' % i)
                print('  data = %s' % data[i])
                print('  sc = %s' % sc[i])
                print('  m = %s' % m[i])
                print('  data2 = %s' % data2[i])
                break
        assert False, 'bug in pack/unpack_scale_min_k4'
    return sc, m

def unpack_q5_k(
    data: npt.NDArray[np.uint8],
) -> Tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float16],
    npt.NDArray[np.float16],
]:
    blocks = data.view(_q5_k_dtype())

    d = blocks['d']
    dmin = blocks['dmin']
    sc, m = _test_unpack_scale_min_k4(blocks['scales'])

    qs = _unpack_k_qs(
        ql = blocks['qs'],
        num_low_bits = 4,
        num_low_sub_blocks = 4,
        qh = blocks['qh'],
        num_high_bits = 1,
        num_high_sub_blocks = 1,
    )

    # Each group of 32 weights uses a common `sc` and `m`.
    qs = qs.reshape(blocks.shape[0], 8, 32)

    return (qs, sc, m, d, dmin)

def dequant_q5_k(
    qs: npt.NDArray[np.uint8],
    sc: npt.NDArray[np.uint8],
    m: npt.NDArray[np.uint8],
    d: npt.NDArray[np.float16],
    dmin: npt.NDArray[np.float16],
) -> npt.NDArray[np.float16]:
    scale = d.reshape(*d.shape, 1) * sc
    minimum = dmin.reshape(*d.shape, 1) * m
    x = scale.reshape(*scale.shape, 1) * qs - minimum.reshape(*minimum.shape, 1)
    return x

def pack_q5_k(
    qs: npt.NDArray[np.uint8],
    sc: npt.NDArray[np.uint8],
    m: npt.NDArray[np.uint8],
    d: npt.NDArray[np.float16],
    dmin: npt.NDArray[np.float16],
) -> npt.NDArray[np.uint8]:
    num_blocks = d.shape[0]
    assert qs.shape == (num_blocks, 8, 32)
    assert sc.shape == (num_blocks, 8)
    assert m.shape == (num_blocks, 8)
    assert d.shape == (num_blocks,)
    assert dmin.shape == (num_blocks,)
    blocks = np.empty((num_blocks,), _q5_k_dtype())

    blocks['d'][:] = d
    blocks['dmin'][:] = dmin
    blocks['scales'][:, :] = _pack_scale_min_k4(sc, m)

    qs = qs.reshape(num_blocks, QK_K)

    ql, qh = _pack_k_qs(
        qs,
        num_low_bits = 4,
        num_low_sub_blocks = 4,
        num_high_bits = 1,
        num_high_sub_blocks = 1,
    )

    blocks['qs'][:] = ql
    blocks['qh'][:] = qh

    return blocks.view(np.uint8)

def test_unpack_q5_k(
    data: npt.NDArray[np.uint8],
) -> Tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float16],
    npt.NDArray[np.float16],
]:
    qs, sc, m, d, dmin = unpack_q5_k(data)

    data2 = pack_q5_k(qs, sc, m, d, dmin)

    if (data2 != data).any():
        blocks = data.view(_q5_k_dtype())
        blocks2 = data2.view(_q5_k_dtype())
        if (blocks2['d'] != blocks['d']).any():
            print('d mismatch')
        if (blocks2['dmin'] != blocks['dmin']).any():
            print('dmin mismatch')
        if (blocks2['scales'] != blocks['scales']).any():
            print('scales mismatch')
        if (blocks2['qs'] != blocks['qs']).any():
            print('qs mismatch')
        if (blocks2['qh'] != blocks['qh']).any():
            print('qh mismatch')
        assert False, 'bug in pack/unpack_q5_k'

    return qs, sc, m, d, dmin


# Q4_K

def _q4_k_dtype() -> np.dtype:
    return np.dtype([
        ('d', np.float16),
        ('dmin', np.float16),
        ('scales', np.uint8, K_SCALE_SIZE),
        ('qs', np.uint8, QK_K // 2),
    ])

def unpack_q4_k(
    data: npt.NDArray[np.uint8],
) -> Tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float16],
    npt.NDArray[np.float16],
]:
    blocks = data.view(_q4_k_dtype())

    d = blocks['d']
    dmin = blocks['dmin']
    sc, m = _test_unpack_scale_min_k4(blocks['scales'])

    qs = _unpack_k_qs(
        ql = blocks['qs'],
        num_low_bits = 4,
        num_low_sub_blocks = 4,
    )

    # Each group of 32 weights uses a common `sc` and `m`.
    qs = qs.reshape(blocks.shape[0], 8, 32)

    return (qs, sc, m, d, dmin)

def dequant_q4_k(
    qs: npt.NDArray[np.uint8],
    sc: npt.NDArray[np.uint8],
    m: npt.NDArray[np.uint8],
    d: npt.NDArray[np.float16],
    dmin: npt.NDArray[np.float16],
) -> npt.NDArray[np.float16]:
    scale = d.reshape(*d.shape, 1) * sc
    minimum = dmin.reshape(*d.shape, 1) * m
    x = scale.reshape(*scale.shape, 1) * qs - minimum.reshape(*minimum.shape, 1)
    return x

def pack_q4_k(
    qs: npt.NDArray[np.uint8],
    sc: npt.NDArray[np.uint8],
    m: npt.NDArray[np.uint8],
    d: npt.NDArray[np.float16],
    dmin: npt.NDArray[np.float16],
) -> npt.NDArray[np.uint8]:
    num_blocks = d.shape[0]
    assert qs.shape == (num_blocks, 8, 32)
    assert sc.shape == (num_blocks, 8)
    assert m.shape == (num_blocks, 8)
    assert d.shape == (num_blocks,)
    assert dmin.shape == (num_blocks,)
    blocks = np.empty((num_blocks,), _q4_k_dtype())

    blocks['d'][:] = d
    blocks['dmin'][:] = dmin
    blocks['scales'][:, :] = _pack_scale_min_k4(sc, m)

    qs = qs.reshape(num_blocks, QK_K)

    ql, qh = _pack_k_qs(
        qs,
        num_low_bits = 4,
        num_low_sub_blocks = 4,
    )
    assert qh is None

    blocks['qs'][:] = ql

    return blocks.view(np.uint8)

def test_unpack_q4_k(
    data: npt.NDArray[np.uint8],
) -> Tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float16],
    npt.NDArray[np.float16],
]:
    qs, sc, m, d, dmin = unpack_q4_k(data)

    data2 = pack_q4_k(qs, sc, m, d, dmin)

    if (data2 != data).any():
        blocks = data.view(_q4_k_dtype())
        blocks2 = data2.view(_q4_k_dtype())
        if (blocks2['d'] != blocks['d']).any():
            print('d mismatch')
        if (blocks2['dmin'] != blocks['dmin']).any():
            print('dmin mismatch')
        if (blocks2['scales'] != blocks['scales']).any():
            print('scales mismatch')
        if (blocks2['qs'] != blocks['qs']).any():
            print('qs mismatch')
        assert False, 'bug in pack/unpack_q4_k'

    return qs, sc, m, d, dmin


# Q3_K

def _q3_k_dtype() -> np.dtype:
    return np.dtype([
        ('hmask', np.uint8, QK_K // 8),
        ('qs', np.uint8, QK_K // 4),
        ('scales', np.uint8, 12),
        ('d', np.float16),
    ])

def unpack_q3_k(
    data: npt.NDArray[np.uint8],
) -> Tuple[
    npt.NDArray[np.int8],
    npt.NDArray[np.int8],
    npt.NDArray[np.float16],
]:
    blocks = data.view(_q3_k_dtype())

    d = blocks['d']

    scales = _unpack_k_qs(
        ql = blocks['scales'][..., 0:8],
        num_low_bits = 4,
        num_low_sub_blocks = 1,
        qh = blocks['scales'][..., 8:12],
        num_high_bits = 2,
        num_high_sub_blocks = 1,
    )
    assert scales.dtype == np.uint8
    scales = scales.astype(np.int8, copy=False)
    scales -= 32

    qs = _unpack_k_qs(
        ql = blocks['qs'],
        num_low_bits = 2,
        num_low_sub_blocks = 2,
        qh = blocks['hmask'],
        num_high_bits = 1,
        num_high_sub_blocks = 1,
    )
    assert qs.dtype == np.uint8
    qs = qs.astype(np.int8, copy=False)
    qs -= 4

    # Each group of 16 weights uses a common scale.
    qs = qs.reshape(blocks.shape[0], 16, 16)

    return (qs, scales, d)

def dequant_q3_k(
    qs: npt.NDArray[np.int8],
    scales: npt.NDArray[np.int8],
    d: npt.NDArray[np.float16],
) -> npt.NDArray[np.float16]:
    # Note the order of operations here.  Doing `q * shape` first would produce
    # a `uint8` output, causing issues with overflow.
    x = (d.reshape(*d.shape, 1, 1) * scales.reshape(*scales.shape, 1)) * qs
    return x

def pack_q3_k(
    qs: npt.NDArray[np.int8],
    scales: npt.NDArray[np.int8],
    d: npt.NDArray[np.float16],
) -> npt.NDArray[np.uint8]:
    num_blocks = d.shape[0]
    assert d.shape == (num_blocks,)
    assert scales.shape == (num_blocks, 16) # FIXME
    assert qs.shape == (num_blocks, 16, 16)
    blocks = np.empty((num_blocks,), _q3_k_dtype())

    blocks['d'][:] = d

    # Don't use += here to avoid mutating the argument.
    scales = scales + 32
    assert scales.dtype == np.int8
    scales = scales.astype(np.uint8, copy=False)
    scales_low, scales_high = _pack_k_qs(
        scales,
        num_low_bits = 4,
        num_low_sub_blocks = 1,
        num_high_bits = 2,
        num_high_sub_blocks = 1,
    )
    blocks['scales'][..., 0:8] = scales_low
    blocks['scales'][..., 8:12] = scales_high

    qs = qs.reshape(num_blocks, QK_K)
    # Don't use += here to avoid mutating the argument.
    qs = qs + 4
    assert qs.dtype == np.int8
    qs = qs.astype(np.uint8, copy=False)

    ql, qh = _pack_k_qs(
        qs,
        num_low_bits = 2,
        num_low_sub_blocks = 2,
        num_high_bits = 1,
        num_high_sub_blocks = 1,
    )

    blocks['qs'][:] = ql
    blocks['hmask'][:] = qh

    return blocks.view(np.uint8)

def test_unpack_q3_k(
    data: npt.NDArray[np.uint8],
) -> Tuple[
    npt.NDArray[np.int8],
    npt.NDArray[np.int8],
    npt.NDArray[np.float16],
]:
    qs, scales, d = unpack_q3_k(data)

    data2 = pack_q3_k(qs, scales, d)

    if (data2 != data).any():
        blocks = data.view(_q3_k_dtype())
        blocks2 = data2.view(_q3_k_dtype())
        if (blocks2['hmask'] != blocks['hmask']).any():
            print('hmask mismatch')
        if (blocks2['qs'] != blocks['qs']).any():
            print('qs mismatch')
        if (blocks2['scales'] != blocks['scales']).any():
            print('scales mismatch')
        if (blocks2['d'] != blocks['d']).any():
            print('d mismatch')
        assert False, 'bug in pack/unpack_q3_k'

    return qs, scales, d


# Q2_K

def _q2_k_dtype() -> np.dtype:
    return np.dtype([
        ('scales', np.uint8, QK_K // 16),
        ('qs', np.uint8, QK_K // 4),
        ('d', np.float16),
        ('dmin', np.float16),
    ])

def unpack_q2_k(
    data: npt.NDArray[np.uint8],
) -> Tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float16],
    npt.NDArray[np.float16],
]:
    blocks = data.view(_q2_k_dtype())

    d = blocks['d']
    dmin = blocks['dmin']

    sc = blocks['scales'] & 0xf
    m = blocks['scales'] >> 4

    qs = _unpack_k_qs(
        ql = blocks['qs'],
        num_low_bits = 2,
        num_low_sub_blocks = 2,
    )

    # Each group of 16 weights uses a common `sc` and `m`.
    qs = qs.reshape(blocks.shape[0], 16, 16)

    return (qs, sc, m, d, dmin)

def dequant_q2_k(
    qs: npt.NDArray[np.uint8],
    sc: npt.NDArray[np.uint8],
    m: npt.NDArray[np.uint8],
    d: npt.NDArray[np.float16],
    dmin: npt.NDArray[np.float16],
) -> npt.NDArray[np.float16]:
    scale = d.reshape(*d.shape, 1) * sc
    minimum = dmin.reshape(*d.shape, 1) * m
    x = scale.reshape(*scale.shape, 1) * qs - minimum.reshape(*minimum.shape, 1)
    return x

def pack_q2_k(
    qs: npt.NDArray[np.uint8],
    sc: npt.NDArray[np.uint8],
    m: npt.NDArray[np.uint8],
    d: npt.NDArray[np.float16],
    dmin: npt.NDArray[np.float16],
) -> npt.NDArray[np.uint8]:
    num_blocks = d.shape[0]
    assert qs.shape == (num_blocks, 16, 16)
    assert sc.shape == (num_blocks, 16)
    assert m.shape == (num_blocks, 16)
    assert d.shape == (num_blocks,)
    assert dmin.shape == (num_blocks,)
    blocks = np.empty((num_blocks,), _q2_k_dtype())

    blocks['d'][:] = d
    blocks['dmin'][:] = dmin
    blocks['scales'][:, :] = sc & 0xf
    blocks['scales'][:, :] |= m << 4

    qs = qs.reshape(num_blocks, QK_K)

    ql, qh = _pack_k_qs(
        qs,
        num_low_bits = 2,
        num_low_sub_blocks = 2,
    )
    assert qh is None

    blocks['qs'][:] = ql

    return blocks.view(np.uint8)

def test_unpack_q2_k(
    data: npt.NDArray[np.uint8],
) -> Tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float16],
    npt.NDArray[np.float16],
]:
    qs, sc, m, d, dmin = unpack_q2_k(data)

    data2 = pack_q2_k(qs, sc, m, d, dmin)

    if (data2 != data).any():
        blocks = data.view(_q2_k_dtype())
        blocks2 = data2.view(_q2_k_dtype())
        if (blocks2['d'] != blocks['d']).any():
            print('d mismatch')
        if (blocks2['dmin'] != blocks['dmin']).any():
            print('dmin mismatch')
        if (blocks2['scales'] != blocks['scales']).any():
            print('scales mismatch')
        if (blocks2['qs'] != blocks['qs']).any():
            print('qs mismatch')
        assert False, 'bug in pack/unpack_q2_k'

    return qs, sc, m, d, dmin


# IQ2_XS

def _iq2_xs_dtype() -> np.dtype:
    return np.dtype([
        ('d', np.float16),
        ('qs', np.uint16, QK_K // 8),
        ('scales', np.uint8, QK_K // 32),
    ])

def unpack_iq2_xs(
    data: npt.NDArray[np.uint8],
) -> Tuple[
    npt.NDArray[np.uint16],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float16],
]:
    blocks = data.view(_iq2_xs_dtype())

    d = blocks['d']

    scales = _split_bits(blocks['scales'], [(0, 4), (4, 8)])

    grids = blocks['qs'] & 0x1ff
    signs = (blocks['qs'] >> 9).astype(np.uint8)

    # In each block, there are 16 scales and 32 grid/sign pairs.  The first two
    # grids are used with the first scale, the next two are used with the
    # second scale, and so on.  Reshape the arrays to reflect that.
    num_blocks = blocks.shape[0]
    scales = scales.reshape(num_blocks, 16)
    grids = grids.reshape(num_blocks, 16, 2)
    signs = signs.reshape(num_blocks, 16, 2)

    print('d', d.shape)
    print('scales', scales.shape)
    print('grids', grids.shape)
    print('signs', signs.shape)

    return (grids, signs, scales, d)

IQ2_XS_GRID_TABLE = np.array([
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
], dtype=np.uint64).view(np.uint8).reshape(-1, 8)

IQ2_XS_SIGN_TABLE = np.array([
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
], dtype=np.uint8)

def dequant_iq2_xs(
    grids: npt.NDArray[np.uint16],
    signs: npt.NDArray[np.uint8],
    scales: npt.NDArray[np.uint8],
    d: npt.NDArray[np.float16],
) -> npt.NDArray[np.float16]:
    grid_vals = IQ2_XS_GRID_TABLE[grids]
    print('grid_vals', grid_vals.shape)
    sign_vals = IQ2_XS_SIGN_TABLE[signs]
    print('sign_vals', sign_vals.shape)
    sign_bits = _split_bits(sign_vals, [(i, i + 1) for i in range(8)])
    print('sign_bits', sign_bits.shape)
    sign_floats = np.array([1, -1], dtype=np.float16)[sign_bits]
    print('sign_floats', sign_floats.shape)
    print('d', d.shape)
    print('scales', scales.shape)

    db = np.float16(0.5) + scales
    db *= 0.25
    db *= d.reshape(*d.shape, 1)
    print('db', db.shape, db.dtype)

    x = db.reshape(*db.shape, 1, 1) * grid_vals * sign_floats
    return x.reshape(*x.shape[:-3], 256)

def pack_iq2_xs(
    grids: npt.NDArray[np.uint16],
    signs: npt.NDArray[np.uint8],
    scales: npt.NDArray[np.uint8],
    d: npt.NDArray[np.float16],
) -> npt.NDArray[np.uint8]:
    num_blocks = d.shape[0]
    assert grids.shape == (num_blocks, 16, 2)
    assert signs.shape == (num_blocks, 16, 2)
    assert scales.shape == (num_blocks, 16)
    assert d.shape == (num_blocks,)
    blocks = np.empty((num_blocks,), _iq2_xs_dtype())

    blocks['d'][:] = d
    blocks['scales'][:, :] = scales[..., 0::2] & 0xf
    blocks['scales'][:, :] |= scales[..., 1::2] << 4

    grids = grids.reshape(num_blocks, 32)
    signs = signs.reshape(num_blocks, 32)
    np.bitwise_and(grids, 0x1ff, out=blocks['qs'][:, :])
    blocks['qs'][:, :] |= signs.astype(np.uint16) << 9

    return blocks.view(np.uint8)

def test_unpack_iq2_xs(
    data: npt.NDArray[np.uint8],
) -> Tuple[
    npt.NDArray[np.uint16],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float16],
]:
    grids, signs, scales, d = unpack_iq2_xs(data)

    data2 = pack_iq2_xs(grids, signs, scales, d)

    if (data2 != data).any():
        blocks = data.view(_iq2_xs_dtype())
        blocks2 = data2.view(_iq2_xs_dtype())
        if (blocks2['d'] != blocks['d']).any():
            print('d mismatch')
        if (blocks2['scales'] != blocks['scales']).any():
            print('scales mismatch')
        if (blocks2['qs'] != blocks['qs']).any():
            print('qs mismatch')
            print('  old[0] = ', blocks['qs'][0])
            print('  new[0] = ', blocks2['qs'][0])
            print('  grids = ', grids[0].reshape(-1))
            print('  signs = ', signs[0].reshape(-1))
        assert False, 'bug in pack/unpack_iq2_xs'

    return grids, signs, scales, d
