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
    assert ql.shape[-1] == QK_K // (8 // num_low_bits)
    assert QK_K % num_low_sub_blocks == 0
    if qh is not None:
        assert 8 % num_high_bits == 0
        assert qh.shape[-1] == QK_K // (8 // num_high_bits)
        assert qh.shape[:-1] == ql.shape[:-1]
        assert QK_K % num_high_sub_blocks == 0

    batch_shape = ql.shape[:-1]

    # Divide each block (the last axis) into sub-blocks.  Different K-quants
    # use different numbers of sub-blocks.
    bytes_per_low_sub_block = QK_K * num_low_bits // 8 // num_low_sub_blocks
    ql = ql.reshape(*batch_shape, num_low_sub_blocks, bytes_per_low_sub_block)
    ql = _unpack_k_qs_part(ql, num_low_bits)
    ql = ql.reshape(*batch_shape, QK_K)

    if qh is not None:
        bytes_per_high_sub_block = QK_K * num_high_bits // 8 // num_high_sub_blocks
        qh = qh.reshape(*batch_shape, num_high_sub_blocks, bytes_per_high_sub_block)
        qh = _unpack_k_qs_part(qh, num_high_bits)
        qh = qh.reshape(*batch_shape, QK_K)

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
    assert qs.shape[-1] == QK_K
    assert 8 % num_low_bits == 0
    assert QK_K % num_low_sub_blocks == 0
    if num_high_bits != 0:
        assert 8 % num_high_bits == 0
        assert QK_K % num_high_sub_blocks == 0

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
    elems_per_low_sub_block = QK_K // num_low_sub_blocks
    ql = ql.reshape(*batch_shape, num_low_sub_blocks, elems_per_low_sub_block)
    ql = _pack_k_qs_part(ql, num_low_bits)
    bytes_per_low_block = QK_K * num_low_bits // 8
    ql = ql.reshape(*batch_shape, bytes_per_low_block)

    if qh is not None:
        elems_per_high_sub_block = QK_K // num_high_sub_blocks
        qh = qh.reshape(*batch_shape, num_high_sub_blocks, elems_per_high_sub_block)
        qh = _pack_k_qs_part(qh, num_high_bits)
        bytes_per_high_block = QK_K * num_high_bits // 8
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
