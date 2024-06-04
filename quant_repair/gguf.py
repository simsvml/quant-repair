from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import os
import struct
import tempfile
from typing import Any, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda it, **kwargs: it


# BEGIN code copied from gguf-py/gguf/constants.py

# License:
#
# MIT License
# 
# Copyright (c) 2023 Georgi Gerganov
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

GGUF_MAGIC             = 0x46554747
GGUF_VERSION           = 3
GGUF_DEFAULT_ALIGNMENT = 32

class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30

# END code copied from gguf-py/gguf/constants.py


DTYPE_HEADER = np.dtype([
    ('magic', np.uint32),
    ('version', np.uint32),
    ('tensor_count', np.uint64),
    ('kv_count', np.uint64),
])

METADATA_VALUE_DTYPE = {
    GGUFValueType.UINT8: np.uint8,
    GGUFValueType.INT8: np.int8,
    GGUFValueType.UINT16: np.uint16,
    GGUFValueType.INT16: np.int16,
    GGUFValueType.UINT32: np.uint32,
    GGUFValueType.INT32: np.int32,
    GGUFValueType.FLOAT32: np.float32,
    GGUFValueType.BOOL: np.bool_,
    # GGUFValueType.STRING: variable length
    # GGUFValueType.ARRAY: variable length
    GGUFValueType.UINT64: np.uint64,
    GGUFValueType.INT64: np.int64,
    GGUFValueType.FLOAT64: np.float64,
}



class HeaderParser:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def take(self, n):
        off = self.offset
        end = off + int(n)
        self.offset = end
        return self.data[off : end]

    def skip(self, n):
        self.offset += int(n)

    def take_value(self, dtype):
        dtype = np.dtype(dtype)
        return self.take(dtype.itemsize).view(dtype)[0]

    def take_values(self, dtype, n):
        dtype = np.dtype(dtype)
        return self.take(dtype.itemsize * n).view(dtype)

    def take_string(self):
        length = self.take_value(np.uint64)
        return bytes(self.take(length)).decode('utf-8')

    def skip_string(self):
        length = self.take_value(np.uint64)
        self.skip(length)

    def skip_metadata_type_and_value(self):
        value_type = GGUFValueType(self.take_value(np.uint32))
        self.skip_metadata_value(value_type)

    def skip_metadata_value(self, value_type: GGUFValueType):
        if (dtype := METADATA_VALUE_DTYPE.get(value_type)):
            self.skip(np.dtype(dtype).itemsize)
        elif value_type == GGUFValueType.STRING:
            self.skip_string()
        elif value_type == GGUFValueType.ARRAY:
            elem_type = GGUFValueType(self.take_value(np.uint32))
            length = self.take_value(np.uint64)
            for i in range(length):
                self.skip_metadata_value(elem_type)
        else:
            raise ValueError('unknown metadata value type %r' % (value_type,))


def align_offset(offset, align):
    return offset + (align - (offset % align)) % align


@dataclass(frozen = True)
class KVEntry:
    key: str
    offset: int
    offset_end: int

@dataclass(frozen = True)
class TensorEntry:
    name: str
    info_offset: int
    info_offset_end: int
    data_offset: int

class GGUFReader2:
    def __init__(self, path, mode = 'r'):
        self.mem = np.memmap(path, mode = mode)

        p = HeaderParser(self.mem)

        header = p.take_value(DTYPE_HEADER)
        print(header)
        assert header['magic'] == GGUF_MAGIC, 'bad magic %x' % header['magic']
        assert header['version'] == GGUF_VERSION, 'bad version %d' % header['version']
        tensor_count = header['tensor_count']
        kv_count = header['kv_count']

        # Determine metadata/KV start and end offsets.
        self.kvs = []
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self.architecture = None
        for i in range(kv_count):
            start = p.offset
            key = p.take_string()
            p.skip_metadata_type_and_value()
            end = p.offset
            self.kvs.append(KVEntry(key, start, end))

            if key == 'general.alignment':
                sub_parser = HeaderParser(self.mem[start : end])
                sub_parser.skip_string()
                value_type = GGUFValueType(sub_parser.take_value(np.uint32))
                dtype = METADATA_VALUE_DTYPE[value_type]
                value = sub_parser.take_value(dtype)
                self.alignment = value

            if key == 'general.architecture':
                sub_parser = HeaderParser(self.mem[start : end])
                sub_parser.skip_string()
                value_type = GGUFValueType(sub_parser.take_value(np.uint32))
                assert value_type == GGUFValueType.STRING
                value = sub_parser.take_string()
                self.architecture = value

        # Determine tensor start and end offsets.
        self.tensors = []
        for i in range(tensor_count):
            start = p.offset
            name = p.take_string()
            n_dimensions = p.take_value(np.uint32)
            p.skip(n_dimensions * np.dtype(np.uint64).itemsize)   # dimensions
            p.skip(np.dtype(np.uint32).itemsize)  # type
            data_offset = p.take_value(np.uint64)
            end = p.offset
            self.tensors.append(TensorEntry(name, start, end, data_offset))

        self.tensor_base = align_offset(p.offset, self.alignment)


class GGUFWriter2:
    def __init__(self, path, alignment: int = GGUF_DEFAULT_ALIGNMENT):
        self.path = path
        self.temp_path = path + '.tmp'
        self.file = open(self.temp_path, 'wb')
        self.alignment = alignment

        # Writing proceeds in several stages:
        #
        # 1. Add tensors and metadata/KV entries.  KV entries are written
        #    directly to disk.  Tensor info is buffered in memory.  Tensor data
        #    is not provided in this phase.
        # 2. `finish_header()`.  The file header is written to disk.  No more
        #    KV entries or tensors may be added after this point.
        # 3. Write tensor data.  Data is written to disk for each tensor.  The
        #    tensor descriptions in memory have their `offset` fields updated
        #    accordingly.
        # 4. `finish_data()`.  Tensor descriptions, including correct offsets,
        #    are written to disk.
        #
        # The general idea is to write large items (KV entries, tensor data)
        # directly to disk and only buffer small items (tensor descriptions,
        # file header).  We use `self.file.seek(...)` to allow writing
        # different items out of order.
        self.finished_header = False
        self.finished_data = False

        # Updated during header writing (stage 1)
        self.kv_count = 0
        self.tensor_descs = {}

        # Updated during data writing (stage 3)
        self.finished_tensor_descs = []
        self.finished_tensor_names = set()
        self.tensor_base = 0
        self.tensor_desc_offset = 0

        # Position the file cursor to start writing KV entries.
        self.file.seek(DTYPE_HEADER.itemsize)

    def write_kv_str(self, key: str, value: str):
        assert not self.finished_header, "can't add KV items after finish_header()"
        key_bytes = key.encode('utf-8')
        value_bytes = value.encode('utf-8')
        dtype = np.dtype([
            ('key_len', np.uint64),
            ('key', np.uint8, (len(key_bytes),)),
            ('value_type', np.uint32),
            ('value_len', np.uint64),
            ('value', np.uint8, (len(value_bytes),)),
        ])
        arr = np.array([(
            len(key_bytes),
            list(key_bytes),
            GGUFValueType.STRING,
            len(value_bytes),
            list(value_bytes),
        )], dtype = dtype)
        self.file.write(arr)

        self.kv_count += 1

    def copy_kv(self, reader: GGUFReader2, kv_entry: KVEntry):
        assert not self.finished_header, "can't add KV items after finish_header()"
        self.file.write(reader.mem[kv_entry.offset : kv_entry.offset_end])
        self.kv_count += 1

    def copy_kv_with_key(self, reader: GGUFReader2, key: str, kv_entry: KVEntry):
        """
        Like `copy_kv(reader, kv_entry)`, but replace the key from `kv_entry`
        with `key`.
        """
        assert not self.finished_header, "can't add KV items after finish_header()"
        mem = reader.mem[kv_entry.offset : kv_entry.offset_end]
        p = HeaderParser(mem)
        p.skip_string()
        value_start = p.offset

        key_bytes = key.encode('utf-8')

        dtype = np.dtype([
            ('key_len', np.uint64),
            ('key', np.uint8, (len(key_bytes),)),
            ('value_bytes', np.uint8, (len(mem) - value_start,)),
        ])
        arr = np.array([(
            len(key_bytes),
            list(key_bytes),
            mem[value_start:],
        )], dtype = dtype)
        self.file.write(arr)

        self.kv_count += 1

    def add_tensor(self, name: str, dimensions: Sequence[int], type_: GGMLQuantizationType):
        """
        Reserve space for a tensor description.  Data for this tensor must be
        provided after calling `finish_header()`.
        """
        assert not self.finished_header, "can't add tensor descriptions after finish_header()"
        assert name not in self.tensor_descs, 'duplicate tensor name %r' % name
        name_bytes = name.encode('utf-8')
        dimensions = tuple(dimensions)
        dtype = np.dtype([
            ('name_len', np.uint64),
            ('name', np.uint8, (len(name_bytes),)),
            ('n_dimensions', np.uint32),
            ('dimensions', np.uint64, (len(dimensions),)),
            ('type_', np.uint32),
            ('offset', np.uint64),
        ])
        arr = np.array([(
            len(name_bytes),
            list(name_bytes),
            len(dimensions),
            list(dimensions),
            int(type_),
            0,  # offset
        )], dtype = dtype)
        self.tensor_descs[name] = arr

    def copy_tensor(self, reader: GGUFReader2, tensor_entry: TensorEntry):
        """
        Copy a tensor description from a reader.  Data for this tensor must be
        provided after calling `finish_header()`.
        """
        assert not self.finished_header, "can't add tensor descriptions after finish_header()"
        name = tensor_entry.name
        assert name not in self.tensor_descs, 'duplicate tensor name %r' % name
        entry_bytes = reader.mem[tensor_entry.info_offset : tensor_entry.info_offset_end]
        dtype = np.dtype([
            ('unknown', np.uint8, (len(entry_bytes) - np.dtype(np.uint64).itemsize,)),
            ('offset', np.uint64),
        ])
        arr = entry_bytes.copy().view(dtype)
        self.tensor_descs[name] = arr

    def copy_tensors(
        self,
        reader: GGUFReader2,
        tensor_entries: Optional[Sequence[TensorEntry]] = None,
    ):
        """
        Copy multiple tensor descriptions from a reader.  If `tensor_entries`
        is not provided, the default is to copy all tensors in `reader`.  Data
        for these tensors must be provided after calling `finish_header()`.
        """
        if tensor_entries is None:
            tensor_entries = reader.tensors
        for tensor in tensor_entries:
            self.copy_tensor(reader, tensor)

    def finish_header(self):
        assert not self.finished_header, "already called finish_header()"
        self.finished_header = True

        kv_end = self.file.tell()
        self.tensor_desc_offset = kv_end
        tensor_desc_end = kv_end \
            + sum(len(arr) * arr.dtype.itemsize for arr in self.tensor_descs.values())
        self.tensor_base = align_offset(tensor_desc_end, self.alignment)

        self.file.seek(0)
        header = np.array([(
            GGUF_MAGIC,
            GGUF_VERSION,
            len(self.tensor_descs),
            self.kv_count,
        )], dtype = DTYPE_HEADER)
        self.file.write(header)

        self.file.seek(self.tensor_base)
        self.data_offset = 0

    def _align_output(self) -> int:
        """
        Align the output position to a multiple of `self.alignment`, and return
        the new position.
        """
        pos = self.file.tell()
        align = self.alignment
        off = pos % align
        if off != 0:
            padding = (align - off) % align
            self.file.seek(padding, 1)
            pos += padding
        return pos

    def _write_with_progress(self, data):
        BLOCK_SIZE = 1024 * 1024
        if len(data) <= BLOCK_SIZE:
            self.file.write(data)
            return

        for i in tqdm(range(0, len(data), BLOCK_SIZE), desc = 'writing blocks'):
            block = data[i : i + BLOCK_SIZE]
            self.file.write(block)

    def _pop_tensor_desc_for_write(self, name: str) -> Any:
        """
        Move the tensor description for `name` from `self.tensor_descs` to
        `self.finished_tensor_descs` and return it.
        """
        arr = self.tensor_descs.pop(name, None)
        if arr is None:
            if name in self.finished_tensor_names:
                raise AssertionError('data was already written for tensor %r' % name)
            else:
                raise AssertionError('no tensor named %r was added to the file' % name)
        self.finished_tensor_descs.append(arr)
        self.finished_tensor_names.add(name)
        return arr

    def write_tensor_data(self, name: str, data):
        assert self.finished_header, "must call finish_header() first"
        assert not self.finished_data, "already called finish_data()"

        arr = self._pop_tensor_desc_for_write(name)
        pos = self._align_output()
        arr['offset'] = pos - self.tensor_base
        self._write_with_progress(data)
        print('%s: wrote at offset %d' % (name, arr['offset']))

    def copy_tensor_data(
        self,
        reader: GGUFReader2,
        tensor_entries: Optional[Sequence[TensorEntry]] = None,
        offset: Optional[int] = None,
        end_offset: Optional[int] = None,
    ):
        """
        Copy data for multiple tensors from `reader`.  Reads data in the range
        `offset : offset_end` (default: `reader.tensor_base :`), appends it to
        the output file, and updates tensors listed in `tensor_entries`
        (default: `reader.tensors`) with offsets into that data.
        """
        assert self.finished_header, "must call finish_header() first"
        assert not self.finished_data, "already called finish_data()"

        if tensor_entries is None:
            tensor_entries = reader.tensors
        if offset is None:
            offset = reader.tensor_base
        dest_offset = self._align_output()
        self._write_with_progress(reader.mem[offset : end_offset])
        offset_delta = (dest_offset - self.tensor_base) - (offset - reader.tensor_base)

        for tensor_entry in tensor_entries:
            name = tensor_entry.name
            arr = self._pop_tensor_desc_for_write(name)
            arr['offset'] = tensor_entry.data_offset + offset_delta
            print('%s: mapped offset %d -> %d' % (name, tensor_entry.data_offset, arr['offset']))
            assert arr['offset'] % self.alignment == 0

    def finish_data(self):
        assert not self.finished_data, "already called finish_data()"
        self.finished_data = True
        
        assert len(self.tensor_descs) == 0, \
            'must provide data for tensors: %r' % list(self.tensor_descs.keys())

        self.file.seek(self.tensor_desc_offset)
        for desc in self.finished_tensor_descs:
            self.file.write(desc)
        assert align_offset(self.file.tell(), self.alignment) == self.tensor_base

        os.rename(self.temp_path, self.path)

