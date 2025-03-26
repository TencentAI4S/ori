# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import datetime
import os
import shutil
import struct
import sys
from functools import lru_cache
from itertools import accumulate
from typing import List, Union

import numpy as np
import torch
from tqdm import tqdm

from tgnn.utils.io import print_rank_0, get_file_timestamp

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16
}


def best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)

    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


class IndexedDataset(torch.utils.data.Dataset):
    """Loader for IndexedDataset"""
    _HDR_MAGIC = b'TNTIDX\x00\x00'

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None  # lazay read binary data
        self.read_index(self.index_file)

    @property
    def index_file(self):
        return f"{self.path}.idx"

    @property
    def bin_file(self):
        return f"{self.path}.bin"

    def read_index(self, path):
        with open(path, 'rb') as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                'Index file doesn\'t match expected format. '
                'Make sure that --dataset-impl is configured properly.'
            )
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack('<QQ', f.read(16))
            self.doc_count = struct.unpack('<Q', f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)
            self.doc_idx = read_longs(f, self.doc_count)

    def read_data(self, path):
        self.data_file = open(path, 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def get_chunk_data(self, start, size):
        a = np.empty(size, dtype=self.dtype)
        self.data_file.seek(start * self.element_size)
        self.data_file.readinto(a)

        return a

    def get_chunk_list(self, start, sizes):
        size = sum(sizes)
        a = self.get_chunk_data(start, size)
        offsets = list(accumulate(sizes))
        sents = np.split(a, offsets[:-1])

        return sents

    def __getitem__(self, idx):
        assert isinstance(idx, (int, slice))
        if isinstance(idx, int):
            if not self.data_file:
                self.read_data(self.bin_file)
            i = idx
            self.check_index(i)
            size = self.sizes[self.dim_offsets[i]: self.dim_offsets[i + 1]]
            return self.get_chunk_data(self.dim_offsets[i], size)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sizes = self.sizes[self.dim_offsets[start]:self.dim_offsets[stop]]
            return self.get_chunk_list(self.dim_offsets[start], sizes)

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path):
        super().__init__(path)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return

        if not self.data_file:
            self.read_data(self.path)

        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]

        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx: ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size

        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    def __getitem__(self, idx):
        assert isinstance(idx, (int, slice))

        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            ptx = self.cache_index[i]
            np.copyto(a, self.cache[ptx: ptx + a.size])

            return a

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            sents = []
            for i in range(start, stop, step):
                sents.append(self[i])
            return sents


class IndexedDatasetBuilder:
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float32: 4,
        np.float64: 8
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = np.dtype(self.dtype).itemsize
        self.doc_idx = [0]

    def add_item(self, tensor):
        bytes = self.out_file.write(np.array(tensor.numpy(), dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def end_document(self):
        self.doc_idx.append(len(self.sizes))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        doc_offset = len(self.sizes)
        begin = self.data_offsets[-1]
        for data_offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + data_offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)
        self.doc_idx.extend((doc_offset + index.doc_idx)[1:])

        with open(another_file + ".bin", 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(IndexedDataset._HDR_MAGIC)
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        index.write(struct.pack('<Q', len(self.doc_idx)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        write_longs(index, self.doc_idx)
        index.close()


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


def exscan_from_cumsum_(arr):
    # given an array holding the result of an inclusive scan (cumsum),
    # convert to an exclusive scan (shift to the right)
    # [10, 30, 35, 50] --> [0, 10, 30, 35]
    if arr.size > 1:
        arr[1:] = arr[:-1]
    if arr.size > 0:
        arr[0] = 0


def get_pointers_with_total(sizes, elemsize, dtype):
    """Return a numpy array of type np.dtype giving the byte offsets.

    Multiplies values in the sizes array by elemsize (bytes),
    and then computes an exclusive scan to get byte offsets.
    Returns the total number of bytes as second item in a tuple.
    """

    # scale values in sizes array by elemsize to get sizes in bytes
    pointers = np.array(sizes, dtype=dtype)
    pointers *= elemsize
    np.cumsum(pointers, axis=0, out=pointers)

    # get total number of bytes from all sizes (last element)
    bytes_last = pointers[-1] if len(sizes) > 0 else 0

    # convert to byte offsets
    exscan_from_cumsum_(pointers)

    return pointers, bytes_last


class MMIndex:
    _HDR_MAGIC = b'MMIDIDX\x00\x00'

    @classmethod
    def write_header(cls, fout, dtype, numsizes, numdocs):
        """Writes header for mmap indexed dataset to given file handle, return number of bytes written."""
        startpos = fout.tell()
        fout.write(cls._HDR_MAGIC)
        fout.write(struct.pack('<Q', 1))
        fout.write(struct.pack('<B', code(dtype)))
        fout.write(struct.pack('<Q', numsizes))
        fout.write(struct.pack('<Q', numdocs))

        endpos = fout.tell()
        return endpos - startpos

    @classmethod
    def writer(cls, path, dtype):
        class _Writer:
            def __enter__(self):
                self._file = open(path, 'wb')
                return self

            @staticmethod
            def _get_pointers(sizes, npdtype):
                """Return a numpy array of byte offsets given a list of sizes.
                Multiplies values in the sizes array by dtype size (bytes),
                and then computes an exclusive scan to get byte offsets.
                """

                # compute element sizes in bytes
                pointers, _ = get_pointers_with_total(sizes, dtype().itemsize, npdtype)
                return pointers

            def write(self, sizes, doc_idx: Union[List, np.ndarray] = None):
                if doc_idx is not None:
                    doc_idx = np.array(doc_idx, dtype=np.int64)
                else:
                    doc_idx = np.arange(len(sizes), dtype=np.int64)

                cls.write_header(self._file, dtype, len(sizes), len(doc_idx))
                sizes32 = np.array(sizes, dtype=np.int32)
                self._file.write(sizes32.tobytes(order='C'))
                del sizes32

                pointers = self._get_pointers(sizes, np.int64)
                self._file.write(pointers.tobytes(order='C'))
                del pointers

                self._file.write(doc_idx.tobytes(order='C'))

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._file.close()

        return _Writer()

    @classmethod
    def is_index(cls, filename):
        file_magic = cls._HDR_MAGIC
        magic = open(filename, mode="rb").read(len(file_magic))
        return magic == file_magic

    def __init__(self, path, skip_warmup=False):
        with open(path, 'rb') as stream:
            magic_test = stream.read(len(self._HDR_MAGIC))
            assert self._HDR_MAGIC == magic_test, (
                'Index file doesn\'t match expected format. '
                'Make sure that --dataset-impl is configured properly.'
            )
            version = struct.unpack('<Q', stream.read(8))
            assert (1,) == version
            dtype_code, = struct.unpack('<B', stream.read(1))
            self._dtype = dtypes[dtype_code]
            self._dtype_size = self._dtype().itemsize
            self._len = struct.unpack('<Q', stream.read(8))[0]
            self._doc_count = struct.unpack('<Q', stream.read(8))[0]
            offset = stream.tell()

        if not skip_warmup:
            print_rank_0("    warming up index mmap file...")
            _warmup_mmap_file(path)

        self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        print_rank_0("    reading sizes...")
        self._sizes = np.frombuffer(
            self._bin_buffer,
            dtype=np.int32,
            count=self._len,
            offset=offset)
        print_rank_0("    reading pointers...")
        self._pointers = np.frombuffer(self._bin_buffer,
                                       dtype=np.int64,
                                       count=self._len,
                                       offset=offset + self._sizes.nbytes)
        print_rank_0("    reading document index...")
        self._doc_idx = np.frombuffer(self._bin_buffer,
                                      dtype=np.int64,
                                      count=self._doc_count,
                                      offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        self.buffer_size = np.sum(self._sizes, dtype=np.int64)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap

    @property
    def data_buffer_ptr(self):
        return self._pointers[0]

    @property
    def pointers(self):
        return self._pointers

    @property
    def dtype(self):
        return self._dtype

    @property
    def sizes(self):
        return self._sizes

    @property
    def doc_idx(self):
        return self._doc_idx

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self._pointers[i], self._sizes[i]

    def __len__(self):
        return self._len


class MMapIndexedDataset(torch.utils.data.Dataset):

    def __init__(self, path, skip_warmup=True):
        super().__init__()
        self._path = None
        self._index = None
        self._bin_buffer = None
        self._bin_buffer_mmap = None
        assert os.path.isfile(path), f"not exist file: {path}"
        self._do_init(path, skip_warmup)

    @property
    def index_file(self):
        return f"{self._path}.idx"

    @property
    def bin_file(self):
        return self._path

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def check_index_timestamp(self, bin_file, idx_file, interval=100):
        t1 = get_file_timestamp(bin_file)
        t2 = get_file_timestamp(idx_file)
        assert abs(t1 - t2) < interval, (f"index file have changed, please rebuild index file\n"
                                         f"Binary: {datetime.datetime.fromtimestamp(t1)}\n"
                                         f"Index: {datetime.datetime.fromtimestamp(t2)}")

    def _do_init(self, path, skip_warmup=True):
        self._path = path
        self._index = self.read_index(self.index_file, skip_warmup)
        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(self.bin_file)

        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(self.bin_file, mode='r', order='C')
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def read_index(self, path, skip_warmup=True):
        assert MMIndex.is_index(path), f"{path} is not valid index file"
        self.check_index_timestamp(self.bin_file, path)
        return MMIndex(path, skip_warmup=skip_warmup)

    def __del__(self):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()

        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @property
    def buffer_size(self):
        return self._index.buffer_size

    @property
    def data_buffer_ptr(self):
        return self._index._pointers[0]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer,
                                     dtype=self._index.dtype,
                                     count=size,
                                     offset=ptr)
            return np_array

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")

            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer,
                                     dtype=self._index.dtype,
                                     count=total_size,
                                     offset=ptr)
            sents = np.split(np_array, offsets[:-1])

            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.
        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer,
                                 dtype=self._index.dtype,
                                 count=length,
                                 offset=ptr)
        return np_array

    def get_data_buffer(self, offset, length):
        ptr = self.data_buffer_ptr
        assert 0 <= offset < self.buffer_size - 1 and 1 <= length <= self.buffer_size, \
            f"buffer size:{self.buffer_size}，offset: {offset}, length: {length}"

        item_size = np.dtype(self._index.dtype).itemsize
        if offset + length > self.buffer_size:
            arr1 = np.frombuffer(self._bin_buffer,
                                 dtype=self._index.dtype,
                                 count=self.buffer_size - offset,
                                 offset=ptr + offset * item_size)

            arr2 = np.frombuffer(self._bin_buffer,
                                 dtype=self._index.dtype,
                                 count=offset + length - self.buffer_size,
                                 offset=0)
            return np.concatenate([arr1, arr2])

        return np.frombuffer(self._bin_buffer,
                             dtype=self._index.dtype,
                             count=length,
                             offset=ptr + offset * item_size)

    @property
    def sizes(self):
        return self._index.sizes

    def size(self, index):
        return self._index.sizes[index]

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index.doc_idx

    def set_doc_idx(self, doc_idx):
        self._index.doc_idx = doc_idx

    @property
    def supports_prefetch(self):
        return False

    @property
    def dtype(self):
        return self._index.dtype


class MMapIndexedDatasetBuilder:
    IndexClass = MMIndex
    IndexDataset = MMapIndexedDataset

    def __init__(self,
                 out_file,
                 dtype=np.int64):
        self._filename = out_file
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def __len__(self):
        return len(self._sizes)

    def add_item(self, tensor):
        if isinstance(tensor, torch.Tensor):
            np_array = tensor.numpy().astype(self._dtype)
        elif isinstance(tensor, (np.ndarray, list, tuple)):
            np_array = np.asarray(tensor, dtype=self._dtype)
        else:
            np_array = np.frombuffer(tensor, dtype=self._dtype)

        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    @classmethod
    def merge_files(cls, files, filename):
        dtype = cls.IndexDataset(files[0]).dtype
        print(f"merge dtype is : {dtype}")
        builder = cls(filename, dtype=dtype)
        for path in files:
            print(f"merge file: {path}")
            assert os.path.exists(path), f"not exist file or dir: {path}"
            if os.stat(path).st_size == 0:
                print(f"it's empty file: {path}", file=sys.stderr)
                continue

            index = cls.IndexClass(path + ".idx", skip_warmup=True)
            assert index.dtype == dtype
            total_len = len(index.sizes) + len(builder)
            print_rank_0(f"    concat {path} size={len(index.sizes)} for a total size of {total_len}")
            offset = len(builder)
            builder._sizes.extend(index.sizes)
            builder._doc_idx.extend(list((offset + index.doc_idx)[1:]))
            # Concatenate data
            with open(path, 'rb') as f:
                shutil.copyfileobj(f, builder._data_file)
        builder.finalize()

    def merge_file(self, filename):
        index_file = filename + ".idx"
        index = self.IndexClass(index_file)
        assert index.dtype == self._dtype

        total_len = len(index.sizes) + len(self._sizes)
        print_rank_0(f"    concat {filename} size={len(index.sizes)} for a total size of {total_len}")

        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend((offset + index.doc_idx)[1:])
        with open(filename, 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file=None):
        index_file = index_file or self._filename + ".idx"
        print_rank_0(f"saving index file: {index_file}")
        self._data_file.close()
        with self.IndexClass.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


def build_chunk_mapping(index_dataset: MMapIndexedDataset, seq_len, min_length=1):
    """only record sample doc index and doc offset for one chunk,
    note that if samples have much short sentence will reduce the speed of dataloader reading and writing
    """
    item_lens = index_dataset.sizes
    # ensure doc can shift 1 token
    item_lens = np.asarray(item_lens) - 1
    num_samples = int(np.sum(np.ceil(item_lens / seq_len)))
    # doc_id, offset
    samples = np.zeros((num_samples, 2), dtype=np.int32)
    start = 0
    for item_idx, item_len in tqdm(enumerate(item_lens)):
        if item_len < min_length:
            continue

        n_chunks_doc = int(np.ceil(item_len / seq_len))  # number of chunks in one doc
        samples[start: start + n_chunks_doc, 0] = item_idx
        offsets = np.arange(0, n_chunks_doc * seq_len, seq_len, dtype=np.int32)
        # last chunk may be less than seq_len
        offsets[-1] = min(offsets[-1], item_len - 1 - seq_len)
        # if offset < 0
        offsets[-1] = max(offsets[-1], 0)
        samples[start: start + n_chunks_doc, 1] = offsets
        start += n_chunks_doc

    return samples[:start]


def build_sample_idx_mapping(index_dataset: MMapIndexedDataset, seq_len):
    """reference megatron"""
    doc_idx = index_dataset.doc_idx
    sizes = index_dataset.sizes
    tokens_per_epoch = np.sum(sizes, dtype=np.int64)
    num_samples = int((tokens_per_epoch - 1) / seq_len)
    samples = np.zeros((num_samples, 2), dtype=np.int32)
    sample_index = 0
    doc_idx_index = 0
    doc_offset = 0
    samples[2 * sample_index] = doc_idx_index
    samples[2 * sample_index + 1] = doc_offset
    while sample_index <= num_samples:
        remaining_seq_len = seq_len + 1
        while remaining_seq_len != 0:
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            remaining_seq_len -= doc_length
            if remaining_seq_len <= 0:
                # arrive file tail, but not add doc index
                doc_offset += remaining_seq_len + doc_length - 1
                remaining_seq_len = 0
            else:
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        samples[2 * sample_index] = doc_idx_index
        samples[2 * sample_index + 1] = doc_offset
        sample_index += 1

    return samples
