# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import io
import json
import os
import sys
from io import BytesIO

import numpy as np
import torch
from tqdm import tqdm

from tgnn.utils.io import get_file_timestamp, set_file_timestamp
from .index_dataset import MMIndex, MMapIndexedDataset, MMapIndexedDatasetBuilder


class JsonlIndex(MMIndex):
    _HDR_MAGIC = b'JSONIDX\x00\x00'

    @classmethod
    def build_index(cls, filename):
        assert filename.endswith("jsonl"), f"file is not jsonl file: {filename}"

        def generator():
            i = 0
            while True:
                yield i
                i += 1

        sizes = []
        with open(filename, 'r', encoding="utf-8") as f:
            for _ in tqdm(generator(), desc="indexing"):
                line = f.readline()
                if not line:
                    break
                length = len(line.encode('utf-8'))
                sizes.append(length)

        idx_filename = f"{filename}.idx"
        with cls.writer(idx_filename) as index:
            index.write(sizes)

        bin_ts = get_file_timestamp(filename)
        set_file_timestamp(idx_filename, bin_ts)

    @classmethod
    def writer(cls, path, dtype=np.uint8):
        return super().writer(path, dtype)


class JsonlDataset(torch.utils.data.Dataset):

    def __init__(self, filename, sizes=None):
        if isinstance(filename, io.BytesIO):
            print("loadding json dataset from bytes io")
            self.data_file = filename
            if sizes is None:
                lines = io.TextIOWrapper(filename, encoding="utf-8")
                self.sizes = []
                for line in lines:
                    if not line:
                        break
                    length = len(line.encode('utf-8'))
                    self.sizes.append(length)
            else:
                self.sizes = sizes

            self.offsets = np.cumsum([0, ] + list(self.sizes[:-1]))
            self.ids = np.arange(len(self.sizes))
        else:
            self.filename = filename
            if not os.path.exists(self.index_file):
                JsonlIndex.build_index(self.filename)
            self.ids, self.offsets, self.sizes = self.read_index(self.index_file)
            self.data_file = None  # lazy loadding avoid copy file handle

        self.num_items = len(self.ids)

    def get_chunk_dataset(self, start, end=None):
        if self.data_file is None:
            self.data_file = self.read_data(self.filename)

        offset = self.offsets[start]
        self.data_file.seek(offset)
        sizes = self.sizes[start: end]
        chunk_size = np.sum(sizes)
        print(f"read bytes chunk size: {chunk_size}")
        chunk_size = int(chunk_size)
        offset += chunk_size
        buffer = BytesIO(self.data_file.read(chunk_size))

        return JsonlDataset(buffer, sizes=sizes)

    def shuffle_dataset(self, filename, seed=42):
        indices = np.arange(len(self))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        with open(filename, mode="xb") as f:
            for idx in tqdm(indices):
                bytes = self.read_bytes(idx)
                f.write(bytes)

        return JsonlDataset(filename)

    def __len__(self):
        return len(self.ids)

    def read_data(self, path):
        return open(path, mode='rb', buffering=0)

    def read_index(self, path, skip_warmup=True):
        assert JsonlIndex.is_index(path), f"{path} is not valid index file"
        index = JsonlIndex(path, skip_warmup=skip_warmup)
        return index.doc_idx, index.pointers, index.sizes

    @property
    def index_file(self):
        return f"{self.filename}.idx"

    @property
    def bin_file(self):
        return self.filename

    def get_json_info(self, index):
        offset = self.offsets[index]
        size = self.sizes[index]

        return offset, size

    def read_bytes(self, index):
        offset, size = self.get_json_info(index)
        if self.data_file is None:
            self.data_file = self.read_data(self.filename)

        self.data_file.seek(offset)
        bytes = self.data_file.read(size)

        return bytes

    def read_json(self, index):
        bytes = self.read_bytes(index)
        data = bytes.decode("utf-8")
        try:
            data = json.loads(data)
        except:
            print(data, file=sys.stderr)
            raise ValueError()
        return data

    def index_to_id(self, index):
        return self.ids[index]

    def __getitem__(self, index):
        data = self.read_json(index)

        return self.index_to_id(index), data


class IndexedJsonCachedDataset(JsonlDataset):

    def __init__(self, path):
        super().__init__(path)
        self.cache = self.cache_dataset()

    def cache_dataset(self):
        if not self.data_file:
            self.data_file = self.read_data(self.path)

        self.cache = []
        for i, lid in enumerate(self.ids):
            self.cache[i] = self.read_json(i)

        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

        return self.cache

    def __getitem__(self, index):
        data = self.cache[index]
        return self.index_to_id(index), data


class MMapIndexedJsonlDataset(MMapIndexedDataset):

    def read_index(self, path, skip_warmup=True):
        if os.path.isfile(path):
            assert JsonlIndex.is_index(path), f"{path} is not valid index file"
        else:
            print(f"not exist index file, start building index: {path}")
            JsonlIndex.build_index(self.bin_file)

        return JsonlIndex(path, skip_warmup=skip_warmup)

    def to_json(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_json(item) for item in data]
        else:
            try:
                data = json.loads(data.tobytes())
            except:
                print(data, file=sys.stderr)
                raise ValueError()
            return data

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return self.to_json(data)


class MMapIndexedJsonlDatasetBuilder(MMapIndexedDatasetBuilder):
    IndexClass = JsonlIndex
    IndexDataset = MMapIndexedJsonlDataset

    def __init__(self, out_file, dtype=np.uint8):
        super().__init__(out_file, dtype=dtype)

    def add_item(self, jsonline, encode="utf-8"):
        if isinstance(jsonline, dict):
            jsonline = json.dumps(jsonline) + "\n"

        if encode is not None:
            bytes = jsonline.encode("utf-8")
        else:
            bytes = jsonline

        super().add_item(bytes)

    def add_items(self, lines, verbose=True):
        desc = os.path.basename(self._filename)
        line_iter = tqdm(lines, desc=desc) if verbose else lines
        for line in line_iter:
            self.add_item(line)
            self.end_document()

    @classmethod
    def merge_files(cls, files, filename):
        for path in tqdm(files, "merge jsonl files"):
            if not os.path.exists(path + ".idx"):
                cls.IndexClass.build_index(path)
        super(MMapIndexedJsonlDatasetBuilder, cls).merge_files(files, filename)

    def merge_file(self, filename):
        index_file = filename + ".idx"
        if not os.path.isfile(index_file):
            JsonlIndex.build_index(index_file)
        super().merge_file(filename)
