# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import json
import os
import re
import sys

import numpy as np
from Bio import SeqIO, Seq
from tqdm import tqdm

from tgnn.utils.io import get_file_timestamp, set_file_timestamp
from .index_dataset import MMIndex, MMapIndexedDatasetBuilder, MMapIndexedDataset


class FastaIndex(MMIndex):
    _HDR_MAGIC = b'FASTAIDX\x00\x00'

    @classmethod
    def build_index(cls, filename):
        assert filename.endswith(("fasta", "fa", "fna")), f"file is not fasta file: {filename}"

        def generator():
            i = 0
            while True:
                yield i
                i += 1

        sizes = []
        with open(filename, 'r', encoding="utf-8") as f:
            fasta_lines = []
            for _ in tqdm(generator(), desc="indexing"):
                line = f.readline()
                if not line:
                    if fasta_lines:
                        length = len("".join(fasta_lines).encode('utf-8'))
                        sizes.append(length)
                    break

                if line.startswith(">") and fasta_lines:
                    length = len("".join(fasta_lines).encode('utf-8'))
                    sizes.append(length)
                    fasta_lines = []
                fasta_lines.append(line)

        idx_filename = f"{filename}.idx"
        with cls.writer(f"{filename}.idx") as index:
            index.write(sizes)

        bin_ts = get_file_timestamp(filename)
        set_file_timestamp(idx_filename, bin_ts)

    @classmethod
    def writer(cls, path, dtype=np.uint8):
        return super().writer(path, dtype)


class MMapIndexedFastaDataset(MMapIndexedDataset):

    def read_index(self, path, skip_warmup=True):
        if os.path.isfile(path):
            assert FastaIndex.is_index(path), f"{path} is not valid index file"
        else:
            print(f"not exist index file, start building index: {path}")
            FastaIndex.build_index(self.bin_file)

        return FastaIndex(path, skip_warmup=skip_warmup)

    def to_fasta(self, data):
        if isinstance(data, (tuple, list)):
            return [self.to_fasta(item) for item in data]

        fasta_line = data.tobytes().decode("utf-8")
        title, *lines = fasta_line.split("\n")
        title = title[1:].rstrip()
        seq = "".join(lines)
        seq_id, *description = title.split(None, 2)
        if description:
            description = description[0]
            jsonline = re.findall(r"{(.+?)}", description)
            try:
                description = json.loads("{" + jsonline[0] + "}")
            except:
                description = {"description": description}
        else:
            description = {}

        meta = {"id": seq_id, **description}
        return seq, meta

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return self.to_fasta(data)


class MMapIndexedFastaDatasetBuilder(MMapIndexedDatasetBuilder):
    IndexClass = FastaIndex

    def __init__(self, out_file):
        super().__init__(out_file, np.uint8)

    def add_item(self,
                 seq,
                 id="<unknown id>",
                 description="<unknown description>"):
        if isinstance(description, dict):
            description = json.dumps(description)

        try:
            record = SeqIO.SeqRecord(Seq.Seq(seq),
                                     id=id,
                                     description=description)
        except:
            print(seq, file=sys.stderr, flush=True)
            raise f"can not write record, seq: {seq}, id: {id}, description: {description}"

        fasta_line = record.format("fasta")
        bytes = fasta_line.encode("utf-8")
        super().add_item(bytes)

    def add_record(self, record: SeqIO.SeqRecord):
        fasta_line = record.format("fasta")
        bytes = fasta_line.encode("utf-8")
        super().add_item(bytes)

    def merge_file(self, filename):
        index_file = filename + ".idx"
        if not os.path.isfile(index_file):
            FastaIndex.build_index(index_file)
        super().merge_file(filename)
