# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from __future__ import annotations

import logging
import os
from typing import Optional, Union, List

import numpy as np
import sentencepiece as sp
import torch

from .base_tokenizer import AbstractTokenizer
from .build import TOKENIZER_REGISTRY
from ..config import configurable
from ..utils.io import cat_files


@TOKENIZER_REGISTRY.register("sentencepiece")
@TOKENIZER_REGISTRY.register()
class SentencePieceTokenizer(AbstractTokenizer):
    """Tokenizer for sentencepiece

    Args:
        model_path: path to sentencepiece model
    """

    @classmethod
    def from_config(cls, cfg):
        return {
            "model_path": cfg.tokenizer.path
        }

    @configurable
    def __init__(self,
                 model_path: str) -> None:
        super().__init__(os.path.basename(model_path).split(".")[0])
        self.model_path = model_path
        self.spm = sp.SentencePieceProcessor(model_file=model_path)
        self.unk_id = self.spm.unk_id()
        self.eos_id = self.spm.eos_id()
        self.bos_id = self.spm.bos_id()
        if self.bos_id < 0:
            neg_bos_id = self.bos_id
            self.bos_id = self.piece_to_id("<s>")
            logging.warning(f"change bos token id from {neg_bos_id} to {self.bos_id}")
            assert self.bos_id > 0

        self.pad_id = self.spm.pad_id()
        if self.pad_id < 0:
            self.pad_id = self.eos_id

        self._vocab = {id: self.spm.id_to_piece(id) for id in range(self.spm.get_piece_size())}

    @property
    def vocab(self):
        return self._vocab

    def id_to_piece(self, token_id):
        return self.spm.IdToPiece(token_id)

    def piece_to_id(self, piece):
        return self.spm.PieceToId(piece)

    @property
    def vocab_size(self) -> int:
        return self.spm.vocab_size()

    @property
    def unk(self):
        return self.unk_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def eod(self):
        # might conflict with <eos>
        return self.eos_id

    @property
    def bos(self):
        # might conflict with <eos>
        return self.bos_id

    @property
    def eos(self):
        # might conflict with <eos>
        return self.eos_id

    def encode(
            self,
            seq: str,
            bos: bool = False,
            eos: bool = False,
            max_length: int = -1,
            pad: bool = False,
            device: Optional[torch.device] = None,
            to_tensor=True
    ) -> torch.Tensor:
        tokens = self.spm.encode(seq)
        if bos:
            tokens = [self.bos_id] + tokens

        if eos:
            tokens = tokens + [self.eos_id]

        if max_length > 0:
            tokens = tokens[:max_length]

        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        if to_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

        return tokens

    def decode(self, tokens: Union[torch.Tensor, np.ndarray]) -> str:
        if isinstance(tokens, (torch.Tensor, np.ndarray)):
            tokens = tokens.tolist()

        return self.spm.decode(tokens)

    def model_to_vocab(self):
        with open(self.model_path[:-5] + "vocab", "w") as f:
            for token_id in range(self.spm.get_piece_size()):
                token = self.spm.id_to_piece(token_id)
                f.write(f"{token}\t{token_id}\n")

    @classmethod
    def add_new_vocabs(cls, model_file, pieces, scores=None, save_name=None):
        import sentencepiece.sentencepiece_model_pb2 as model_pb2
        model = model_pb2.ModelProto()
        model.ParseFromString(open(model_file, "rb").read())

        if scores is None:
            scores = [0, ] * len(pieces)

        for (piece, score) in zip(pieces, scores):
            token = model_pb2.ModelProto().SentencePiece()
            token.piece = piece
            token.score = score
            model.pieces.append(token)
        print(f"number of vocab: {len(model.pieces)}")
        if save_name is not None:
            with open(save_name, 'wb') as f:
                f.write(model.SerializeToString())
            cls(save_name).model_to_vocab()

    @classmethod
    def modif_vocabs(cls, model_file, pieces_dict, save_name=None):
        import sentencepiece.sentencepiece_model_pb2 as model_pb2
        model = model_pb2.ModelProto()
        model.ParseFromString(open(model_file, "rb").read())
        # pieces = []
        for token in model.pieces:
            if token.piece in pieces_dict:
                dst_piece = pieces_dict[token.piece]
                token.piece = dst_piece
            # pieces.append(token)

        # model.pieces = pieces
        print(f"number of vocab: {len(model.pieces)}")
        if save_name is not None:
            with open(save_name, 'wb') as f:
                f.write(model.SerializeToString())
            cls(save_name).model_to_vocab()

    @classmethod
    def get_model_pieces(cls, model_or_file):
        if isinstance(model_or_file, str):
            import sentencepiece.sentencepiece_model_pb2 as sp_model
            model = sp_model.ModelProto()
            model.ParseFromString(open(model_or_file, "rb").read())
        else:
            model = model_or_file

        peices = []
        scores = []
        for token in model.pieces:
            peice = token.piece
            score = token.score
            peices.append(peice)
            scores.append(score)

        return peices, scores

    @staticmethod
    def train(corpus_txt: Union[str, List[str]],
              model_prefix: str = None,
              vocab_size=20000,
              model_type="bpe",
              user_defined_symbols=None,
              character_coverage=0.995,
              num_threads=1
              ):
        if isinstance(corpus_txt, (tuple, list)):
            filename = f"{os.getcwd()}/corpus.txt"
            cat_files(corpus_txt, filename, end="")
            corpus_txt = filename

        if model_prefix is None:
            model_prefix = f"{os.path.split(corpus_txt)[0]}/{model_type}"

        if isinstance(user_defined_symbols, str):
            assert os.path.exists(user_defined_symbols), f"user defined symbol text not exist: {user_defined_symbols}"
            user_defined_symbols = open(user_defined_symbols, "r").read().split("\n")

        sp.SentencePieceTrainer.Train(input=corpus_txt,
                                      model_prefix=model_prefix,
                                      model_type=model_type,
                                      vocab_size=vocab_size,
                                      user_defined_symbols=user_defined_symbols,
                                      num_threads=num_threads,
                                      character_coverage=character_coverage)

    @classmethod
    def merge_models(cls, model_files, save_name):
        import sentencepiece.sentencepiece_model_pb2 as sp_model
        model = sp_model.ModelProto()
        main_model_file = model_files[0]
        model.ParseFromString(open(main_model_file, "rb").read())
        scores = []
        peices = set()
        for token in model.pieces:
            peice = token.piece
            score = token.score
            scores.append(score)
            peices.add(peice)

        print(f"score min: {np.min(scores)}, max: {np.max(scores)}")
        min_score = np.min(scores)
        # protein token have a higher priority
        for path in model_files[1:]:
            m = sp_model.ModelProto()
            m.ParseFromString(open(path, "rb").read())
            for token in m.pieces:
                peice = token.piece
                score = token.score
                if peice in peices:
                    continue

                token = sp_model.ModelProto().SentencePiece()
                token.piece = peice
                token.score = score + min_score
                scores.append(token.score)
                model.pieces.append(token)

        min_score = np.min(scores)
        print(f"after merge, min score: {min_score}")
        vocab_size = len(model.pieces)
        num_aligns = 64 - vocab_size % 64
        for i in range(num_aligns):
            token = sp_model.ModelProto().SentencePiece()
            token.piece = f"<R{i}>"
            assert token.piece not in model.pieces
            print(f"add special token: {token.piece}")
            token.score = i + min_score
            scores.append(token.score)
            model.pieces.append(token)

        print(f"number of vocab: {len(model.pieces)}")
        with open(save_name, 'wb') as f:
            f.write(model.SerializeToString())

        cls(save_name).model_to_vocab()
