# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import itertools
from typing import Sequence, Tuple, List, Optional

import torch

from tgnn.config import configurable
from .base_tokenizer import AbstractTokenizer
from .build import TOKENIZER_REGISTRY

proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R',
             'T', 'I', 'D', 'P', 'K', 'Q', 'N',
             'F', 'Y', 'M', 'H', 'W', 'C', 'X',
             'B', 'U', 'Z', 'O', '.', '-']
}


@TOKENIZER_REGISTRY.register()
class Alphabet(AbstractTokenizer):

    @classmethod
    def model_name_to_tokens(cls, name):
        if name.startswith("esm2"):
            name = "ESM-1b"

        if name in ("ESM-1", "protein_bert_base"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
        elif name in ("MSA Transformer", "msa_transformer"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = False
        elif "invariant_gvp" in name.lower():
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<cath>", "<af2>")
            prepend_bos = True
            append_eos = False
        else:
            raise ValueError("Unknown architecture selected")

        return standard_toks, prepend_toks, append_toks, prepend_bos, append_eos

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        standard_toks, prepend_toks, append_toks, prepend_bos, append_eos = cls.model_name_to_tokens(name)
        return cls(standard_toks,
                   prepend_toks,
                   append_toks,
                   prepend_bos=prepend_bos,
                   append_eos=append_eos)

    @classmethod
    def from_config(cls, cfg):
        name = cfg.tokenizer.path
        standard_toks, prepend_toks, append_toks, _, _ = cls.model_name_to_tokens(name)
        prepen_bos = cfg.tokenizer.prepend_bos
        append_eos = cfg.tokenizer.append_eos
        return {"standard_toks": standard_toks,
                "prepend_toks": prepend_toks,
                "append_toks": append_toks,
                "prepend_bos": prepen_bos,
                "append_eos": append_eos}

    @configurable
    def __init__(
            self,
            standard_toks: Sequence[str],
            prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
            append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
            prepend_bos: bool = False,
            append_eos: bool = False,
            name=None,
    ):
        super().__init__(name or self.__class__.__name__)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i + 1}>")
        self.all_toks.extend(self.append_toks)
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
        self.unk_id = self.tok_to_idx["<unk>"]
        self.pad_id = self.get_idx("<pad>")
        self.cls_id = self.get_idx("<cls>")
        self.mask_id = self.get_idx("<mask>")
        self.bos_id = self.get_idx("<eos>")
        self.eos_id = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks

    @property
    def unk(self):
        return self.unk_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def bos(self):
        # might conflict with <eos>
        return self.bos_id

    @property
    def eos(self):
        # might conflict with <eos>
        return self.eos_id

    @property
    def vocab(self):
        return self.tok_to_idx

    @property
    def vocab_size(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_id)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self,
               text: str,
               bos: bool = None,
               eos: bool = None,
               max_length: int = None,
               pad: bool = False,
               device: Optional[torch.device] = None,
               to_tensor=True):
        bos = self.prepend_bos if bos is None else bos
        eos = self.append_eos if eos is None else eos
        tokens = [self.tok_to_idx[tok] for tok in self.tokenize(text)]

        # append cls id for bert model
        if bos:
            tokens = [self.cls_id] + tokens

        if eos:
            tokens = tokens + [self.eos_id]

        if max_length is not None:
            tokens = tokens[:max_length]

        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        if to_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

        return tokens

    def batch_encode(self, texts,
                     bos: bool = False,
                     eos: bool = False,
                     max_length: int = None,
                     pad: bool = False,
                     device: Optional[torch.device] = None
                     ):
        token_id_list = [self.encode(text,
                                     bos=bos,
                                     eos=eos,
                                     max_length=max_length,
                                     pad=pad,
                                     device=device,
                                     to_tensor=True) for text in texts]
        return torch.stack(token_id_list, dim=0)
