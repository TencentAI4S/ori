# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from dataclasses import dataclass
from typing import Optional, List, Any

import torch

from tgnn.utils.generic import ModelOutput


def top_k_filtering(logits: torch.Tensor, top_k: int):
    """Set the logits for none top-k values to -inf.

    Args:
        logits: [bs, vocab_size]
        top_k: keep top k probility values

    Returns:
        logits: [bs, vocab_size], mask none topk -inf
    """
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits.masked_fill_(logits < v[..., [-1]], float('-Inf'))

    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float):
    """Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        logits: tensor[bs, vocab_size], probability distribution tensor.
        top_p: probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.
    """
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Filteration based on the cumulative sum.
    filter = cumulative_probs > top_p
    filter[..., 1:] = filter[..., :-1].clone()
    # Make sure we at least have one token to select from.
    filter[..., 0] = 0
    # Fill in the filtered part
    filter = filter.scatter(1, sorted_indices, filter)
    logits.masked_fill_(filter, float('-Inf'))

    return logits


def sample(logits: torch.Tensor,
           top_k: int = 0,
           top_p: float = 0.0,
           temperature: float = 1.0):
    """ Sample and generate a token.
    Args:
        logits: tensor[bs, v], v is vocab size

    Return:
        output: tensor[bs, ], selected tokens
    """
    assert 0.0 <= top_p <= 1.0, 'p in the range[0, 1]'
    assert 0 <= top_k <= logits.size(1), 'top-k is larger than logit size.'
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    # Clone so we do not modify the inputs,
    logits = logits.clone()
    if temperature != 1.0:
        logits.div_(temperature)

    if top_k > 1:
        top_k_filtering(logits, top_k)
    elif top_p > 0.0:
        assert top_p <= 1.0, 'top-p should be in (0, 1].'
        top_p_filtering(logits, top_p)
    # After filtering, we need to recalculate the distribution.
    probs = logits.softmax(dim=-1)

    return torch.multinomial(probs, num_samples=1)


@dataclass
class GenerationOutput(ModelOutput):
    """
    Args:
        sequences: tensor[batch_size, sequence_length], The generated sequences. The second dimension (sequence_length)
            is either equal to `max_length` or shorter if all batches finished early due to the `eos_token_id`.
        scores: `tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed
    """
    sequences: torch.LongTensor = None
    scores: Optional[List[torch.FloatTensor]] = None


class GenerationMixin:

    def reset_cache(self):
        pass

    def update_cache(self, batch_ids):
        pass

    def step_forward(self, token_ids, pos_ids=None):
        return self.forward(token_ids, pos_ids=pos_ids)

    @torch.no_grad()
    def generate(self,
                 token_ids,
                 *,
                 max_new_tokens=None,
                 top_k: int = 0,
                 top_p: float = 0.,
                 temperature: float = 1.0,
                 output_score: bool = False,
                 stop_ids: Any = None):
        """
        Args:
            token_ids: [bs, seq_len]
        """
        only_one = token_ids.dim() == 1
        if only_one:
            token_ids = token_ids[None]

        assert token_ids.dim() == 2, f"only support batch input or one sample"
        bs, seq_len = token_ids.shape
        max_new_tokens = max_new_tokens or (self.max_len - seq_len)
        assert seq_len < self.max_len, f"input token is too long"
        device, dtype = token_ids.device, token_ids.dtype
        max_len = min(self.max_len, seq_len + max_new_tokens)
        output_ids = list(range(bs))
        outputs = [GenerationOutput(scores=[] if output_score else None) for _ in output_ids]
        input_pos = torch.arange(0, seq_len, device=device)
        for cur_pos in range(seq_len, max_len):
            bs = token_ids.shape[0]
            input_ids = token_ids.index_select(1, input_pos).view(bs, -1)
            logits = self.step_forward(input_ids, pos_ids=input_pos)[:, -1]
            next_ids = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
            input_pos = input_pos[-1:] + 1
            token_ids = torch.cat((token_ids, next_ids), dim=-1)  # [bs, seq_len]
            if output_score:
                probs = logits.softmax(dim=-1)
                for i, p in enumerate(probs):
                    p = p[next_ids[i].item()]
                    outputs[output_ids[i]].scores.append(p)

            remained_batch_ids = list(range(token_ids.size(0)))
            if stop_ids is not None:
                for i, tidx in enumerate(next_ids.view(-1).tolist()):
                    if tidx in stop_ids:
                        outputs[output_ids[i]].sequences = token_ids[i]
                        remained_batch_ids.remove(i)

                if len(remained_batch_ids) == 0:
                    break
                remained_batch_ids = torch.tensor(remained_batch_ids, dtype=torch.int64, device=device)
                token_ids = token_ids[remained_batch_ids]
                self.update_cache(remained_batch_ids)

            output_ids = [output_ids[idx] for idx in remained_batch_ids]
            if cur_pos == max_len - 1 and len(output_ids) > 0:
                for i, tensor in zip(output_ids, token_ids):
                    outputs[i].sequences = tensor

        self.reset_cache()
        outputs = outputs[0] if only_one else outputs
        return outputs
