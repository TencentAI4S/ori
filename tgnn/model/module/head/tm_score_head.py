# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from tgnn.model.layer import Linear


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def _calculate_expected_aligned_error(
        alignment_confidence_breaks: torch.Tensor,
        aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7

    Args:
        c_z: Input channel dimension
        num_bins: Number of bins
    """

    def __init__(self, c_z, num_bins=64):
        super().__init__()
        self.c_z = c_z
        self.num_bins = num_bins
        self.linear = Linear(self.c_z, self.num_bins, init="final")

    def forward(self, z):
        """
        Args:
            z: [*, seq_len, seq_len, c_z] pairwise embedding

        Returns:
            [*, seq_len, seq_len, no_bins] prediction
        """
        # [*, N, N, num_bins]
        logits = self.linear(z)
        return logits

    @classmethod
    def compute_tm_score(
            cls,
            logits: torch.Tensor,
            asym_id: Optional[torch.Tensor] = None,
            residue_weights: Optional[torch.Tensor] = None,
            max_bin: int = 31,
            num_bins: int = 64,
            eps: float = 1e-8
    ) -> torch.Tensor:
        """Computes pTM and ipTM from logits.

        Args；
            logits: [*, seq_len, seq_len, num_bins], pairwise prediction
            residue_weights: [*, seq_len] the per-residue weights to use for the expectation
            asym_id: [*, seq_len] the asymmetric unit ID - the chain ID. Only needed for ipTM calculation

        Returns:
            score: the predicted TM alignment or the predicted iTM score.
        """
        if residue_weights is None:
            residue_weights = logits.new_ones(logits.shape[-2])

        boundaries = torch.linspace(
            0, max_bin, steps=(num_bins - 1), device=logits.device
        )

        bin_centers = _calculate_bin_centers(boundaries)
        clipped_n = max(torch.sum(residue_weights), 19)

        d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

        probs = logits.softmax(dim=-1)

        tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
        predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

        n = residue_weights.shape[-1]
        pair_mask = residue_weights.new_ones((n, n), dtype=torch.int32)

        if asym_id is not None:
            if len(asym_id.shape) > 1:
                assert len(asym_id.shape) <= 2
                batch_size = asym_id.shape[0]
                pair_mask = residue_weights.new_ones((batch_size, n, n), dtype=torch.int32)
            pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(dtype=pair_mask.dtype)

        predicted_tm_term *= pair_mask

        pair_residue_weights = pair_mask * (
                residue_weights[..., None, :] * residue_weights[..., :, None]
        )
        denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
        normed_residue_mask = pair_residue_weights / denom
        per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

        weighted = per_alignment * residue_weights

        argmax = (weighted == torch.max(weighted)).nonzero()[0]
        return per_alignment[tuple(argmax)]

    @classmethod
    def compute_weighted_ptm_score(cls,
                                   logits: torch.Tensor,
                                   asym_id: Optional[torch.Tensor] = None,
                                   residue_weights: Optional[torch.Tensor] = None,
                                   max_bin: int = 31,
                                   num_bins: int = 64,
                                   iptm_weight: float = 0.8,
                                   ptm_weight: float = 0.2,
                                   eps: float = 1e-8
                                   ):
        scores = {}
        scores["ptm"] = cls.compute_tm_score(
            logits, residue_weights=residue_weights, max_bin=max_bin, num_bins=num_bins, eps=eps
        )

        if asym_id is not None:
            scores["iptm"] = cls.compute_tm_score(logits, asym_id=asym_id, residue_weights=residue_weights,
                                                  max_bin=max_bin, num_bins=num_bins, eps=eps)
            scores["weighted_ptm"] = iptm_weight * scores["ptm"] + ptm_weight * scores["iptm"]
        return scores

    @classmethod
    def compute_predicted_aligned_error(
            cls,
            logits: torch.Tensor,
            max_bin: int = 31,
            num_bins: int = 64
    ) -> Dict[str, torch.Tensor]:
        """Computes aligned confidence metrics from logits.

        Args:
              logits: [*, num_res, num_res, num_bins] the logits output from
                PredictedAlignedErrorHead.
              max_bin: Maximum bin value
              num_bins: Number of bins
        Returns:
              aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
                aligned error probabilities over bins for each residue pair.
              predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
                error for each pair of residues.
              max_predicted_aligned_error: [*] the maximum predicted error possible.
        """
        boundaries = torch.linspace(
            0, max_bin, steps=(num_bins - 1), device=logits.device
        )
        aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)

        (
            predicted_aligned_error,
            max_predicted_aligned_error,
        ) = _calculate_expected_aligned_error(
            alignment_confidence_breaks=boundaries,
            aligned_distance_error_probs=aligned_confidence_probs,
        )

        return {
            "aligned_confidence_probs": aligned_confidence_probs,
            "predicted_aligned_error": predicted_aligned_error,
            "max_predicted_aligned_error": max_predicted_aligned_error
        }
