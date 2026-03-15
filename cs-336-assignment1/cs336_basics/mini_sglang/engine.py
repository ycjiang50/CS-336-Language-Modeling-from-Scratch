"""
Model engine and sampling utilities.

ModelEngine wraps a TransformerLM for inference: it disables gradients,
runs forward passes, and exposes a clean interface for the scheduler.

The sampling helpers (temperature scaling, top-p, top-k, repetition
penalty) are implemented as composable pure functions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .core import SamplingParams


# ═══════════════════════════════════════════════════════════════════════
# Sampling helpers
# ═══════════════════════════════════════════════════════════════════════

def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        temperature = 1e-8
    return torch.softmax(logits / temperature, dim=-1)


def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: keep the smallest set with cum-prob >= p."""
    if p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum <= p
    mask[..., 0] = True  # always keep the most-probable token
    filtered = sorted_probs * mask.float()
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, filtered)
    return out


def top_k_filter(probs: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= probs.shape[-1]:
        return probs
    topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
    out = torch.zeros_like(probs)
    out.scatter_(-1, topk_idx, topk_vals)
    out = out / out.sum(dim=-1, keepdim=True)
    return out


def apply_repetition_penalty(
    logits: torch.Tensor, past_tokens: List[int], penalty: float
) -> torch.Tensor:
    if penalty == 1.0 or not past_tokens:
        return logits
    unique_ids = list(set(past_tokens))
    idx = torch.tensor(unique_ids, dtype=torch.long, device=logits.device)
    scores = logits[idx]
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits = logits.clone()
    logits[idx] = scores
    return logits


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    past_tokens: Optional[List[int]] = None,
) -> int:
    """Apply the full sampling pipeline and return a single token id."""
    logits = logits.float()
    if params.repetition_penalty != 1.0 and past_tokens:
        logits = apply_repetition_penalty(logits, past_tokens, params.repetition_penalty)

    if params.temperature == 0:
        return logits.argmax(dim=-1).item()

    probs = softmax_with_temperature(logits, params.temperature)

    if params.top_k > 0:
        probs = top_k_filter(probs, params.top_k)
    if params.top_p < 1.0:
        probs = top_p_filter(probs, params.top_p)

    return torch.multinomial(probs, num_samples=1).item()


# ═══════════════════════════════════════════════════════════════════════
# Model engine
# ═══════════════════════════════════════════════════════════════════════

class ModelEngine:
    """
    Thin wrapper around a TransformerLM that manages eval mode and
    ``torch.no_grad``.  Returns only the *last-position* logits, which
    is all the scheduler needs for autoregressive sampling.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Parameters
        ----------
        input_ids       : (1, seq_len)  token ids to process
        past_key_values : model-format KV cache or ``None``

        Returns
        -------
        logits          : (1, vocab_size) for the *last* position only
        new_kv          : full (concatenated) KV cache for every layer
        """
        logits, new_kv = self.model(
            input_ids.to(self.device),
            past_key_values=past_key_values,
            use_cache=True,
        )
        return logits[:, -1, :], new_kv
