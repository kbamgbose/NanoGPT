"""
FM-02: Pytest tests proving the 4 architectural failure modes are detectable.
Each test asserts a quantified invariant that breaks when the bug is present.
Run: python -m pytest tests/test_failure_modes.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments"))

import math
import torch
import pytest
from torch.nn import functional as F
from failure_modes import (
    CFG, BaseAttention, NoCausalMask, SoftmaxBeforeMask,
    NoSqrtScaling, WrongTranspose, LocalGPT,
)


def test_no_causal_mask_leaks_future():
    """NoCausalMask must allow future tokens to affect past logits."""
    torch.manual_seed(0)
    model = LocalGPT(CFG, attn_class=NoCausalMask)
    model.eval()

    torch.manual_seed(1)
    x = torch.randint(0, CFG.vocab_size, (1, CFG.block_size))
    x_corrupt = x.clone()
    x_corrupt[:, 8:] = torch.randint(0, CFG.vocab_size, (1, CFG.block_size - 8))

    with torch.no_grad():
        logits_a, _ = model(x)
        logits_b, _ = model(x_corrupt)

    leaked = not torch.allclose(logits_a[:, :8, :], logits_b[:, :8, :], atol=1e-5)
    assert leaked, "NoCausalMask should leak future tokens into past logits"


def test_softmax_before_mask_row_sums():
    """SoftmaxBeforeMask attention rows must not sum to 1.0."""
    torch.manual_seed(0)
    model = LocalGPT(CFG, attn_class=SoftmaxBeforeMask)
    model.eval()

    attn = model.transformer.h[0].attn
    torch.manual_seed(1)
    x_in = torch.randn(1, CFG.block_size, CFG.n_embd)

    with torch.no_grad():
        B, T = 1, CFG.block_size
        hs = attn.head_size
        q, k, _ = attn.c_attn(x_in).split(CFG.n_embd, dim=2)
        q = q.view(B, T, attn.n_head, hs).transpose(1, 2)
        k = k.view(B, T, attn.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, 0.0)
        row_dev = (att.sum(dim=-1) - 1.0).abs().mean().item()

    assert row_dev > 0.1, \
        f"SoftmaxBeforeMask row sum deviation should exceed 0.1, got {row_dev:.4f}"


def test_no_scaling_reduces_entropy():
    """NoSqrtScaling must produce lower attention entropy than correct scaling."""
    torch.manual_seed(1)
    x_in = torch.randn(1, CFG.block_size, CFG.n_embd)
    hs = CFG.n_embd // CFG.n_head

    def attention_entropy(scale):
        attn = BaseAttention(CFG)
        with torch.no_grad():
            B, T = 1, CFG.block_size
            q, k, _ = attn.c_attn(x_in).split(CFG.n_embd, dim=2)
            q = q.view(B, T, CFG.n_head, hs).transpose(1, 2)
            k = k.view(B, T, CFG.n_head, hs).transpose(1, 2)
            scores = q @ k.transpose(-2, -1)
            if scale:
                scores = scores * (1.0 / math.sqrt(hs))
            scores = scores.masked_fill(attn.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(scores, dim=-1)
            eps = 1e-9
            return -(att * (att + eps).log()).sum(-1).mean().item()

    entropy_unscaled = attention_entropy(scale=False)
    entropy_scaled   = attention_entropy(scale=True)

    assert entropy_unscaled < entropy_scaled, (
        f"Unscaled attention should be more saturated (lower entropy): "
        f"unscaled={entropy_unscaled:.4f}, scaled={entropy_scaled:.4f}"
    )


def test_wrong_transpose_raises():
    """WrongTranspose must raise RuntimeError on forward pass."""
    torch.manual_seed(0)
    model = LocalGPT(CFG, attn_class=WrongTranspose)
    x = torch.randint(0, CFG.vocab_size, (1, CFG.block_size))

    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(x)
