"""
Invariant tests for transformer correctness.
Each test proves a specific guarantee that must hold for the model to be correct.
Run: python -m pytest tests/test_transformer.py -v
"""
import pytest
import torch
from dataclasses import dataclass


# ── minimal config for fast deterministic tests ───────────────────────────────

@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 256
    n_layer:    int = 2
    n_head:     int = 2
    n_embd:     int = 64


def get_model():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from model import GPT
    torch.manual_seed(0)
    return GPT(GPTConfig())


# ── test 1: causal masking — future tokens must not affect past logits ─────────

def test_causal_masking():
    """
    Proves the causal mask is correct.
    If token at position t+1 is changed, logits at positions 0..t must not change.
    If they do change, the model is attending to future tokens — it's cheating.
    """
    model = get_model()
    model.eval()

    B, T = 1, 16
    torch.manual_seed(42)
    x = torch.randint(0, 256, (B, T))

    with torch.no_grad():
        logits_original, _ = model(x)

    # corrupt every token after position 7
    x_corrupted = x.clone()
    x_corrupted[:, 8:] = torch.randint(0, 256, (B, T - 8))

    with torch.no_grad():
        logits_corrupted, _ = model(x_corrupted)

    # logits at positions 0..7 must be identical — future tokens had no effect
    assert torch.allclose(logits_original[:, :8, :], logits_corrupted[:, :8, :]), \
        "FAIL: future tokens affected past logits — causal mask is broken"


# ── test 2: attention output shape ────────────────────────────────────────────

def test_attention_output_shape():
    """
    Proves attention preserves the input shape (B, T, C) -> (B, T, C).
    If this fails, something is wrong with the head split/merge or output projection.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from attention import CausalSelfAttention  # noqa: E402

    config = GPTConfig()
    attn = CausalSelfAttention(config)
    attn.eval()

    B, T, C = 2, 16, config.n_embd
    x = torch.randn(B, T, C)

    with torch.no_grad():
        out = attn(x)

    assert out.shape == (B, T, C), \
        f"FAIL: expected shape {(B, T, C)}, got {out.shape}"


# ── test 3: forward pass output shape ────────────────────────────────────────

def test_forward_pass_shape():
    """
    Proves the full model forward pass produces logits of shape (B, T, vocab_size).
    Catches any dimension mismatch across embeddings, blocks, and the lm_head.
    """
    model = get_model()
    model.eval()

    B, T = 3, 16
    x = torch.randint(0, 256, (B, T))

    with torch.no_grad():
        logits, loss = model(x)

    assert logits.shape == (B, T, GPTConfig().vocab_size), \
        f"FAIL: expected logits shape {(B, T, GPTConfig().vocab_size)}, got {logits.shape}"
    assert loss is None, "FAIL: loss should be None when no targets are passed"


# ── test 4: loss is finite ────────────────────────────────────────────────────

def test_loss_is_finite():
    """
    Proves the forward pass produces a finite loss given valid inputs and targets.
    NaN or inf loss means something overflowed or a masking error produced 0/0.
    """
    model = get_model()
    model.eval()

    B, T = 2, 16
    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))

    with torch.no_grad():
        _, loss = model(x, y)

    assert loss is not None
    assert torch.isfinite(loss).all(), \
        f"FAIL: loss is not finite — got {loss.item()}"


# ── test 5: no NaNs in forward or backward ────────────────────────────────────

def test_no_nans_forward_and_backward():
    """
    Proves no NaN or inf appears in logits or gradients during a full training step.
    NaNs in the forward pass usually mean a bad mask or dtype overflow.
    NaNs in gradients mean the loss surface exploded — often from missing grad clipping
    or bad init, but can also point to a structural bug.
    """
    model = get_model()
    model.train()

    B, T = 2, 16
    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))

    logits, loss = model(x, y)

    # check forward
    assert not torch.isnan(logits).any(), "FAIL: NaN in logits (forward pass)"
    assert not torch.isinf(logits).any(), "FAIL: inf in logits (forward pass)"

    loss.backward()

    # check backward
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), \
                f"FAIL: NaN in gradient of {name}"
            assert not torch.isinf(param.grad).any(), \
                f"FAIL: inf in gradient of {name}"
