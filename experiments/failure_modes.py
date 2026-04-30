"""
FM-01: Controlled failure mode experiments.
Six isolated bug patches on a tiny CPU model. No production files modified.

Run: python experiments/failure_modes.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# ── tiny config — fast on CPU ─────────────────────────────────────────────────

@dataclass
class Config:
    block_size: int = 32
    vocab_size: int = 256
    n_layer:    int = 2
    n_head:     int = 2
    n_embd:     int = 64


CFG = Config()


# ── helpers ───────────────────────────────────────────────────────────────────

def section(label):
    print()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)


def train_n_steps(model, steps, lr=3e-4, clip=1.0):
    """
    Returns list of {step, loss, grad_norm, nan_occurred}.
    Catches all exceptions. clip=None disables gradient clipping.
    Seeded for reproducibility.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    results = []
    torch.manual_seed(42)

    for step in range(steps):
        x = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
        y = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))

        try:
            logits, loss = model(x, y)

            if not torch.isfinite(loss):
                results.append({"step": step, "loss": float("nan"),
                                 "grad_norm": float("nan"), "nan_occurred": True})
                break

            optimizer.zero_grad()
            loss.backward()

            if clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip).item()
            else:
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5

            optimizer.step()
            results.append({"step": step, "loss": loss.item(),
                             "grad_norm": grad_norm, "nan_occurred": False})

        except Exception as e:
            results.append({"step": step, "loss": float("nan"),
                             "grad_norm": float("nan"), "nan_occurred": True,
                             "error": str(e)})
            break

    return results


# ── base attention (local copy — production attention.py is never imported) ───

class BaseAttention(nn.Module):
    """Correct causal self-attention. Baseline all bug subclasses diverge from."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn   = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj   = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head    = config.n_head
        self.n_embd    = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        hs = self.head_size
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# ── local MLP, Block, GPT (accept attn_class injection) ──────────────────────

class LocalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class LocalBlock(nn.Module):
    def __init__(self, config, attn_class):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = attn_class(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = LocalMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LocalGPT(nn.Module):
    def __init__(self, config, attn_class=BaseAttention):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([LocalBlock(config, attn_class)
                                  for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                if hasattr(module, 'NANOGPT_SCALE_INIT'):
                    std *= (2 * self.config.n_layer) ** -0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos    = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── bug subclasses ────────────────────────────────────────────────────────────

class NoCausalMask(BaseAttention):
    """Bug 1: masked_fill removed — all positions attend to all positions."""
    def forward(self, x):
        B, T, C = x.size()
        hs = self.head_size
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        # BUG: mask line removed
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SoftmaxBeforeMask(BaseAttention):
    """Bug 2: softmax before masking — row sums no longer equal 1."""
    def forward(self, x):
        B, T, C = x.size()
        hs = self.head_size
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        # BUG: softmax first, then zero out future (not -inf, not renormalized)
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0.0)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class NoSqrtScaling(BaseAttention):
    """Bug 3: scaling removed — dot products grow large, softmax saturates."""
    def forward(self, x):
        B, T, C = x.size()
        hs = self.head_size
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # BUG: no * (1.0 / math.sqrt(hs))
        att = q @ k.transpose(-2, -1)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class WrongTranspose(BaseAttention):
    """Bug 4: transpose before view — corrupts head/sequence dimension layout."""
    def forward(self, x):
        B, T, C = x.size()
        hs = self.head_size
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # BUG: transpose(1,2) before view — tensor is non-contiguous, view raises
        q = q.transpose(1, 2).view(B, T, self.n_head, hs)
        k = k.transpose(1, 2).view(B, T, self.n_head, hs)
        v = v.transpose(1, 2).view(B, T, self.n_head, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(
            self.bias[:, :, :T, :T].unsqueeze(-1) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).contiguous().view(B, T, C)
        return self.c_proj(y)


# ── experiments ───────────────────────────────────────────────────────────────

def exp1_no_causal_mask():
    section("[BUG 1] No causal mask")
    torch.manual_seed(0)
    model = LocalGPT(CFG, attn_class=NoCausalMask)
    results = train_n_steps(model, steps=50, lr=3e-4, clip=1.0)

    print(f"  final loss:  {results[-1]['loss']:.4f}")

    # causal leak check: corrupt t+1 onward, compare logits at 0..7
    model.eval()
    torch.manual_seed(1)
    x = torch.randint(0, CFG.vocab_size, (1, CFG.block_size))
    x_corrupt = x.clone()
    x_corrupt[:, 8:] = torch.randint(0, CFG.vocab_size, (1, CFG.block_size - 8))
    with torch.no_grad():
        logits_a, _ = model(x)
        logits_b, _ = model(x_corrupt)
    max_diff = (logits_a[:, :8, :] - logits_b[:, :8, :]).abs().max().item()
    leaked = not torch.allclose(logits_a[:, :8, :], logits_b[:, :8, :], atol=1e-5)
    print(f"  leaked:      {leaked}")
    print(f"  max logit diff at past positions: {max_diff:.6f}")
    print(f"  (correct model: leaked=False, max_diff≈0.000000)")


def exp2_softmax_before_mask():
    section("[BUG 2] Softmax before masking")
    torch.manual_seed(0)
    model = LocalGPT(CFG, attn_class=SoftmaxBeforeMask)
    results = train_n_steps(model, steps=50, lr=3e-4, clip=1.0)

    print(f"  final loss:  {results[-1]['loss']:.4f}")

    # row sum deviation
    model.eval()
    attn = model.transformer.h[0].attn
    torch.manual_seed(1)
    x_in = torch.randn(1, CFG.block_size, CFG.n_embd)
    with torch.no_grad():
        B, T, C = x_in.size()
        hs = attn.head_size
        q, k, _ = attn.c_attn(x_in).split(CFG.n_embd, dim=2)
        q = q.view(B, T, attn.n_head, hs).transpose(1, 2)
        k = k.view(B, T, attn.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, 0.0)
        row_dev = (att.sum(dim=-1) - 1.0).abs().mean().item()
    print(f"  attention row sum deviation from 1.0: {row_dev:.4f}")
    print(f"  (correct model: deviation≈0.0000)")


def exp3_no_sqrt_scaling():
    section("[BUG 3] No sqrt(d_k) scaling")
    torch.manual_seed(0)
    model = LocalGPT(CFG, attn_class=NoSqrtScaling)
    results = train_n_steps(model, steps=50, lr=3e-4, clip=1.0)

    print(f"  final loss:  {results[-1]['loss']:.4f}")

    # score std: unscaled vs scaled (same weights, same input)
    torch.manual_seed(1)
    x_in  = torch.randn(1, CFG.block_size, CFG.n_embd)
    attn  = BaseAttention(CFG)
    hs    = CFG.n_embd // CFG.n_head
    with torch.no_grad():
        B, T = 1, CFG.block_size
        q, k, _ = attn.c_attn(x_in).split(CFG.n_embd, dim=2)
        q = q.view(B, T, CFG.n_head, hs).transpose(1, 2)
        k = k.view(B, T, CFG.n_head, hs).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        print(f"  pre-softmax score std (unscaled): {scores.std().item():.4f}")
        print(f"  pre-softmax score std (scaled):   {(scores / math.sqrt(hs)).std().item():.4f}")

        # attention entropy for no-scaling model
        att = scores.masked_fill(attn.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        eps     = 1e-9
        entropy = -(att * (att + eps).log()).sum(-1).mean().item()
    print(f"  attention entropy (unscaled): {entropy:.4f}")
    print(f"  (lower entropy = more saturated = vanishing gradients)")


def exp4_wrong_transpose():
    section("[BUG 4] Wrong view/transpose order")
    torch.manual_seed(0)
    model = LocalGPT(CFG, attn_class=WrongTranspose)
    torch.manual_seed(1)
    x = torch.randint(0, CFG.vocab_size, (1, CFG.block_size))
    try:
        with torch.no_grad():
            logits, _ = model(x)
        expected = (1, CFG.block_size, CFG.vocab_size)
        shape_ok  = tuple(logits.shape) == expected
        print(f"  No RuntimeError raised")
        print(f"  output shape: {tuple(logits.shape)}, expected: {expected}")
        print(f"  shape correct: {shape_ok}")
        if shape_ok:
            correct = LocalGPT(CFG, attn_class=BaseAttention)
            correct.load_state_dict(model.state_dict())
            with torch.no_grad():
                logits_correct, _ = correct(x)
            max_diff = (logits - logits_correct).abs().max().item()
            print(f"  max diff vs correct model: {max_diff:.4f}")
            print(f"  silent corruption detected: {max_diff > 0.01}")
    except RuntimeError as e:
        print(f"  RuntimeError raised (expected):")
        print(f"  {e}")


def exp5_high_lr():
    section("[BUG 5] High learning rate  (lr=1.0 vs normal 3e-4)")
    torch.manual_seed(0)
    model   = LocalGPT(CFG, attn_class=BaseAttention)
    results = train_n_steps(model, steps=10, lr=1.0, clip=1.0)

    nan_occurred = any(r['nan_occurred'] for r in results)
    print(f"  {'step':>4}  {'loss':>10}  {'grad_norm':>10}")
    print(f"  {'-' * 30}")
    for r in results:
        l = f"{r['loss']:>10.4f}" if not math.isnan(r['loss']) else "       NaN"
        g = f"{r['grad_norm']:>10.4f}" if not math.isnan(r['grad_norm']) else "       NaN"
        print(f"  {r['step']:>4}  {l}  {g}")
    print(f"  nan_occurred: {nan_occurred}")


def exp6_no_grad_clipping():
    section("[BUG 6] No gradient clipping")

    torch.manual_seed(0)
    model_u = LocalGPT(CFG, attn_class=BaseAttention)
    res_u   = train_n_steps(model_u, steps=50, lr=3e-4, clip=None)

    torch.manual_seed(0)
    model_c = LocalGPT(CFG, attn_class=BaseAttention)
    res_c   = train_n_steps(model_c, steps=50, lr=3e-4, clip=1.0)

    valid_u = [r for r in res_u if not math.isnan(r['grad_norm'])]
    valid_c = [r for r in res_c if not math.isnan(r['grad_norm'])]

    max_u = max(r['grad_norm'] for r in valid_u)
    max_c = max(r['grad_norm'] for r in valid_c)
    ratio = max_u / max_c if max_c > 0 else float('inf')

    print(f"  max grad norm (unclipped): {max_u:.4f}")
    print(f"  max grad norm (clipped):   {max_c:.4f}")
    print(f"  ratio (unclipped/clipped): {ratio:.2f}x")
    print(f"  final loss (unclipped):    {res_u[-1]['loss']:.4f}")
    print(f"  final loss (clipped):      {res_c[-1]['loss']:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FM-01: Failure Mode Experiments")
    print("Config: block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64")
    print("Device: CPU")

    exp1_no_causal_mask()
    exp2_softmax_before_mask()
    exp3_no_sqrt_scaling()
    exp4_wrong_transpose()
    exp5_high_lr()
    exp6_no_grad_clipping()

    print()
    print("=" * 60)
    print("  All experiments complete.")
    print("=" * 60)
