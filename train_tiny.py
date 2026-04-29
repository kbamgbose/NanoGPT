"""
Trains a small GPT on input.txt (Shakespeare).
Goal: overfit quickly to verify the architecture works end-to-end.
Run: python train_tiny.py
"""
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken

from attention import CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer:    int = 6
    n_head:     int = 6
    n_embd:     int = 384


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
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
        assert T <= self.config.block_size
        pos     = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.transformer.wpe(pos)                                 # (T, n_embd)
        tok_emb = self.transformer.wte(idx)                                 # (B, T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x      = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                            # (B, T, vocab_size)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── data ──────────────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
tokens = torch.tensor(enc.encode(text), dtype=torch.long)
print(f"dataset: {len(tokens):,} tokens")

# ── device ────────────────────────────────────────────────────────────────────
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# ── model ─────────────────────────────────────────────────────────────────────
torch.manual_seed(1337)
config = GPTConfig()
model  = GPT(config).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── training ──────────────────────────────────────────────────────────────────
B, T      = 4, 256
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
max_steps = 200

for step in range(max_steps):
    ix = torch.randint(len(tokens) - T, (B,))
    x  = torch.stack([tokens[i:i+T]     for i in ix]).to(device)
    y  = torch.stack([tokens[i+1:i+T+1] for i in ix]).to(device)

    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0 or step == max_steps - 1:
        print(f"step {step:3d} | loss: {loss.item():.4f}")

# loss should be well below 4.0 by step 200 — confirms the model is learning
os.makedirs("checkpoints", exist_ok=True)
torch.save({'model': model.state_dict(), 'config': config}, 'checkpoints/tiny.pt')
print("saved to checkpoints/tiny.pt")
