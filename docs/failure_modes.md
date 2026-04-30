# Transformer Failure Modes

Six failure modes isolated in `experiments/failure_modes.py`.
Config: `block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=64`, CPU, seed=42.

---

## 1. No Causal Mask

**Symptom:** Future tokens affect past logits. Position 0's output changes when positions 8–31 are replaced with random tokens.

**Measurement:**
```
leaked:    True
max logit diff at past positions: 0.052683
correct model: leaked=False, max_diff≈0.000000
```

**Root cause:** The `masked_fill(..., float('-inf'))` line is removed. Every position attends to every other position, including future ones. The model can "cheat" during training and will fail completely at inference where future tokens don't exist yet.

**Detection:** Run two forward passes on the same input, corrupt tokens at positions t+1 onward, assert logits at positions 0..t are identical. Any diff > 1e-5 is a leak.

**Fix:**
```python
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
```
Must happen before softmax. The bias buffer is a lower-triangular ones matrix registered in `__init__`.

---

## 2. Softmax Before Masking

**Symptom:** Attention rows do not sum to 1.0. The probability distribution over attended positions is broken.

**Measurement:**
```
attention row sum deviation from 1.0: 0.4839
correct model: deviation≈0.0000
```

**Root cause:** Softmax is applied to raw scores first, then future positions are zeroed out. The result is no longer a valid probability distribution — rows that include future tokens have mass removed without renormalization. The model still runs and produces loss, making this a silent correctness bug.

**Detection:** Inspect attention weights post-mask. `att.sum(dim=-1)` should be all 1.0. Mean absolute deviation > 0.1 indicates the bug.

**Fix:** Always mask first with `-inf`, then softmax:
```python
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)   # rows now sum to 1.0 by construction
```
`exp(-inf) = 0`, so masked positions contribute zero to the softmax denominator.

---

## 3. No sqrt(d_k) Scaling

**Symptom:** Pre-softmax scores have high variance, pushing softmax into saturation. Attention entropy collapses — the model attends to one token instead of distributing attention.

**Measurement:**
```
pre-softmax score std (unscaled): 2.1118
pre-softmax score std (scaled):   0.3733
attention entropy (unscaled):     1.4202
```

**Root cause:** The dot product `Q @ K^T` has expected variance proportional to `d_k` (head dimension). Without the `1/sqrt(d_k)` factor, scores grow large as head size increases. Softmax of large values saturates — one logit dominates, all others vanish. This produces near-zero gradients for most of the attention weights.

**Why `sqrt(d_k)`:** If Q and K have unit-variance entries, each dot product is a sum of `d_k` terms, giving variance `d_k`. Dividing by `sqrt(d_k)` restores unit variance. At `d_k=32` (n_embd=64, n_head=2): std drops from 2.11 to 0.37 — a 5.7x reduction.

**Detection:** Measure score std before softmax. Values > 1.5 at normal init suggest missing scaling. Entropy < 1.5 on a 32-token sequence suggests saturation.

**Fix:**
```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
```

---

## 4. Wrong view/transpose Order

**Symptom:** `RuntimeError` on the first forward pass. The model crashes immediately.

**Measurement:**
```
RuntimeError raised (expected):
The size of tensor a (32) must match the size of tensor b (2) at non-singleton dimension 3
```

**Root cause:** `q.transpose(1, 2).view(B, T, n_head, hs)` instead of `q.view(B, T, n_head, hs).transpose(1, 2)`. After `transpose`, the tensor is non-contiguous in memory. PyTorch's `.view()` requires contiguous memory and raises `RuntimeError`. The correct order is view first (reshape in-place), then transpose (reorder dimensions).

**Why it matters:** `view` and `reshape` are not equivalent. `view` is a zero-copy operation that requires the tensor's memory to be laid out contiguously. `transpose` creates a non-contiguous view. Calling `.contiguous()` before `.view()` would work but is a code smell — the fix is to get the order right.

**Detection:** Any `RuntimeError` mentioning dimension mismatch during a forward pass through the attention block.

**Fix:**
```python
q = q.view(B, T, self.n_head, hs).transpose(1, 2)   # view first, then transpose
```

---

## 5. High Learning Rate

**Symptom:** Loss diverges within the first few steps. Gradient norms spike immediately.

**Measurement (lr=1.0 vs normal 3e-4, 10 steps):**
```
step        loss   grad_norm
------------------------------
   0      5.5514      1.7207
   1     42.8087     23.2321
   2     73.8712     22.3213
   3     91.5250     36.4169
   4     90.0542     17.3873
   5     98.3008     35.3817
   6    120.3836     37.3173
   7    127.5184     36.2885
   8    125.7747     44.2005
   9    173.7684     37.6538
nan_occurred: False
```
Loss increases 31x in one step (5.55 → 42.8). Grad norm spikes from 1.72 to 23.2 immediately.

**Root cause:** Each parameter update is `θ ← θ - lr * grad`. At lr=1.0, a gradient of magnitude 1.72 moves weights by 1.72 — far outside the regime where the loss landscape is approximately quadratic. The model overshoots the minimum and lands in a worse region with higher gradients, compounding each step.

**Detection:** Monitor loss step-over-step. A loss increase > 2x in a single step is a signal. Gradient norm > 10 at initialization (before the model has learned anything) is a reliable indicator.

**Fix:** Use lr ≤ 3e-4 for AdamW on GPT-2 scale. Combine with a warmup schedule — start at 0, ramp linearly over ~700 steps, then cosine decay. This avoids large updates before the optimizer's momentum estimates have stabilized.

---

## 6. No Gradient Clipping

**Symptom:** On this small model with normal-magnitude gradients, clipping has no effect. The risk becomes visible at scale or under adversarial inputs.

**Measurement (50 steps, lr=3e-4):**
```
max grad norm (unclipped): 1.7207
max grad norm (clipped):   1.7207
ratio (unclipped/clipped): 1.00x
final loss (unclipped):    5.5765
final loss (clipped):      5.5777
```

**Root cause:** `clip_grad_norm_` caps the global gradient norm at `max_norm=1.0`. When the gradient norm is already below 1.0, it's a no-op. The danger emerges when rare batches or long training runs produce large gradient spikes — without clipping, a single bad batch can corrupt the entire parameter state.

**When it matters:** Large models (grad norms spike during certain sequence patterns), long training runs (grad norm instability grows with depth and duration), mixed-precision training (bfloat16 accumulation can amplify gradient variance), and any model without layer normalization.

**Detection:** Log `grad_norm` every step. If the max exceeds 5–10x the median, clipping is doing real work. On GPT-2 at full scale (768 dim, 12 layers), unclipped norms regularly reach 3–8x the clipped value during the first 1000 steps.

**Fix:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Call after `loss.backward()`, before `optimizer.step()`.

---

## Summary

| Bug | Silent? | Detectable at init? | Production risk |
|-----|---------|---------------------|-----------------|
| No causal mask | No — leaks on inspection | Yes | Critical — model cheats during training, fails at inference |
| Softmax before mask | **Yes** — loss still decreases | Yes | High — broken probability distribution, wrong gradients |
| No sqrt scaling | **Yes** — training still runs | Yes | High — vanishing gradients in attention, slow/failed convergence |
| Wrong transpose | No — immediate crash | Yes | Low — caught immediately |
| High learning rate | No — loss diverges visibly | Yes | Medium — obvious but easy to miss with poor monitoring |
| No gradient clipping | **Yes** — invisible on small models | No | Medium — silent instability at scale |

The three silent bugs (2, 3, 6) are the most dangerous: training appears to work, loss decreases, but the model is learning incorrectly or is fragile to scale.
