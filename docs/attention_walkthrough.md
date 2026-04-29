# Attention Walkthrough

## 1. What attention does in one sentence

For each token in a sequence, attention computes a weighted average of all other tokens' values — where the weights come from how relevant each token is to the current one.

---

## 2. Step-by-step derivation

### Step 1: Project inputs to Q, K, V

Given input `X` of shape `(B, T, C)`:

```
Q = X @ W_q    # what am I looking for?
K = X @ W_k    # what do I contain?
V = X @ W_v    # what do I return if selected?
```

Each is shape `(B, T, C)`. In practice we fuse this into one matmul:

```
QKV = X @ W_qkv    # (B, T, 3C)
Q, K, V = split(QKV, dim=-1)
```

### Step 2: Split into heads

We split `C` into `n_head` chunks of size `head_size = C / n_head`. Each head runs attention independently on its own slice, learning different relationships.

```
Q -> (B, n_head, T, head_size)
K -> (B, n_head, T, head_size)
V -> (B, n_head, T, head_size)
```

### Step 3: Compute attention scores

```
scores = Q @ K^T    # (B, n_head, T, T)
```

Entry `scores[b, h, i, j]` = how much token `i` attends to token `j` in head `h`.

### Step 4: Scale

```
scores = scores / sqrt(head_size)
```

Without this, dot products grow large as `head_size` increases (variance of the dot product is proportional to dimension). Large logits push softmax toward saturation — outputs near 0 or 1, gradients vanish. Dividing by `sqrt(d_k)` keeps variance ~1 regardless of head size. This is from the original "Attention is All You Need" paper.

### Step 5: Apply causal mask

```
scores = masked_fill(scores, mask == 0, -inf)
```

The mask is lower-triangular: position `i` can only attend to positions `0..i`. Future positions become `-inf` so softmax gives them exactly zero weight.

**Why masking happens before softmax:**
Softmax normalizes over all keys. If you softmax first, future tokens receive nonzero probability. Zeroing them out afterward breaks the normalization — probabilities no longer sum to 1. Mask first, softmax second: `-inf → exp(-inf) = 0`, and the remaining weights renormalize correctly.

### Step 6: Softmax

```
att = softmax(scores, dim=-1)    # (B, n_head, T, T)
```

Each row is now a probability distribution over past tokens.

### Step 7: Weighted sum of values

```
out = att @ V    # (B, n_head, T, head_size)
```

Each output token is a mixture of value vectors, weighted by how relevant each past token was.

### Step 8: Merge heads and project

```
out -> (B, T, C)          # concatenate heads
out = out @ W_o           # output projection
```

---

## 3. Where O(n²) shows up

### ASCII diagram

```
         Keys (T)
         ┌─────────────────────┐
    Q  t0│  s00  s01  s02 ...  │
    u  t1│  s10  s11  s12 ...  │
    e  t2│  s20  s21  s22 ...  │
    r  ...│  ...               │
    i  tT│  sT0  sT1  sT2 ...  │
    e     └─────────────────────┘
    s          (B, n_head, T, T)
```

Every query attends to every key → `T × T` entries per head.

### Compute: O(n²)

`Q @ K^T` is a `(T, head_size) @ (head_size, T)` matmul — cost is `O(T² * head_size)`. Double the sequence length → 4x the compute.

### Memory: O(n²)

The attention matrix `(B, n_head, T, T)` must be materialized in full. For `T=1024`, `n_head=12`, `B=16` in bfloat16:

```
16 * 12 * 1024 * 1024 * 2 bytes = ~402 MB
```

At `T=4096` that becomes ~6.4 GB — just for the attention matrix, before any activations.

### Why it's memory-bound in practice

On modern GPUs, compute (FLOP/s) has scaled faster than memory bandwidth (GB/s). For the attention matrix:

- Reading Q, K from HBM → compute scores → write scores back to HBM → read scores → softmax → write back → read for `att @ V`
- Each round trip to HBM is expensive
- The actual matmuls are fast; the bottleneck is moving the `T × T` matrix in and out of memory

This is exactly what FlashAttention solves: it computes attention in tiles that fit in SRAM (on-chip), never materializing the full `T × T` matrix in HBM. Same result, fraction of the memory movement.

---

## 4. What breaks if the mask is wrong

| Mistake | Symptom | Why |
|---|---|---|
| Upper triangular instead of lower | Loss drops fast in training, garbage at inference | Model cheats by seeing future tokens — works when future is available, fails when it isn't |
| Off-by-one (diagonal included in mask) | Model can't attend to current token | Each position attends to nothing at position 0 |
| Mask not applied before softmax | Future tokens get nonzero weight | Probabilities don't sum to 1 after zeroing |
| Wrong shape (not broadcast-compatible) | Silent wrong results or shape error | Mask applies to wrong positions |

---

## 5. Where NaNs come from

1. **All positions masked**: if every key in a row is `-inf`, softmax computes `exp(-inf) / sum(exp(-inf)) = 0/0 = NaN`. Happens with empty or fully-padded sequences.
2. **Score overflow in float16**: large dot products overflow to `inf` before scaling. `exp(inf)` = `inf`, `inf/inf` = `NaN`. Prevented by the `sqrt(d_k)` scaling and using bfloat16 (wider dynamic range than float16).
3. **NaN propagation**: one NaN in the attention matrix poisons the entire weighted sum — loss becomes NaN, gradients become NaN, training dies.
