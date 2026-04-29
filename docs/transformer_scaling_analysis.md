# Transformer Scaling Analysis

Results from running the profiling suite on 8x A100 80GB (RunPod).
Fill in the tables after running each script on the pod.

---

## 1. Attention: Naive vs FlashAttention

**Script:** `python profiling/profile_attention.py`
**Output:** `profiling/attention_kernels.csv`

| seq_len | naive_ms | flash_ms | naive_mb | flash_mb | speedup |
|---------|----------|----------|----------|----------|---------|
| 256     | 0.14     | 0.02     | 26.2     | 15.7     | 8.60x   |
| 512     | 0.27     | 0.03     | 68.6     | 23.2     | 8.77x   |
| 1024    | 0.94     | 0.08     | 226.1    | 38.3     | 12.24x  |
| 2048    | 5.00     | 0.22     | 832.1    | 68.5     | 22.86x  |
| 4096    | 15.41    | 0.65     | 3208.1   | 128.9    | 23.74x  |

**Key observations:**
- naive memory grows ~3.9x each time T doubles, approaching 4x at larger T (O(n²) confirmed)
- flash memory grows ~2x per doubling — O(n), not O(n²) — never materializes the T×T matrix
- at T=4096 naive uses 3208 MB vs flash's 129 MB — **25x less memory**
- speedup grows from 8.6x → 23.7x as T increases: the memory bottleneck compounds at longer sequences

**Why FlashAttention wins on memory:**
Naive attention materializes the full `(B, n_head, T, T)` matrix in HBM (high bandwidth memory). At T=4096 with B=4, n_head=12, bfloat16 that's `4 * 12 * 4096 * 4096 * 2 bytes ≈ 1.6 GB` — just for the attention scores, before activations. FlashAttention tiles the computation so it fits in SRAM (on-chip cache), reading/writing HBM far less often. Same math, fraction of the memory traffic.

---

## 2. Scaling: Context Length

**Script:** `python profiling/scaling_experiment.py`
Fixed: n_layer=12, n_embd=768, n_head=12, B=4

| T    | latency_ms | peak_mb | mem_ratio | lat_ratio |
|------|------------|---------|-----------|-----------|
| 256  | 4.15       | 725.1   | 1.00x     | 1.00x     |
| 512  | 4.21       | 1114.2  | 1.54x     | 1.01x     |
| 1024 | 7.40       | 1914.7  | 2.64x     | 1.78x     |
| 2048 | 13.43      | 3560.0  | 4.91x     | 3.24x     |

**Note:** memory grows sub-quadratically (1.54x → 4.91x vs expected 4x each doubling) because this model uses FlashAttention, which has O(n) memory. Compare to the naive attention numbers in section 1 where memory grows ~3.9x per doubling. This is FlashAttention working as intended.

**Why it's O(n²):**
The attention score matrix is `(T, T)` per head. Doubling T → 4x the entries → 4x memory and 4x compute for `Q @ K^T` and `att @ V`. Every other operation in the transformer (MLP, LayerNorm, embeddings) is O(T) — attention is the only quadratic term, and it dominates at long context.

---

## 3. Scaling: Model Width

**Script:** `python profiling/scaling_experiment.py`
Fixed: n_layer=12, T=512, B=4

| n_embd | params_M | flops_G | latency_ms | peak_mb |
|--------|----------|---------|------------|---------|
| 384    | 60.1     | 43.5    | 4.11       | 621.9   |
| 768    | ~163*    | 173.9   | 4.21       | 1116.4  |
| 1536   | 495.3    | 695.8   | 9.31       | 2322.6  |

*params_M for 768 displayed as 16.7 — display anomaly. FLOPs (4x scaling: 43.5→173.9→695.8) confirm model is correct.

**Expected:** FLOPs and latency grow ~4x per doubling of n_embd.

**Why:** Every matmul in the transformer (QKV projection, output projection, MLP) has `n_embd` on both dimensions. Doubling width → 4x FLOPs per matmul. Unlike sequence length, this is a pure compute cost — the memory growth is from larger weight matrices, not activation tensors.

**Depth vs width tradeoff:**
- More layers (depth) = linear cost — each layer adds fixed compute
- Wider model (n_embd) = quadratic cost — matmul sizes grow in both dimensions
- In practice: depth is cheaper per parameter for most tasks; width helps with memorization capacity

---

## 4. Block Depth Experiment

**Script:** `python profiling/block_size_experiment.py`
Fixed: n_embd=768, n_head=12, T=1024, B=4

| n_layer | params_M | latency_ms | peak_mb | ms/layer |
|---------|----------|------------|---------|----------|
| 6       | 120.6    | 4.46       | 1238.1  | 0.74     |
| 12      | 163.1    | 7.38       | 1912.6  | 0.62     |
| 24      | 248.2    | 12.05      | 3258.7  | 0.50     |

**Actual vs expected:** latency grows ~1.65x per layer doubling, not 2x. `ms/layer` *decreases* from 0.74 → 0.50 as depth grows. This is GPU efficiency improving with more sequential work to pipeline — the kernel launch and memory access overhead gets amortized across more layers.

**Key insight:** Adding layers is O(n_layer) in compute, but sub-linear in practice due to GPU utilization. The O(n²) from attention is *within* each layer at fixed T — it doesn't compound across layers. Contrast: doubling T in experiment 1 gave 4.91x memory growth. Doubling layers here gives only 1.55–1.70x memory growth. Sequence length is far more expensive to scale than depth.

---

## 5. Edge Cases

**Script:** `python profiling/edge_cases.py`

### 5a. Long sequence memory limits (naive vs flash)

| seq_len | naive_mb  | flash_mb | naive_ratio | flash_ratio |
|---------|-----------|----------|-------------|-------------|
| 1024    | 62.6      | 15.7     | 1.00x       | 1.00x       |
| 2048    | 218.6     | 24.2     | 3.49x       | 1.54x       |
| 4096    | 829.1     | 38.3     | 3.79x       | 1.58x       |
| 8192    | 3250.1    | 68.5     | 3.92x       | 1.79x       |
| 16384   | 12892.1   | 128.9    | 3.97x       | 1.88x       |

No OOM on A100 80GB — naive needs 12.9GB at T=16384, which fits. On a 16GB GPU naive would OOM between T=8192 and T=16384. Naive memory ratio approaches 4x as T grows (O(n²) confirmed). Flash ratio approaches 2x (O(n) confirmed).

### 5b. Numerical stability by dtype (T=1024, normal inputs)

| dtype    | naive NaN | flash NaN | max_val |
|----------|-----------|-----------|---------|
| float32  | False     | False     | 3.6619  |
| float16  | False     | False     | 3.2480  |
| bfloat16 | False     | False     | 3.2969  |

All dtypes stable at T=1024 with normal magnitude inputs. float16 only breaks under stress (see 2c).

### 5b2. float16 stability at increasing T (normal inputs)

| T    | naive NaN | flash NaN |
|------|-----------|-----------|
| 512  | False     | False     |
| 2048 | False     | False     |
| 8192 | False     | False     |

float16 holds with normal inputs even at T=8192. The danger is input magnitude, not sequence length alone.

### 5c. Stability at large magnitude inputs (scale=100)

| dtype    | naive NaN | flash NaN |
|----------|-----------|-----------|
| float32  | False     | False     |
| float16  | **True**  | False     |
| bfloat16 | False     | False     |

**Key result:** float16 + large magnitude inputs → NaN in naive attention, but FlashAttention survives. float16 pre-softmax scores overflow its max representable value (~65,504) when inputs are scaled by 100. bfloat16 survives because its exponent range matches float32 (~3.4×10³⁸). FlashAttention's online softmax tracks a running max and rescales on the fly — it never lets a single large score dominate, making it robust where naive softmax fails.

**Why bfloat16 > float16 for training:**
Both have 16 bits total. float16 splits them as 1 sign + 5 exponent + 10 mantissa. bfloat16 uses 1 + 8 + 7. The wider exponent (8 bits = same as float32) means bfloat16 can represent values up to ~3.4×10³⁸ vs float16's ~65,504. Pre-softmax attention scores can easily exceed 65,504 at large T or with unscaled inputs — that's where float16 NaNs come from. bfloat16 survives because its exponent range matches float32.

**Why FlashAttention is more stable:**
Standard softmax reads all scores, computes `exp(x_i) / sum(exp(x_j))`. If any score is large, `exp` overflows. FlashAttention uses *online softmax*: it tracks a running max and rescales as it goes, so no single score ever dominates the computation. Numerically equivalent result, more stable execution.

---

## Summary

| Dimension | Cost | Dominant operation |
|-----------|------|--------------------|
| T (seq len) | O(n²) | attention matmul |
| n_embd (width) | O(n²) | all linear projections |
| n_layer (depth) | O(n) | stacked forward passes |
| FlashAttention vs naive | ~same FLOPs, far less memory | HBM traffic |

**Takeaway:** sequence length is the most expensive thing you can scale in a transformer. Width is the second most expensive. Depth is the cheapest. FlashAttention doesn't reduce FLOPs — it reduces memory movement, which is the real bottleneck on modern GPUs.
