"""
Block size (depth) experiment: compare 6, 12, 24 transformer layers.
Measures memory, latency, and throughput at fixed model width and sequence length.
Documents the O(n²) attention cost compounding across layers.

Run on the pod:
    python profiling/block_size_experiment.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from model import GPT, GPTConfig

DEVICE = "cuda"
DTYPE  = torch.bfloat16
RUNS   = 10
WARMUP = 3


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(model, idx):
    for _ in range(WARMUP):
        _ = model(idx)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(RUNS):
        _ = model(idx)
    end.record()
    torch.cuda.synchronize()

    avg_ms  = start.elapsed_time(end) / RUNS
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    return avg_ms, peak_mb


# ── experiment ────────────────────────────────────────────────────────────────

def run():
    B, T = 4, 1024
    layer_counts = [6, 12, 24]

    print("=" * 65)
    print("EXPERIMENT: Varying transformer depth (n_layer)")
    print(f"Fixed: n_embd=768, n_head=12, T={T}, B={B}")
    print("=" * 65)
    print(f"{'n_layer':>8} {'params_M':>10} {'latency_ms':>12} {'peak_mb':>10} {'ms/layer':>10}")
    print("-" * 58)

    base_ms = None

    for n_layer in layer_counts:
        config = GPTConfig(block_size=T, vocab_size=50304, n_layer=n_layer, n_embd=768, n_head=12)
        model  = GPT(config).to(DEVICE).to(DTYPE)
        idx    = torch.randint(0, config.vocab_size, (B, T), device=DEVICE)
        params = sum(p.numel() for p in model.parameters()) / 1e6

        avg_ms, peak_mb = benchmark(model, idx)
        ms_per_layer = avg_ms / n_layer

        if base_ms is None:
            base_ms = avg_ms

        print(f"{n_layer:>8} {params:>10.1f} {avg_ms:>12.2f} {peak_mb:>10.1f} {ms_per_layer:>10.2f}")
        del model

    print()
    print("Key observations to record:")
    print("  - latency should scale linearly with n_layer (attention cost per layer is fixed)")
    print("  - memory grows with depth due to activation storage for backprop")
    print("  - ms/layer stays roughly constant — depth is linear cost, not quadratic")
    print("  - contrast with T doubling (experiment 1): that was quadratic in attention")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    run()
