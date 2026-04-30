"""
Scaling tradeoff analysis:
  1. Double context length (T): track memory and latency
  2. Double model width (n_embd): track FLOPs and GPU utilization

Run on the pod:
    python profiling/scaling_experiment.py

Outputs results to stdout — copy into docs/transformer_scaling_analysis.md
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


# ── helpers ───────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def estimate_flops(config, T, B=1):
    """
    Rough FLOPs estimate for one forward pass.
    Each transformer block: ~12 * n_embd^2 * T FLOPs (from Chinchilla paper).
    """
    return 12 * config.n_embd ** 2 * T * config.n_layer * B

def benchmark_forward(model, idx):
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


# ── experiment 1: double context length ───────────────────────────────────────

def context_length_scaling():
    print("\n" + "=" * 65)
    print("EXPERIMENT 1: Doubling context length (T)")
    print("Fixed: n_layer=12, n_embd=768, n_head=12, B=4")
    print("=" * 65)
    print(f"{'T':>6} {'latency_ms':>12} {'peak_mb':>10} {'mem_ratio':>10} {'lat_ratio':>10}")
    print("-" * 55)

    seq_lens = [256, 512, 1024, 2048]
    base_ms, base_mb = None, None

    for T in seq_lens:
        config = GPTConfig(block_size=T, vocab_size=50304, n_embd=768, n_layer=12, n_head=12)
        model  = GPT(config).to(DEVICE).to(DTYPE)
        idx    = torch.randint(0, config.vocab_size, (4, T), device=DEVICE)

        avg_ms, peak_mb = benchmark_forward(model, idx)

        if base_ms is None:
            base_ms, base_mb = avg_ms, peak_mb
            mem_ratio = lat_ratio = 1.0
        else:
            mem_ratio = peak_mb / base_mb
            lat_ratio = avg_ms / base_ms

        print(f"{T:>6} {avg_ms:>12.2f} {peak_mb:>10.1f} {mem_ratio:>10.2f}x {lat_ratio:>10.2f}x")
        del model

    print("\nExpected: memory and latency should grow ~4x each time T doubles (O(n²))")


# ── experiment 2: double model width ──────────────────────────────────────────

def model_width_scaling():
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: Doubling model width (n_embd)")
    print("Fixed: n_layer=12, T=512, B=4")
    print("=" * 65)
    print(f"{'n_embd':>8} {'params_M':>10} {'flops_G':>10} {'latency_ms':>12} {'peak_mb':>10}")
    print("-" * 58)

    widths = [384, 768, 1536]

    for n_embd in widths:
        n_head = n_embd // 64  # keep head_size=64
        T      = 512
        config = GPTConfig(block_size=T, vocab_size=50304, n_embd=n_embd, n_layer=12, n_head=n_head)
        model  = GPT(config).to(DEVICE).to(DTYPE)
        idx    = torch.randint(0, config.vocab_size, (4, T), device=DEVICE)

        params_m = count_params(model) / 1e6
        flops_g  = estimate_flops(config, T, B=4) / 1e9

        avg_ms, peak_mb = benchmark_forward(model, idx)

        print(f"{n_embd:>8} {params_m:>10.1f} {flops_g:>10.1f} {avg_ms:>12.2f} {peak_mb:>10.1f}")
        del model

    print("\nExpected: FLOPs grow ~4x per doubling (O(n_embd²)); latency grows similarly")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    torch.set_float32_matmul_precision('high')

    context_length_scaling()
    model_width_scaling()

    print("\nDone. Copy results into docs/transformer_scaling_analysis.md")
