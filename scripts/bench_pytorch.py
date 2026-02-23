#!/usr/bin/env python3
"""Wall-clock benchmark for PyTorch operations.

Outputs JSON to target/bench_compare/pytorch.json with timing stats
(median, std, min, max in microseconds) for each operation.
"""

import json
import os
import time
from pathlib import Path
from statistics import median, stdev

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP = 5
ITERS_FAST = 50
ITERS_SLOW = 20  # matmul 512, training step

OUT_DIR = Path("target/bench_compare")


def bench(fn, iters):
    """Run fn for warmup + iters, return list of durations in microseconds."""
    for _ in range(WARMUP):
        fn()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1_000)  # ns -> us
    return times


def stats(times):
    """Return dict with median, std, min, max (all in microseconds)."""
    return {
        "median_us": round(median(times), 2),
        "std_us": round(stdev(times), 2) if len(times) > 1 else 0.0,
        "min_us": round(min(times), 2),
        "max_us": round(max(times), 2),
        "iters": len(times),
    }


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def bench_matmul():
    results = []
    for size in [128, 256, 512]:
        iters = ITERS_SLOW if size == 512 else ITERS_FAST
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        times = bench(lambda: torch.mm(a, b), iters)
        results.append({"op": f"matmul_{size}x{size}", **stats(times)})
    return results


def bench_add():
    results = []
    for n in [100_000, 500_000]:
        a = torch.randn(n)
        b = torch.randn(n)
        times = bench(lambda: torch.add(a, b), ITERS_FAST)
        label = f"add_{n // 1000}k"
        results.append({"op": label, **stats(times)})
    return results


def bench_mul():
    results = []
    for n in [100_000, 500_000]:
        a = torch.randn(n)
        b = torch.randn(n)
        times = bench(lambda: torch.mul(a, b), ITERS_FAST)
        label = f"mul_{n // 1000}k"
        results.append({"op": label, **stats(times)})
    return results


def bench_exp():
    results = []
    for n in [100_000, 500_000]:
        a = torch.randn(n)
        times = bench(lambda: torch.exp(a), ITERS_FAST)
        label = f"exp_{n // 1000}k"
        results.append({"op": label, **stats(times)})
    return results


def bench_relu():
    a = torch.randn(100_000)
    times = bench(lambda: torch.relu(a), ITERS_FAST)
    return [{"op": "relu_100k", **stats(times)}]


def bench_softmax():
    results = []
    for seq in [128, 512]:
        x = torch.randn(8, seq)
        times = bench(lambda: torch.softmax(x, dim=-1), ITERS_FAST)
        results.append({"op": f"softmax_8x{seq}", **stats(times)})
    return results


def bench_mlp_forward():
    w1 = torch.randn(784, 128)
    b1 = torch.randn(1, 128)
    w2 = torch.randn(128, 64)
    b2 = torch.randn(1, 64)
    w3 = torch.randn(64, 10)
    b3 = torch.randn(1, 10)

    x = torch.randn(64, 784)

    def fwd():
        h1 = torch.relu(x @ w1 + b1)
        h2 = torch.relu(h1 @ w2 + b2)
        return h2 @ w3 + b3

    times = bench(fwd, ITERS_FAST)
    return [{"op": "mlp_fwd_64x784", **stats(times)}]


def bench_training_step():
    w1 = torch.randn(784, 128, requires_grad=True)
    b1 = torch.zeros(1, 128, requires_grad=True)
    w2 = torch.randn(128, 64, requires_grad=True)
    b2 = torch.zeros(1, 64, requires_grad=True)
    w3 = torch.randn(64, 10, requires_grad=True)
    b3 = torch.zeros(1, 10, requires_grad=True)

    params = [w1, b1, w2, b2, w3, b3]
    opt = torch.optim.Adam(params, lr=1e-3)
    targets = torch.arange(64) % 10

    def step():
        opt.zero_grad()
        x = torch.randn(64, 784)
        h1 = torch.relu(x @ w1 + b1)
        h2 = torch.relu(h1 @ w2 + b2)
        logits = h2 @ w3 + b3
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        opt.step()

    times = bench(step, ITERS_SLOW)
    return [{"op": "train_step_64", **stats(times)}]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for fn in [
        bench_matmul,
        bench_add,
        bench_mul,
        bench_exp,
        bench_relu,
        bench_softmax,
        bench_mlp_forward,
        bench_training_step,
    ]:
        print(f"  PyTorch: {fn.__name__} ...")
        all_results.extend(fn())

    output = {
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "results": all_results,
    }

    out_path = OUT_DIR / "pytorch.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
