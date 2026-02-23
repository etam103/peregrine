#!/usr/bin/env python3
"""Wall-clock benchmark for tinygrad operations (CPU mode).

Outputs JSON to target/bench_compare/tinygrad.json with timing stats
(median, std, min, max in microseconds) for each operation.
"""

import json
import os
import time
from pathlib import Path
from statistics import median, stdev

# Force CPU before importing tinygrad
os.environ["DEVICE"] = "CPU"

from tinygrad import Tensor, Device
from tinygrad.nn.optim import Adam

Device.DEFAULT = "CPU"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP = 5
ITERS_FAST = 50
ITERS_SLOW = 20

OUT_DIR = Path("target/bench_compare")


def bench(fn, iters):
    for _ in range(WARMUP):
        fn()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1_000)
    return times


def stats(times):
    return {
        "median_us": round(median(times), 2),
        "std_us": round(stdev(times), 2) if len(times) > 1 else 0.0,
        "min_us": round(min(times), 2),
        "max_us": round(max(times), 2),
        "iters": len(times),
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_matmul():
    results = []
    for size in [128, 256, 512]:
        iters = ITERS_SLOW if size == 512 else ITERS_FAST
        a = Tensor.randn(size, size).realize()
        b = Tensor.randn(size, size).realize()

        def run(a=a, b=b):
            (a @ b).realize()

        times = bench(run, iters)
        results.append({"op": f"matmul_{size}x{size}", **stats(times)})
    return results


def bench_add():
    results = []
    for n in [100_000, 500_000]:
        a = Tensor.randn(n).realize()
        b = Tensor.randn(n).realize()

        def run(a=a, b=b):
            (a + b).realize()

        times = bench(run, ITERS_FAST)
        results.append({"op": f"add_{n // 1000}k", **stats(times)})
    return results


def bench_mul():
    results = []
    for n in [100_000, 500_000]:
        a = Tensor.randn(n).realize()
        b = Tensor.randn(n).realize()

        def run(a=a, b=b):
            (a * b).realize()

        times = bench(run, ITERS_FAST)
        results.append({"op": f"mul_{n // 1000}k", **stats(times)})
    return results


def bench_exp():
    results = []
    for n in [100_000, 500_000]:
        a = Tensor.randn(n).realize()

        def run(a=a):
            a.exp().realize()

        times = bench(run, ITERS_FAST)
        results.append({"op": f"exp_{n // 1000}k", **stats(times)})
    return results


def bench_relu():
    a = Tensor.randn(100_000).realize()

    def run():
        a.relu().realize()

    times = bench(run, ITERS_FAST)
    return [{"op": "relu_100k", **stats(times)}]


def bench_softmax():
    results = []
    for seq in [128, 512]:
        x = Tensor.randn(8, seq).realize()

        def run(x=x):
            x.softmax(axis=-1).realize()

        times = bench(run, ITERS_FAST)
        results.append({"op": f"softmax_8x{seq}", **stats(times)})
    return results


def bench_mlp_forward():
    w1 = Tensor.randn(784, 128).realize()
    b1 = Tensor.randn(1, 128).realize()
    w2 = Tensor.randn(128, 64).realize()
    b2 = Tensor.randn(1, 64).realize()
    w3 = Tensor.randn(64, 10).realize()
    b3 = Tensor.randn(1, 10).realize()
    x = Tensor.randn(64, 784).realize()

    def fwd():
        h1 = ((x @ w1) + b1).relu()
        h2 = ((h1 @ w2) + b2).relu()
        out = (h2 @ w3) + b3
        out.realize()

    times = bench(fwd, ITERS_FAST)
    return [{"op": "mlp_fwd_64x784", **stats(times)}]


def bench_training_step():
    Tensor.training = True
    w1 = Tensor.randn(784, 128, requires_grad=True).realize()
    b1 = Tensor.zeros(1, 128, requires_grad=True).realize()
    w2 = Tensor.randn(128, 64, requires_grad=True).realize()
    b2 = Tensor.zeros(1, 64, requires_grad=True).realize()
    w3 = Tensor.randn(64, 10, requires_grad=True).realize()
    b3 = Tensor.zeros(1, 10, requires_grad=True).realize()

    params = [w1, b1, w2, b2, w3, b3]
    opt = Adam(params, lr=1e-3)
    targets = Tensor(list(range(64))) % 10

    def step():
        opt.zero_grad()
        x = Tensor.randn(64, 784)
        h1 = ((x @ w1) + b1).relu()
        h2 = ((h1 @ w2) + b2).relu()
        logits = (h2 @ w3) + b3
        loss = logits.sparse_categorical_crossentropy(targets)
        loss.backward()
        opt.step()
        loss.realize()

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
        print(f"  tinygrad: {fn.__name__} ...")
        all_results.extend(fn())

    output = {
        "framework": "tinygrad",
        "device": "cpu",
        "results": all_results,
    }

    out_path = OUT_DIR / "tinygrad.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
