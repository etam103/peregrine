#!/usr/bin/env python3
"""Wall-clock benchmark for tinygrad operations (CPU mode).

Outputs JSON to target/bench_compare/tinygrad.json with timing stats
(median, std, min, max in microseconds) for each operation.

Covers operations from Peregrine benchmarks where tinygrad has support.
Uses try/except to gracefully skip unsupported ops.
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


def try_bench(name, fn, iters=ITERS_FAST):
    """Attempt a benchmark; return results list or empty list on failure."""
    try:
        return [{"op": name, **stats(bench(fn, iters))}]
    except Exception as e:
        print(f"  tinygrad: skipping {name} ({e})")
        return []


# ---------------------------------------------------------------------------
# Existing benchmarks
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
    Tensor.training = False
    return [{"op": "train_step_64", **stats(times)}]


# ---------------------------------------------------------------------------
# Phase 1A: Unary math ops (100k elements)
# ---------------------------------------------------------------------------


def bench_unary_math():
    results = []
    n = 100_000

    a = Tensor.randn(n).realize()
    a_pos = (Tensor.randn(n).abs() + 1e-5).realize()
    a_unit = Tensor.linspace(-0.9, 0.9, steps=n).realize()

    # reciprocal
    results.extend(try_bench("reciprocal_100k", lambda: a_pos.reciprocal().realize()))

    # square (a * a)
    results.extend(try_bench("square_100k", lambda: (a * a).realize()))

    # floor
    results.extend(try_bench("floor_100k", lambda: a.floor().realize()))

    # ceil
    results.extend(try_bench("ceil_100k", lambda: a.ceil().realize()))

    # sign
    if hasattr(Tensor, "sign") or hasattr(a, "sign"):
        results.extend(try_bench("sign_100k", lambda: a.sign().realize()))
    else:
        print("  tinygrad: skipping sign_100k (not available)")

    # log2
    results.extend(try_bench("log2_100k", lambda: a_pos.log2().realize()))

    # sinh
    results.extend(try_bench("sinh_100k", lambda: a.sinh().realize()))

    # cosh
    results.extend(try_bench("cosh_100k", lambda: a.cosh().realize()))

    # arcsin (asin)
    if hasattr(a_unit, "asin"):
        results.extend(try_bench("arcsin_100k", lambda: a_unit.asin().realize()))
    elif hasattr(a_unit, "arcsin"):
        results.extend(try_bench("arcsin_100k", lambda: a_unit.arcsin().realize()))
    else:
        print("  tinygrad: skipping arcsin_100k (not available)")

    # arctan (atan)
    if hasattr(a, "atan"):
        results.extend(try_bench("arctan_100k", lambda: a.atan().realize()))
    elif hasattr(a, "arctan"):
        results.extend(try_bench("arctan_100k", lambda: a.arctan().realize()))
    else:
        print("  tinygrad: skipping arctan_100k (not available)")

    return results


# ---------------------------------------------------------------------------
# Phase 1B-D: Binary math / clip / compare (100k elements)
# ---------------------------------------------------------------------------


def bench_binary_math():
    results = []
    n = 100_000

    a = Tensor.randn(n).realize()
    b = Tensor.randn(n).realize()
    cond = (Tensor.randn(n) > 0).realize()

    # maximum
    if hasattr(Tensor, "maximum"):
        results.extend(try_bench("maximum_100k", lambda: Tensor.maximum(a, b).realize()))
    else:
        results.extend(try_bench("maximum_100k", lambda: (a * (a >= b).float() + b * (b > a).float()).realize()))

    # minimum
    if hasattr(Tensor, "minimum"):
        results.extend(try_bench("minimum_100k", lambda: Tensor.minimum(a, b).realize()))
    else:
        results.extend(try_bench("minimum_100k", lambda: (a * (a <= b).float() + b * (b < a).float()).realize()))

    # clip
    results.extend(try_bench("clip_100k", lambda: a.clip(-0.5, 0.5).realize()))

    # where
    results.extend(try_bench("where_100k", lambda: cond.where(a, b).realize()))

    # greater
    results.extend(try_bench("greater_100k", lambda: (a > b).realize()))

    # equal
    results.extend(try_bench("equal_100k", lambda: (a == b).realize()))

    return results


# ---------------------------------------------------------------------------
# Phase 1E: Axis reductions (256x512, reduce along axis=1)
# ---------------------------------------------------------------------------


def bench_axis_reductions():
    results = []

    x = Tensor.randn(256, 512).realize()

    # sum
    results.extend(try_bench("sum_axis_256x512", lambda: x.sum(axis=1).realize()))

    # mean
    results.extend(try_bench("mean_axis_256x512", lambda: x.mean(axis=1).realize()))

    # max
    results.extend(try_bench("max_axis_256x512", lambda: x.max(axis=1).realize()))

    # min
    results.extend(try_bench("min_axis_256x512", lambda: x.min(axis=1).realize()))

    # cumsum
    results.extend(try_bench("cumsum_256x512", lambda: x.cumsum(axis=1).realize()))

    # argmax
    results.extend(try_bench("argmax_axis_256x512", lambda: x.argmax(axis=1).realize()))

    return results


# ---------------------------------------------------------------------------
# Phase 1F: Shape ops
# ---------------------------------------------------------------------------


def bench_shape_ops():
    results = []

    # tril 256x256
    x_sq = Tensor.randn(256, 256).realize()
    if hasattr(Tensor, "tril") or hasattr(x_sq, "tril"):
        results.extend(try_bench("tril_256x256", lambda: x_sq.tril().realize()))
    else:
        print("  tinygrad: skipping tril_256x256 (not available)")

    # triu 256x256
    if hasattr(Tensor, "triu") or hasattr(x_sq, "triu"):
        results.extend(try_bench("triu_256x256", lambda: x_sq.triu().realize()))
    else:
        print("  tinygrad: skipping triu_256x256 (not available)")

    # pad 64x128
    x_small = Tensor.randn(64, 128).realize()
    # tinygrad pad takes ((top, bottom), (left, right)) or similar tuple-of-tuples
    results.extend(try_bench("pad_64x128", lambda: x_small.pad(((1, 1), (2, 2))).realize()))

    # stack 8 tensors of [64, 128]
    tensors = [Tensor.randn(64, 128).realize() for _ in range(8)]
    results.extend(try_bench("stack_8x64x128", lambda: Tensor.stack(*tensors, dim=0).realize()))

    return results


# ---------------------------------------------------------------------------
# Phase 2: Activations (100k elements)
# ---------------------------------------------------------------------------


def bench_activations():
    results = []
    n = 100_000
    a = Tensor.randn(n).realize()

    # silu
    results.extend(try_bench("silu_100k", lambda: a.silu().realize()))

    # softplus
    results.extend(try_bench("softplus_100k", lambda: a.softplus().realize()))

    # mish
    results.extend(try_bench("mish_100k", lambda: a.mish().realize()))

    # leaky_relu
    results.extend(try_bench("leaky_relu_100k", lambda: a.leakyrelu(0.01).realize()))

    # elu
    results.extend(try_bench("elu_100k", lambda: a.elu().realize()))

    # gelu
    results.extend(try_bench("gelu_100k", lambda: a.gelu().realize()))

    # relu6 (relu clipped to 6)
    results.extend(try_bench("relu6_100k", lambda: a.relu().clip(0, 6).realize()))

    # selu
    if hasattr(a, "selu"):
        results.extend(try_bench("selu_100k", lambda: a.selu().realize()))
    else:
        # Manual SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        results.extend(try_bench(
            "selu_100k",
            lambda: (scale * (a.relu() + (a.exp() - 1).clip(float("-inf"), 0) * alpha)).realize()
        ))

    return results


# ---------------------------------------------------------------------------
# Phase 3A: Loss functions (batch=64, classes=10)
# ---------------------------------------------------------------------------


def bench_losses():
    results = []
    batch = 64
    classes = 10

    pred = Tensor.randn(batch, classes).realize()
    target = Tensor.randn(batch, classes).realize()
    targets_idx = Tensor(list(range(batch))) % classes

    # Cross-entropy (manual: log_softmax + nll)
    def cross_entropy_fn():
        log_probs = pred.log_softmax(axis=-1)
        loss = log_probs.sparse_categorical_crossentropy(targets_idx)
        loss.realize()

    results.extend(try_bench("cross_entropy_64x10", cross_entropy_fn))

    # MSE loss (manual)
    def mse_loss_fn():
        diff = pred - target
        loss = (diff * diff).mean()
        loss.realize()

    results.extend(try_bench("mse_loss_64x10", mse_loss_fn))

    # L1 loss (manual)
    def l1_loss_fn():
        loss = (pred - target).abs().mean()
        loss.realize()

    results.extend(try_bench("l1_loss_64x10", l1_loss_fn))

    return results


# ---------------------------------------------------------------------------
# Phase 5: Random
# ---------------------------------------------------------------------------


def bench_random():
    results = []

    results.extend(try_bench("rand_uniform_100k", lambda: Tensor.rand(100_000).realize()))
    results.extend(try_bench("rand_normal_100k", lambda: Tensor.randn(100_000).realize()))
    results.extend(try_bench("rand_uniform_1M", lambda: Tensor.rand(1_000_000).realize()))
    results.extend(try_bench("rand_normal_1M", lambda: Tensor.randn(1_000_000).realize()))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for fn in [
        # Existing core benchmarks
        bench_matmul,
        bench_add,
        bench_mul,
        bench_exp,
        bench_relu,
        bench_softmax,
        bench_mlp_forward,
        bench_training_step,
        # Phase 1A: Unary math
        bench_unary_math,
        # Phase 1B-D: Binary math / clip / compare
        bench_binary_math,
        # Phase 1E: Axis reductions
        bench_axis_reductions,
        # Phase 1F: Shape ops
        bench_shape_ops,
        # Phase 2: Activations
        bench_activations,
        # Phase 3A: Loss functions
        bench_losses,
        # Phase 5: Random
        bench_random,
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
    print(f"  Total operations benchmarked: {len(all_results)}")


if __name__ == "__main__":
    main()
