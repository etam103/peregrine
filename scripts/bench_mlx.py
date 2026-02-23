#!/usr/bin/env python3
"""Wall-clock benchmark for MLX operations (CPU mode).

Outputs JSON to target/bench_compare/mlx.json with timing stats
(median, std, min, max in microseconds) for each operation.
"""

import json
import time
from pathlib import Path
from statistics import median, stdev

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

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
        a = mx.random.normal((size, size))
        b = mx.random.normal((size, size))
        mx.eval(a, b)

        def run(a=a, b=b):
            c = a @ b
            mx.eval(c)

        times = bench(run, iters)
        results.append({"op": f"matmul_{size}x{size}", **stats(times)})
    return results


def bench_add():
    results = []
    for n in [100_000, 500_000]:
        a = mx.random.normal((n,))
        b = mx.random.normal((n,))
        mx.eval(a, b)

        def run(a=a, b=b):
            c = mx.add(a, b)
            mx.eval(c)

        times = bench(run, ITERS_FAST)
        results.append({"op": f"add_{n // 1000}k", **stats(times)})
    return results


def bench_mul():
    results = []
    for n in [100_000, 500_000]:
        a = mx.random.normal((n,))
        b = mx.random.normal((n,))
        mx.eval(a, b)

        def run(a=a, b=b):
            c = mx.multiply(a, b)
            mx.eval(c)

        times = bench(run, ITERS_FAST)
        results.append({"op": f"mul_{n // 1000}k", **stats(times)})
    return results


def bench_exp():
    results = []
    for n in [100_000, 500_000]:
        a = mx.random.normal((n,))
        mx.eval(a)

        def run(a=a):
            c = mx.exp(a)
            mx.eval(c)

        times = bench(run, ITERS_FAST)
        results.append({"op": f"exp_{n // 1000}k", **stats(times)})
    return results


def bench_relu():
    a = mx.random.normal((100_000,))
    mx.eval(a)

    def run():
        c = mx.maximum(a, 0)
        mx.eval(c)

    times = bench(run, ITERS_FAST)
    return [{"op": "relu_100k", **stats(times)}]


def bench_softmax():
    results = []
    for seq in [128, 512]:
        x = mx.random.normal((8, seq))
        mx.eval(x)

        def run(x=x):
            c = mx.softmax(x, axis=-1)
            mx.eval(c)

        times = bench(run, ITERS_FAST)
        results.append({"op": f"softmax_8x{seq}", **stats(times)})
    return results


def bench_mlp_forward():
    w1 = mx.random.normal((784, 128))
    b1 = mx.random.normal((1, 128))
    w2 = mx.random.normal((128, 64))
    b2 = mx.random.normal((1, 64))
    w3 = mx.random.normal((64, 10))
    b3 = mx.random.normal((1, 10))
    x = mx.random.normal((64, 784))
    mx.eval(w1, b1, w2, b2, w3, b3, x)

    def fwd():
        h1 = mx.maximum((x @ w1) + b1, 0)
        h2 = mx.maximum((h1 @ w2) + b2, 0)
        out = (h2 @ w3) + b3
        mx.eval(out)

    times = bench(fwd, ITERS_FAST)
    return [{"op": "mlp_fwd_64x784", **stats(times)}]


def bench_training_step():
    w1 = mx.random.normal((784, 128))
    b1 = mx.zeros((1, 128))
    w2 = mx.random.normal((128, 64))
    b2 = mx.zeros((1, 64))
    w3 = mx.random.normal((64, 10))
    b3 = mx.zeros((1, 10))
    mx.eval(w1, b1, w2, b2, w3, b3)

    targets = mx.array(list(range(64)), dtype=mx.int32) % 10

    # Adam state
    lr = 1e-3
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    params = [w1, b1, w2, b2, w3, b3]
    m = [mx.zeros_like(p) for p in params]
    v = [mx.zeros_like(p) for p in params]
    t = [0]

    def loss_fn(params):
        w1, b1, w2, b2, w3, b3 = params
        x = mx.random.normal((64, 784))
        h1 = mx.maximum((x @ w1) + b1, 0)
        h2 = mx.maximum((h1 @ w2) + b2, 0)
        logits = (h2 @ w3) + b3
        return nn.losses.cross_entropy(logits, targets).mean()

    loss_grad_fn = mx.value_and_grad(loss_fn)

    def step():
        nonlocal params, m, v
        t[0] += 1
        loss, grads = loss_grad_fn(params)
        # Manual Adam update
        new_params = []
        new_m = []
        new_v = []
        bc1 = 1 - beta1 ** t[0]
        bc2 = 1 - beta2 ** t[0]
        for p, g, mi, vi in zip(params, grads, m, v):
            mi = beta1 * mi + (1 - beta1) * g
            vi = beta2 * vi + (1 - beta2) * (g * g)
            m_hat = mi / bc1
            v_hat = vi / bc2
            p = p - lr * m_hat / (mx.sqrt(v_hat) + eps)
            new_params.append(p)
            new_m.append(mi)
            new_v.append(vi)
        params = new_params
        m = new_m
        v = new_v
        mx.eval(loss, *params, *m, *v)

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
        print(f"  MLX: {fn.__name__} ...")
        all_results.extend(fn())

    output = {
        "framework": "mlx",
        "mlx_version": mx.__version__ if hasattr(mx, "__version__") else "unknown",
        "device": "cpu",
        "results": all_results,
    }

    out_path = OUT_DIR / "mlx.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
