#!/usr/bin/env python3
"""Wall-clock benchmark for JAX operations (CPU mode).

Outputs JSON to target/bench_compare/jax.json with timing stats
(median, std, min, max in microseconds) for each operation.
"""

import json
import os
import time
from pathlib import Path
from statistics import median, stdev

# Force CPU before importing JAX
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

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
        a = jnp.array(jax.random.normal(jax.random.key(0), (size, size)))
        b = jnp.array(jax.random.normal(jax.random.key(1), (size, size)))

        def run(a=a, b=b):
            c = a @ b
            c.block_until_ready()

        times = bench(run, iters)
        results.append({"op": f"matmul_{size}x{size}", **stats(times)})
    return results


def bench_add():
    results = []
    for n in [100_000, 500_000]:
        a = jnp.array(jax.random.normal(jax.random.key(0), (n,)))
        b = jnp.array(jax.random.normal(jax.random.key(1), (n,)))

        def run(a=a, b=b):
            c = jnp.add(a, b)
            c.block_until_ready()

        times = bench(run, ITERS_FAST)
        results.append({"op": f"add_{n // 1000}k", **stats(times)})
    return results


def bench_mul():
    results = []
    for n in [100_000, 500_000]:
        a = jnp.array(jax.random.normal(jax.random.key(0), (n,)))
        b = jnp.array(jax.random.normal(jax.random.key(1), (n,)))

        def run(a=a, b=b):
            c = jnp.multiply(a, b)
            c.block_until_ready()

        times = bench(run, ITERS_FAST)
        results.append({"op": f"mul_{n // 1000}k", **stats(times)})
    return results


def bench_exp():
    results = []
    for n in [100_000, 500_000]:
        a = jnp.array(jax.random.normal(jax.random.key(0), (n,)))

        def run(a=a):
            c = jnp.exp(a)
            c.block_until_ready()

        times = bench(run, ITERS_FAST)
        results.append({"op": f"exp_{n // 1000}k", **stats(times)})
    return results


def bench_relu():
    a = jnp.array(jax.random.normal(jax.random.key(0), (100_000,)))

    def run():
        c = jax.nn.relu(a)
        c.block_until_ready()

    times = bench(run, ITERS_FAST)
    return [{"op": "relu_100k", **stats(times)}]


def bench_softmax():
    results = []
    for seq in [128, 512]:
        x = jnp.array(jax.random.normal(jax.random.key(0), (8, seq)))

        def run(x=x):
            c = jax.nn.softmax(x, axis=-1)
            c.block_until_ready()

        times = bench(run, ITERS_FAST)
        results.append({"op": f"softmax_8x{seq}", **stats(times)})
    return results


def bench_mlp_forward():
    key = jax.random.key(42)
    keys = jax.random.split(key, 7)
    w1 = jax.random.normal(keys[0], (784, 128))
    b1 = jax.random.normal(keys[1], (1, 128))
    w2 = jax.random.normal(keys[2], (128, 64))
    b2 = jax.random.normal(keys[3], (1, 64))
    w3 = jax.random.normal(keys[4], (64, 10))
    b3 = jax.random.normal(keys[5], (1, 10))
    x = jax.random.normal(keys[6], (64, 784))

    def fwd():
        h1 = jax.nn.relu(x @ w1 + b1)
        h2 = jax.nn.relu(h1 @ w2 + b2)
        out = h2 @ w3 + b3
        out.block_until_ready()

    times = bench(fwd, ITERS_FAST)
    return [{"op": "mlp_fwd_64x784", **stats(times)}]


def bench_training_step():
    key = jax.random.key(42)
    keys = jax.random.split(key, 6)
    w1 = jax.random.normal(keys[0], (784, 128))
    b1 = jnp.zeros((1, 128))
    w2 = jax.random.normal(keys[1], (128, 64))
    b2 = jnp.zeros((1, 64))
    w3 = jax.random.normal(keys[2], (64, 10))
    b3 = jnp.zeros((1, 10))

    targets = jnp.arange(64) % 10

    # Adam state
    lr = 1e-3
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    params = [w1, b1, w2, b2, w3, b3]
    m_state = [jnp.zeros_like(p) for p in params]
    v_state = [jnp.zeros_like(p) for p in params]
    t = [0]

    def loss_fn(params, x):
        w1, b1, w2, b2, w3, b3 = params
        h1 = jax.nn.relu(x @ w1 + b1)
        h2 = jax.nn.relu(h1 @ w2 + b2)
        logits = h2 @ w3 + b3
        # Cross entropy
        one_hot = jax.nn.one_hot(targets, 10)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    grad_fn = jax.grad(loss_fn)

    def step():
        nonlocal params, m_state, v_state
        t[0] += 1
        x = jax.random.normal(jax.random.key(t[0]), (64, 784))
        grads = grad_fn(params, x)
        # Manual Adam update
        new_params = []
        new_m = []
        new_v = []
        bc1 = 1 - beta1 ** t[0]
        bc2 = 1 - beta2 ** t[0]
        for p, g, mi, vi in zip(params, grads, m_state, v_state):
            mi = beta1 * mi + (1 - beta1) * g
            vi = beta2 * vi + (1 - beta2) * (g * g)
            m_hat = mi / bc1
            v_hat = vi / bc2
            p = p - lr * m_hat / (jnp.sqrt(v_hat) + eps)
            new_params.append(p)
            new_m.append(mi)
            new_v.append(vi)
        params = new_params
        m_state = new_m
        v_state = new_v
        # Block until all results are ready
        jax.block_until_ready(params + m_state + v_state)

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
        print(f"  JAX: {fn.__name__} ...")
        all_results.extend(fn())

    output = {
        "framework": "jax",
        "version": jax.__version__,
        "device": "cpu",
        "results": all_results,
    }

    out_path = OUT_DIR / "jax.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
