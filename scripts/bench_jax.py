#!/usr/bin/env python3
"""Wall-clock benchmark for JAX operations (CPU mode).

Outputs JSON to target/bench_compare/jax.json with timing stats
(median, std, min, max in microseconds) for each operation.

Covers ALL Peregrine benchmark operations with matching op names.
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
import jax.scipy.special

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
# Existing Benchmarks
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
# Phase 1A: Unary math ops (100k elements)
# ---------------------------------------------------------------------------


def bench_unary_math():
    results = []
    n = 100_000
    key = jax.random.key(0)
    keys = jax.random.split(key, 3)

    a = jax.random.normal(keys[0], (n,))
    # Positive values for log/reciprocal ops
    a_pos = jnp.abs(jax.random.normal(keys[1], (n,))) + 1e-5
    # Values in [-0.9, 0.9] for arcsin/arccos
    a_unit = jax.random.uniform(keys[2], (n,), minval=-0.9, maxval=0.9)

    # reciprocal_100k
    def run_reciprocal(a=a_pos):
        c = jnp.reciprocal(a)
        c.block_until_ready()
    times = bench(run_reciprocal, ITERS_FAST)
    results.append({"op": "reciprocal_100k", **stats(times)})

    # square_100k
    def run_square(a=a):
        c = jnp.square(a)
        c.block_until_ready()
    times = bench(run_square, ITERS_FAST)
    results.append({"op": "square_100k", **stats(times)})

    # rsqrt_100k
    def run_rsqrt(a=a_pos):
        c = jax.lax.rsqrt(a)
        c.block_until_ready()
    times = bench(run_rsqrt, ITERS_FAST)
    results.append({"op": "rsqrt_100k", **stats(times)})

    # floor_100k
    def run_floor(a=a):
        c = jnp.floor(a)
        c.block_until_ready()
    times = bench(run_floor, ITERS_FAST)
    results.append({"op": "floor_100k", **stats(times)})

    # ceil_100k
    def run_ceil(a=a):
        c = jnp.ceil(a)
        c.block_until_ready()
    times = bench(run_ceil, ITERS_FAST)
    results.append({"op": "ceil_100k", **stats(times)})

    # round_100k
    def run_round(a=a):
        c = jnp.round(a)
        c.block_until_ready()
    times = bench(run_round, ITERS_FAST)
    results.append({"op": "round_100k", **stats(times)})

    # sign_100k
    def run_sign(a=a):
        c = jnp.sign(a)
        c.block_until_ready()
    times = bench(run_sign, ITERS_FAST)
    results.append({"op": "sign_100k", **stats(times)})

    # expm1_100k
    def run_expm1(a=a):
        c = jnp.expm1(a)
        c.block_until_ready()
    times = bench(run_expm1, ITERS_FAST)
    results.append({"op": "expm1_100k", **stats(times)})

    # log2_100k
    def run_log2(a=a_pos):
        c = jnp.log2(a)
        c.block_until_ready()
    times = bench(run_log2, ITERS_FAST)
    results.append({"op": "log2_100k", **stats(times)})

    # log10_100k
    def run_log10(a=a_pos):
        c = jnp.log10(a)
        c.block_until_ready()
    times = bench(run_log10, ITERS_FAST)
    results.append({"op": "log10_100k", **stats(times)})

    # log1p_100k
    def run_log1p(a=a_pos):
        c = jnp.log1p(a)
        c.block_until_ready()
    times = bench(run_log1p, ITERS_FAST)
    results.append({"op": "log1p_100k", **stats(times)})

    # erf_100k
    def run_erf(a=a):
        c = jax.scipy.special.erf(a)
        c.block_until_ready()
    times = bench(run_erf, ITERS_FAST)
    results.append({"op": "erf_100k", **stats(times)})

    # sinh_100k
    def run_sinh(a=a):
        c = jnp.sinh(a)
        c.block_until_ready()
    times = bench(run_sinh, ITERS_FAST)
    results.append({"op": "sinh_100k", **stats(times)})

    # cosh_100k
    def run_cosh(a=a):
        c = jnp.cosh(a)
        c.block_until_ready()
    times = bench(run_cosh, ITERS_FAST)
    results.append({"op": "cosh_100k", **stats(times)})

    # arcsin_100k
    def run_arcsin(a=a_unit):
        c = jnp.arcsin(a)
        c.block_until_ready()
    times = bench(run_arcsin, ITERS_FAST)
    results.append({"op": "arcsin_100k", **stats(times)})

    # arccos_100k
    def run_arccos(a=a_unit):
        c = jnp.arccos(a)
        c.block_until_ready()
    times = bench(run_arccos, ITERS_FAST)
    results.append({"op": "arccos_100k", **stats(times)})

    # arctan_100k
    def run_arctan(a=a):
        c = jnp.arctan(a)
        c.block_until_ready()
    times = bench(run_arctan, ITERS_FAST)
    results.append({"op": "arctan_100k", **stats(times)})

    # arcsinh_100k
    def run_arcsinh(a=a):
        c = jnp.arcsinh(a)
        c.block_until_ready()
    times = bench(run_arcsinh, ITERS_FAST)
    results.append({"op": "arcsinh_100k", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 1B-D: Binary/clip/compare ops (100k elements)
# ---------------------------------------------------------------------------


def bench_binary_math():
    results = []
    n = 100_000
    key = jax.random.key(0)
    keys = jax.random.split(key, 4)

    a = jax.random.normal(keys[0], (n,))
    b = jax.random.normal(keys[1], (n,))
    a_pos = jnp.abs(jax.random.normal(keys[2], (n,))) + 1e-4
    b_pos = jnp.abs(jax.random.normal(keys[3], (n,))) + 1e-4

    # maximum_100k
    def run_maximum(a=a, b=b):
        c = jnp.maximum(a, b)
        c.block_until_ready()
    times = bench(run_maximum, ITERS_FAST)
    results.append({"op": "maximum_100k", **stats(times)})

    # minimum_100k
    def run_minimum(a=a, b=b):
        c = jnp.minimum(a, b)
        c.block_until_ready()
    times = bench(run_minimum, ITERS_FAST)
    results.append({"op": "minimum_100k", **stats(times)})

    # power_100k
    def run_power(a=a_pos, b=b_pos):
        c = jnp.power(a, b)
        c.block_until_ready()
    times = bench(run_power, ITERS_FAST)
    results.append({"op": "power_100k", **stats(times)})

    # arctan2_100k
    def run_arctan2(a=a, b=b):
        c = jnp.arctan2(a, b)
        c.block_until_ready()
    times = bench(run_arctan2, ITERS_FAST)
    results.append({"op": "arctan2_100k", **stats(times)})

    # logaddexp_100k
    def run_logaddexp(a=a, b=b):
        c = jnp.logaddexp(a, b)
        c.block_until_ready()
    times = bench(run_logaddexp, ITERS_FAST)
    results.append({"op": "logaddexp_100k", **stats(times)})

    # clip_100k
    def run_clip(a=a):
        c = jnp.clip(a, -0.5, 0.5)
        c.block_until_ready()
    times = bench(run_clip, ITERS_FAST)
    results.append({"op": "clip_100k", **stats(times)})

    # where_100k
    cond = jnp.array([i % 2 == 0 for i in range(n)])
    def run_where(cond=cond, a=a, b=b):
        c = jnp.where(cond, a, b)
        c.block_until_ready()
    times = bench(run_where, ITERS_FAST)
    results.append({"op": "where_100k", **stats(times)})

    # greater_100k
    def run_greater(a=a, b=b):
        c = jnp.greater(a, b)
        c.block_until_ready()
    times = bench(run_greater, ITERS_FAST)
    results.append({"op": "greater_100k", **stats(times)})

    # equal_100k
    def run_equal(a=a, b=b):
        c = jnp.equal(a, b)
        c.block_until_ready()
    times = bench(run_equal, ITERS_FAST)
    results.append({"op": "equal_100k", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 1E: Axis reductions (256x512, axis=1)
# ---------------------------------------------------------------------------


def bench_axis_reductions():
    results = []

    key = jax.random.key(0)
    keys = jax.random.split(key, 2)
    x = jax.random.normal(keys[0], (256, 512))
    # Positive values for prod to avoid overflow
    x_pos = jnp.abs(jax.random.uniform(keys[1], (256, 512), minval=0.5, maxval=1.5))

    # sum_axis_256x512
    def run_sum(x=x):
        c = jnp.sum(x, axis=1)
        c.block_until_ready()
    times = bench(run_sum, ITERS_FAST)
    results.append({"op": "sum_axis_256x512", **stats(times)})

    # mean_axis_256x512
    def run_mean(x=x):
        c = jnp.mean(x, axis=1)
        c.block_until_ready()
    times = bench(run_mean, ITERS_FAST)
    results.append({"op": "mean_axis_256x512", **stats(times)})

    # max_axis_256x512
    def run_max(x=x):
        c = jnp.max(x, axis=1)
        c.block_until_ready()
    times = bench(run_max, ITERS_FAST)
    results.append({"op": "max_axis_256x512", **stats(times)})

    # min_axis_256x512
    def run_min(x=x):
        c = jnp.min(x, axis=1)
        c.block_until_ready()
    times = bench(run_min, ITERS_FAST)
    results.append({"op": "min_axis_256x512", **stats(times)})

    # var_256x512
    def run_var(x=x):
        c = jnp.var(x, axis=1)
        c.block_until_ready()
    times = bench(run_var, ITERS_FAST)
    results.append({"op": "var_256x512", **stats(times)})

    # prod_axis_256x512
    def run_prod(x=x_pos):
        c = jnp.prod(x, axis=1)
        c.block_until_ready()
    times = bench(run_prod, ITERS_FAST)
    results.append({"op": "prod_axis_256x512", **stats(times)})

    # logsumexp_256x512
    def run_logsumexp(x=x):
        c = jax.scipy.special.logsumexp(x, axis=1)
        c.block_until_ready()
    times = bench(run_logsumexp, ITERS_FAST)
    results.append({"op": "logsumexp_256x512", **stats(times)})

    # cumsum_256x512
    def run_cumsum(x=x):
        c = jnp.cumsum(x, axis=1)
        c.block_until_ready()
    times = bench(run_cumsum, ITERS_FAST)
    results.append({"op": "cumsum_256x512", **stats(times)})

    # argmax_axis_256x512
    def run_argmax(x=x):
        c = jnp.argmax(x, axis=1)
        c.block_until_ready()
    times = bench(run_argmax, ITERS_FAST)
    results.append({"op": "argmax_axis_256x512", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 1F: Shape ops
# ---------------------------------------------------------------------------


def bench_shape_ops():
    results = []
    key = jax.random.key(0)
    keys = jax.random.split(key, 4)

    # tril_256x256
    x256 = jax.random.normal(keys[0], (256, 256))
    def run_tril(x=x256):
        c = jnp.tril(x)
        c.block_until_ready()
    times = bench(run_tril, ITERS_FAST)
    results.append({"op": "tril_256x256", **stats(times)})

    # triu_256x256
    def run_triu(x=x256):
        c = jnp.triu(x)
        c.block_until_ready()
    times = bench(run_triu, ITERS_FAST)
    results.append({"op": "triu_256x256", **stats(times)})

    # repeat_64x128_2x3 (using jnp.tile)
    x_small = jax.random.normal(keys[1], (64, 128))
    def run_repeat(x=x_small):
        c = jnp.tile(x, (2, 3))
        c.block_until_ready()
    times = bench(run_repeat, ITERS_FAST)
    results.append({"op": "repeat_64x128_2x3", **stats(times)})

    # pad_64x128
    def run_pad(x=x_small):
        c = jnp.pad(x, ((1, 1), (2, 2)))
        c.block_until_ready()
    times = bench(run_pad, ITERS_FAST)
    results.append({"op": "pad_64x128", **stats(times)})

    # stack_8x64x128
    tensors = [jax.random.normal(jax.random.key(i + 10), (64, 128)) for i in range(8)]
    def run_stack(ts=tensors):
        c = jnp.stack(ts, axis=0)
        c.block_until_ready()
    times = bench(run_stack, ITERS_FAST)
    results.append({"op": "stack_8x64x128", **stats(times)})

    # diagonal_512x512
    x512 = jax.random.normal(keys[2], (512, 512))
    def run_diagonal(x=x512):
        c = jnp.diagonal(x)
        c.block_until_ready()
    times = bench(run_diagonal, ITERS_FAST)
    results.append({"op": "diagonal_512x512", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 2: Activations (100k elements)
# ---------------------------------------------------------------------------


def bench_activations():
    results = []
    n = 100_000
    a = jax.random.normal(jax.random.key(0), (n,))

    # silu_100k
    def run_silu(a=a):
        c = jax.nn.silu(a)
        c.block_until_ready()
    times = bench(run_silu, ITERS_FAST)
    results.append({"op": "silu_100k", **stats(times)})

    # softplus_100k
    def run_softplus(a=a):
        c = jax.nn.softplus(a)
        c.block_until_ready()
    times = bench(run_softplus, ITERS_FAST)
    results.append({"op": "softplus_100k", **stats(times)})

    # mish_100k: x * tanh(softplus(x))
    def run_mish(a=a):
        c = a * jnp.tanh(jax.nn.softplus(a))
        c.block_until_ready()
    times = bench(run_mish, ITERS_FAST)
    results.append({"op": "mish_100k", **stats(times)})

    # leaky_relu_100k
    def run_leaky_relu(a=a):
        c = jax.nn.leaky_relu(a)
        c.block_until_ready()
    times = bench(run_leaky_relu, ITERS_FAST)
    results.append({"op": "leaky_relu_100k", **stats(times)})

    # elu_100k
    def run_elu(a=a):
        c = jax.nn.elu(a)
        c.block_until_ready()
    times = bench(run_elu, ITERS_FAST)
    results.append({"op": "elu_100k", **stats(times)})

    # hard_tanh_100k: clip(a, -1, 1)
    def run_hard_tanh(a=a):
        c = jnp.clip(a, -1.0, 1.0)
        c.block_until_ready()
    times = bench(run_hard_tanh, ITERS_FAST)
    results.append({"op": "hard_tanh_100k", **stats(times)})

    # relu6_100k: clip(relu(a), 0, 6)
    def run_relu6(a=a):
        c = jnp.clip(jax.nn.relu(a), 0.0, 6.0)
        c.block_until_ready()
    times = bench(run_relu6, ITERS_FAST)
    results.append({"op": "relu6_100k", **stats(times)})

    # hardswish_100k
    def run_hardswish(a=a):
        c = jax.nn.hard_swish(a)
        c.block_until_ready()
    times = bench(run_hardswish, ITERS_FAST)
    results.append({"op": "hardswish_100k", **stats(times)})

    # gelu_100k
    def run_gelu(a=a):
        c = jax.nn.gelu(a)
        c.block_until_ready()
    times = bench(run_gelu, ITERS_FAST)
    results.append({"op": "gelu_100k", **stats(times)})

    # selu_100k
    def run_selu(a=a):
        c = jax.nn.selu(a)
        c.block_until_ready()
    times = bench(run_selu, ITERS_FAST)
    results.append({"op": "selu_100k", **stats(times)})

    # softsign_100k: x / (1 + |x|)
    def run_softsign(a=a):
        c = a / (1.0 + jnp.abs(a))
        c.block_until_ready()
    times = bench(run_softsign, ITERS_FAST)
    results.append({"op": "softsign_100k", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 3A: Loss functions (batch=64, classes=10)
# ---------------------------------------------------------------------------


def bench_losses():
    results = []
    batch = 64
    classes = 10

    key = jax.random.key(0)
    keys = jax.random.split(key, 4)

    pred = jax.random.normal(keys[0], (batch, classes))
    target = jax.random.normal(keys[1], (batch, classes))
    target_abs = jnp.abs(jax.random.normal(keys[2], (batch, classes))) + 1e-6
    targets_idx = jnp.arange(batch) % classes

    # cross_entropy_64x10 (manual softmax cross-entropy)
    def run_cross_entropy(pred=pred, targets_idx=targets_idx):
        one_hot = jax.nn.one_hot(targets_idx, classes)
        log_probs = jax.nn.log_softmax(pred, axis=-1)
        loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
        loss.block_until_ready()
    times = bench(run_cross_entropy, ITERS_FAST)
    results.append({"op": "cross_entropy_64x10", **stats(times)})

    # l1_loss_64x10
    def run_l1_loss(pred=pred, target=target):
        loss = jnp.mean(jnp.abs(pred - target))
        loss.block_until_ready()
    times = bench(run_l1_loss, ITERS_FAST)
    results.append({"op": "l1_loss_64x10", **stats(times)})

    # mse_loss_64x10
    def run_mse_loss(pred=pred, target=target):
        loss = jnp.mean((pred - target) ** 2)
        loss.block_until_ready()
    times = bench(run_mse_loss, ITERS_FAST)
    results.append({"op": "mse_loss_64x10", **stats(times)})

    # huber_loss_64x10 (manual: delta=1.0)
    def run_huber_loss(pred=pred, target=target):
        diff = jnp.abs(pred - target)
        loss = jnp.mean(jnp.where(diff <= 1.0, 0.5 * diff ** 2, diff - 0.5))
        loss.block_until_ready()
    times = bench(run_huber_loss, ITERS_FAST)
    results.append({"op": "huber_loss_64x10", **stats(times)})

    # smooth_l1_loss_64x10 (manual: beta=1.0)
    def run_smooth_l1_loss(pred=pred, target=target):
        diff = jnp.abs(pred - target)
        loss = jnp.mean(jnp.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5))
        loss.block_until_ready()
    times = bench(run_smooth_l1_loss, ITERS_FAST)
    results.append({"op": "smooth_l1_loss_64x10", **stats(times)})

    # kl_div_loss_64x10 (manual KL divergence)
    def run_kl_div_loss(pred=pred, target=target_abs):
        log_pred = jax.nn.log_softmax(pred, axis=-1)
        target_probs = target / jnp.sum(target, axis=-1, keepdims=True)
        loss = jnp.mean(jnp.sum(target_probs * (jnp.log(target_probs + 1e-8) - log_pred), axis=-1))
        loss.block_until_ready()
    times = bench(run_kl_div_loss, ITERS_FAST)
    results.append({"op": "kl_div_loss_64x10", **stats(times)})

    # cosine_sim_loss_64x64
    a_emb = jax.random.normal(keys[3], (batch, 64))
    b_emb = jax.random.normal(jax.random.key(99), (batch, 64))
    def run_cosine_sim_loss(a=a_emb, b=b_emb):
        a_norm = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        b_norm = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
        loss = jnp.mean(1.0 - jnp.sum(a_norm * b_norm, axis=-1))
        loss.block_until_ready()
    times = bench(run_cosine_sim_loss, ITERS_FAST)
    results.append({"op": "cosine_sim_loss_64x64", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 3B: NN layers
# ---------------------------------------------------------------------------


def bench_nn_layers():
    results = []
    key = jax.random.key(0)
    keys = jax.random.split(key, 6)

    # rmsnorm_64x512 (manual: x * rsqrt(mean(x^2) + eps))
    x_rms = jax.random.normal(keys[0], (64, 512))
    def run_rmsnorm(x=x_rms):
        ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
        c = x * jax.lax.rsqrt(ms + 1e-5)
        c.block_until_ready()
    times = bench(run_rmsnorm, ITERS_FAST)
    results.append({"op": "rmsnorm_64x512", **stats(times)})

    # conv1d_1x32x128_k3 (using jax.lax.conv_general_dilated)
    # JAX conv expects (batch, in_channels, spatial) = (1, 32, 128)
    # kernel shape (out_channels, in_channels, kernel_size) = (64, 32, 3)
    x_conv = jax.random.normal(keys[1], (1, 32, 128))
    w_conv = jax.random.normal(keys[2], (64, 32, 3))
    def run_conv1d(x=x_conv, w=w_conv):
        c = jax.lax.conv(x, w, window_strides=(1,), padding="VALID")
        c.block_until_ready()
    times = bench(run_conv1d, ITERS_FAST)
    results.append({"op": "conv1d_1x32x128_k3", **stats(times)})

    # avgpool2d_1x16x32x32 (manual average pooling with kernel_size=2, stride=2)
    x_pool = jax.random.normal(keys[3], (1, 16, 32, 32))
    def run_avgpool2d(x=x_pool):
        # Reshape then mean: (1, 16, 16, 2, 16, 2) -> mean over pool dims
        n, c, h, w = x.shape
        kh, kw = 2, 2
        x_reshaped = x.reshape(n, c, h // kh, kh, w // kw, kw)
        c_out = jnp.mean(x_reshaped, axis=(3, 5))
        c_out.block_until_ready()
    times = bench(run_avgpool2d, ITERS_FAST)
    results.append({"op": "avgpool2d_1x16x32x32", **stats(times)})

    # groupnorm_4x64x16x16 (manual group norm: 8 groups over 64 channels)
    x_gn = jax.random.normal(keys[4], (4, 64, 16, 16))
    num_groups = 8
    def run_groupnorm(x=x_gn):
        n, c, h, w = x.shape
        x_g = x.reshape(n, num_groups, c // num_groups, h, w)
        mean = jnp.mean(x_g, axis=(2, 3, 4), keepdims=True)
        var = jnp.var(x_g, axis=(2, 3, 4), keepdims=True)
        c_out = ((x_g - mean) / jnp.sqrt(var + 1e-5)).reshape(n, c, h, w)
        c_out.block_until_ready()
    times = bench(run_groupnorm, ITERS_FAST)
    results.append({"op": "groupnorm_4x64x16x16", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 5: Random
# ---------------------------------------------------------------------------


def bench_random():
    results = []

    # rand_uniform_100k
    def run_uniform_100k():
        c = jax.random.uniform(jax.random.key(0), (100_000,))
        c.block_until_ready()
    times = bench(run_uniform_100k, ITERS_FAST)
    results.append({"op": "rand_uniform_100k", **stats(times)})

    # rand_normal_100k
    def run_normal_100k():
        c = jax.random.normal(jax.random.key(0), (100_000,))
        c.block_until_ready()
    times = bench(run_normal_100k, ITERS_FAST)
    results.append({"op": "rand_normal_100k", **stats(times)})

    # rand_bernoulli_100k
    def run_bernoulli_100k():
        c = jax.random.bernoulli(jax.random.key(0), 0.5, (100_000,))
        c.block_until_ready()
    times = bench(run_bernoulli_100k, ITERS_FAST)
    results.append({"op": "rand_bernoulli_100k", **stats(times)})

    # rand_uniform_1M
    def run_uniform_1M():
        c = jax.random.uniform(jax.random.key(0), (1_000_000,))
        c.block_until_ready()
    times = bench(run_uniform_1M, ITERS_SLOW)
    results.append({"op": "rand_uniform_1M", **stats(times)})

    # rand_normal_1M
    def run_normal_1M():
        c = jax.random.normal(jax.random.key(0), (1_000_000,))
        c.block_until_ready()
    times = bench(run_normal_1M, ITERS_SLOW)
    results.append({"op": "rand_normal_1M", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 6: FFT
# ---------------------------------------------------------------------------


def bench_fft():
    results = []

    # rfft_1k, rfft_4k, rfft_16k
    for n in [1024, 4096, 16384]:
        x = jax.random.normal(jax.random.key(0), (n,))
        label = f"{n // 1000}k"

        def run_rfft(x=x):
            c = jnp.fft.rfft(x)
            c.block_until_ready()
        times = bench(run_rfft, ITERS_FAST)
        results.append({"op": f"rfft_{label}", **stats(times)})

    # fft_1k, fft_4k (complex input)
    for n in [1024, 4096]:
        x = jax.random.normal(jax.random.key(0), (n,)) + 1j * jax.random.normal(jax.random.key(1), (n,))
        label = f"{n // 1000}k"

        def run_fft(x=x):
            c = jnp.fft.fft(x)
            c.block_until_ready()
        times = bench(run_fft, ITERS_FAST)
        results.append({"op": f"fft_{label}", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 7: Linear algebra
# ---------------------------------------------------------------------------


def bench_linalg():
    results = []

    # norm_l2_1k
    x_norm = jax.random.normal(jax.random.key(0), (1000,))
    def run_norm(x=x_norm):
        c = jnp.linalg.norm(x, ord=2)
        c.block_until_ready()
    times = bench(run_norm, ITERS_FAST)
    results.append({"op": "norm_l2_1k", **stats(times)})

    for n in [64, 128, 256]:
        iters = ITERS_SLOW if n >= 256 else ITERS_FAST

        # Create positive-definite matrix: A_pd = B^T @ B + n*I
        key = jax.random.key(n)
        keys = jax.random.split(key, 3)
        b_mat = jax.random.normal(keys[0], (n, n))
        eye = jnp.eye(n)
        a_pd = b_mat.T @ b_mat + n * eye

        # General matrix for SVD, QR, det
        a_gen = jax.random.normal(keys[1], (n, n))

        # Symmetric matrix for eigh
        a_sym = b_mat.T @ b_mat

        # solve_NxN
        b_vec = jax.random.normal(keys[2], (n, 1))
        def run_solve(a=a_pd, b=b_vec):
            c = jnp.linalg.solve(a, b)
            c.block_until_ready()
        times = bench(run_solve, iters)
        results.append({"op": f"solve_{n}x{n}", **stats(times)})

        # inv_NxN
        def run_inv(a=a_pd):
            c = jnp.linalg.inv(a)
            c.block_until_ready()
        times = bench(run_inv, iters)
        results.append({"op": f"inv_{n}x{n}", **stats(times)})

        # cholesky_NxN
        def run_cholesky(a=a_pd):
            c = jnp.linalg.cholesky(a)
            c.block_until_ready()
        times = bench(run_cholesky, iters)
        results.append({"op": f"cholesky_{n}x{n}", **stats(times)})

        # svd_NxN
        def run_svd(a=a_gen):
            u, s, vt = jnp.linalg.svd(a)
            u.block_until_ready()
            s.block_until_ready()
            vt.block_until_ready()
        times = bench(run_svd, iters)
        results.append({"op": f"svd_{n}x{n}", **stats(times)})

        # qr_NxN
        def run_qr(a=a_gen):
            q, r = jnp.linalg.qr(a)
            q.block_until_ready()
            r.block_until_ready()
        times = bench(run_qr, iters)
        results.append({"op": f"qr_{n}x{n}", **stats(times)})

        # eigh_NxN
        def run_eigh(a=a_sym):
            w, v = jnp.linalg.eigh(a)
            w.block_until_ready()
            v.block_until_ready()
        times = bench(run_eigh, iters)
        results.append({"op": f"eigh_{n}x{n}", **stats(times)})

        # det_NxN
        def run_det(a=a_gen):
            c = jnp.linalg.det(a)
            c.block_until_ready()
        times = bench(run_det, iters)
        results.append({"op": f"det_{n}x{n}", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Pipeline ops (fused matmul+bias+gelu, add+layernorm)
# ---------------------------------------------------------------------------


def bench_pipeline_ops():
    results = []

    # matmul + bias + gelu at transformer FFN sizes
    for m, k, n, label in [
        (196, 768, 3072, "196x768x3072"),
        (196, 1024, 4096, "196x1024x4096"),
    ]:
        key = jax.random.key(42)
        keys = jax.random.split(key, 3)
        x = jax.random.normal(keys[0], (m, k))
        w = jax.random.normal(keys[1], (k, n))
        b = jax.random.normal(keys[2], (1, n))

        def run(x=x, w=w, b=b):
            h = jax.nn.gelu(x @ w + b)
            h.block_until_ready()

        times = bench(run, ITERS_SLOW)
        results.append({"op": f"matmul_bias_gelu_{label}", **stats(times)})

    # add + layernorm at transformer sizes
    for batch, dim, label in [
        (196, 768, "196x768"),
        (196, 1024, "196x1024"),
    ]:
        key = jax.random.key(0)
        keys = jax.random.split(key, 4)
        x = jax.random.normal(keys[0], (batch, dim))
        r = jax.random.normal(keys[1], (batch, dim))
        g = jnp.ones((dim,))
        b = jnp.zeros((dim,))

        def run(x=x, r=r, g=g, b=b):
            s = x + r
            mean = jnp.mean(s, axis=-1, keepdims=True)
            var = jnp.var(s, axis=-1, keepdims=True)
            out = (s - mean) / jnp.sqrt(var + 1e-5) * g + b
            out.block_until_ready()

        times = bench(run, ITERS_FAST)
        results.append({"op": f"add_layernorm_{label}", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for fn in [
        # Existing benchmarks
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
        # Phase 1B-D: Binary/clip/compare
        bench_binary_math,
        # Phase 1E: Axis reductions
        bench_axis_reductions,
        # Phase 1F: Shape ops
        bench_shape_ops,
        # Phase 2: Activations
        bench_activations,
        # Phase 3A: Loss functions
        bench_losses,
        # Phase 3B: NN layers
        bench_nn_layers,
        # Phase 5: Random
        bench_random,
        # Phase 6: FFT
        bench_fft,
        # Phase 7: Linear algebra
        bench_linalg,
        # Pipeline ops
        bench_pipeline_ops,
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
