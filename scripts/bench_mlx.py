#!/usr/bin/env python3
"""Wall-clock benchmark for MLX operations (CPU mode).

Outputs JSON to target/bench_compare/mlx.json with timing stats
(median, std, min, max in microseconds) for each operation.

Covers all Peregrine benchmark operations across phases 1-7.
"""

import json
import time
from pathlib import Path
from statistics import median, stdev

import numpy as np
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
# Existing benchmarks
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
# Phase 1A: Unary math ops (100k elements)
# ---------------------------------------------------------------------------


def bench_unary_math():
    results = []
    a = mx.random.normal((100_000,))
    a_pos = mx.abs(mx.random.normal((100_000,))) + 1e-5
    a_unit = mx.array(np.linspace(-0.9, 0.9, 100_000).astype(np.float32))
    mx.eval(a, a_pos, a_unit)

    # Each entry: (op_name, lambda, input_ref)
    unary_ops = [
        ("reciprocal_100k", lambda x: mx.reciprocal(x), a_pos),
        ("square_100k", lambda x: mx.square(x), a),
        ("rsqrt_100k", lambda x: mx.rsqrt(x), a_pos),
        ("floor_100k", lambda x: mx.floor(x), a),
        ("ceil_100k", lambda x: mx.ceil(x), a),
        ("round_100k", lambda x: mx.round(x), a),
        ("sign_100k", lambda x: mx.sign(x), a),
        ("expm1_100k", lambda x: mx.expm1(x), a),
        ("log2_100k", lambda x: mx.log2(x), a_pos),
        ("log10_100k", lambda x: mx.log10(x), a_pos),
        ("log1p_100k", lambda x: mx.log1p(x), a_pos),
        ("erf_100k", lambda x: mx.erf(x), a),
        ("sinh_100k", lambda x: mx.sinh(x), a),
        ("cosh_100k", lambda x: mx.cosh(x), a),
        ("arcsin_100k", lambda x: mx.arcsin(x), a_unit),
        ("arccos_100k", lambda x: mx.arccos(x), a_unit),
        ("arctan_100k", lambda x: mx.arctan(x), a),
        ("arcsinh_100k", lambda x: mx.arcsinh(x), a),
    ]

    for op_name, fn, inp in unary_ops:
        try:
            def run(fn=fn, inp=inp):
                c = fn(inp)
                mx.eval(c)

            times = bench(run, ITERS_FAST)
            results.append({"op": op_name, **stats(times)})
        except Exception as e:
            print(f"    SKIP {op_name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Phase 1B-D: Binary, clip, compare ops (100k elements)
# ---------------------------------------------------------------------------


def bench_binary_ops():
    results = []
    a = mx.random.normal((100_000,))
    b = mx.random.normal((100_000,))
    a_pos = mx.abs(mx.random.normal((100_000,))) + 1e-5
    b_pos = mx.abs(mx.random.normal((100_000,))) + 0.5
    cond = mx.random.bernoulli(p=0.5, shape=(100_000,))
    mx.eval(a, b, a_pos, b_pos, cond)

    binary_ops = [
        ("maximum_100k", lambda: mx.maximum(a, b)),
        ("minimum_100k", lambda: mx.minimum(a, b)),
        ("power_100k", lambda: mx.power(a_pos, b_pos)),
        ("arctan2_100k", lambda: mx.arctan2(a, b)),
        ("logaddexp_100k", lambda: mx.logaddexp(a, b)),
        ("clip_100k", lambda: mx.clip(a, -0.5, 0.5)),
        ("where_100k", lambda: mx.where(cond, a, b)),
        ("greater_100k", lambda: mx.greater(a, b)),
        ("equal_100k", lambda: mx.equal(a, b)),
    ]

    for op_name, fn in binary_ops:
        try:
            def run(fn=fn):
                c = fn()
                mx.eval(c)

            times = bench(run, ITERS_FAST)
            results.append({"op": op_name, **stats(times)})
        except Exception as e:
            print(f"    SKIP {op_name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Phase 1E: Axis reductions (256x512, axis=1)
# ---------------------------------------------------------------------------


def bench_reductions():
    results = []
    x = mx.random.normal((256, 512))
    mx.eval(x)

    reduction_ops = [
        ("sum_axis_256x512", lambda: mx.sum(x, axis=1)),
        ("mean_axis_256x512", lambda: mx.mean(x, axis=1)),
        ("max_axis_256x512", lambda: mx.max(x, axis=1)),
        ("min_axis_256x512", lambda: mx.min(x, axis=1)),
        ("var_256x512", lambda: mx.var(x, axis=1)),
        ("prod_axis_256x512", lambda: mx.prod(x, axis=1)),
        ("logsumexp_256x512", lambda: mx.logsumexp(x, axis=1)),
        ("cumsum_256x512", lambda: mx.cumsum(x, axis=1)),
        ("argmax_axis_256x512", lambda: mx.argmax(x, axis=1)),
    ]

    for op_name, fn in reduction_ops:
        try:
            def run(fn=fn):
                c = fn()
                mx.eval(c)

            times = bench(run, ITERS_FAST)
            results.append({"op": op_name, **stats(times)})
        except Exception as e:
            print(f"    SKIP {op_name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Phase 1F: Shape ops
# ---------------------------------------------------------------------------


def bench_shape_ops():
    results = []

    # tril / triu on 256x256
    sq = mx.random.normal((256, 256))
    mx.eval(sq)

    shape_ops_list = []

    shape_ops_list.append(("tril_256x256", lambda: mx.tril(sq)))
    shape_ops_list.append(("triu_256x256", lambda: mx.triu(sq)))

    # repeat: tile a 64x128 tensor 2x along dim0 and 3x along dim1
    tile_input = mx.random.normal((64, 128))
    mx.eval(tile_input)
    try:
        # mx.tile is the standard way
        shape_ops_list.append(("repeat_64x128_2x3", lambda: mx.tile(tile_input, (2, 3))))
    except AttributeError:
        # fallback: use mx.repeat if tile not available
        try:
            shape_ops_list.append(("repeat_64x128_2x3", lambda: mx.repeat(tile_input, 2, axis=0)))
        except Exception:
            pass

    # pad 64x128
    pad_input = mx.random.normal((64, 128))
    mx.eval(pad_input)
    shape_ops_list.append(("pad_64x128", lambda: mx.pad(pad_input, ((1, 1), (2, 2)))))

    # stack 8 tensors of 64x128
    stack_inputs = [mx.random.normal((64, 128)) for _ in range(8)]
    mx.eval(*stack_inputs)
    shape_ops_list.append(("stack_8x64x128", lambda: mx.stack(stack_inputs, axis=0)))

    # diagonal of 512x512
    diag_input = mx.random.normal((512, 512))
    mx.eval(diag_input)
    try:
        shape_ops_list.append(("diagonal_512x512", lambda: mx.diagonal(diag_input)))
    except AttributeError:
        # Older MLX versions may use mx.diag
        try:
            shape_ops_list.append(("diagonal_512x512", lambda: mx.diag(diag_input)))
        except Exception:
            pass

    for op_name, fn in shape_ops_list:
        try:
            def run(fn=fn):
                c = fn()
                mx.eval(c)

            times = bench(run, ITERS_FAST)
            results.append({"op": op_name, **stats(times)})
        except Exception as e:
            print(f"    SKIP {op_name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Phase 2: Activations (100k elements)
# ---------------------------------------------------------------------------


def bench_activations():
    results = []
    a = mx.random.normal((100_000,))
    mx.eval(a)

    activation_ops = [
        ("silu_100k", lambda x: x * mx.sigmoid(x)),
        ("softplus_100k", lambda x: mx.logaddexp(mx.zeros_like(x), x)),
        ("mish_100k", lambda x: x * mx.tanh(mx.logaddexp(mx.zeros_like(x), x))),
        ("leaky_relu_100k", lambda x: mx.where(x > 0, x, 0.01 * x)),
        ("elu_100k", lambda x: mx.where(x > 0, x, mx.exp(x) - 1)),
        ("hard_tanh_100k", lambda x: mx.clip(x, -1.0, 1.0)),
        ("relu6_100k", lambda x: mx.clip(mx.maximum(x, 0), 0, 6)),
        ("hardswish_100k", lambda x: x * mx.clip(x + 3, 0, 6) / 6),
        ("gelu_100k", lambda x: nn.gelu(x)),
        ("selu_100k", lambda x: nn.selu(x)),
        ("softsign_100k", lambda x: x / (1 + mx.abs(x))),
    ]

    # Try to use mlx.nn versions where available, fallback to manual
    for op_name, fn in activation_ops:
        try:
            def run(fn=fn, a=a):
                c = fn(a)
                mx.eval(c)

            times = bench(run, ITERS_FAST)
            results.append({"op": op_name, **stats(times)})
        except Exception as e:
            print(f"    SKIP {op_name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Phase 3A: Loss functions (batch=64, classes=10)
# ---------------------------------------------------------------------------


def bench_losses():
    results = []

    # Inputs for classification losses
    pred = mx.random.normal((64, 10))
    target = mx.random.normal((64, 10))
    target_int = mx.array(np.random.randint(0, 10, size=(64,)).astype(np.int32))
    target_abs = mx.abs(mx.random.normal((64, 10))) + 1e-5
    # Normalize target_abs to be a proper distribution for KL div
    target_abs_sum = mx.sum(target_abs, axis=1, keepdims=True)
    target_prob = target_abs / target_abs_sum

    # Inputs for cosine similarity
    cos_a = mx.random.normal((64, 64))
    cos_b = mx.random.normal((64, 64))
    mx.eval(pred, target, target_int, target_prob, cos_a, cos_b)

    # cross_entropy
    try:
        def run_ce():
            c = nn.losses.cross_entropy(pred, target_int).mean()
            mx.eval(c)

        times = bench(run_ce, ITERS_FAST)
        results.append({"op": "cross_entropy_64x10", **stats(times)})
    except Exception as e:
        print(f"    SKIP cross_entropy_64x10: {e}")

    # l1_loss
    try:
        def run_l1():
            c = mx.mean(mx.abs(pred - target))
            mx.eval(c)

        times = bench(run_l1, ITERS_FAST)
        results.append({"op": "l1_loss_64x10", **stats(times)})
    except Exception as e:
        print(f"    SKIP l1_loss_64x10: {e}")

    # mse_loss
    try:
        def run_mse():
            c = mx.mean((pred - target) ** 2)
            mx.eval(c)

        times = bench(run_mse, ITERS_FAST)
        results.append({"op": "mse_loss_64x10", **stats(times)})
    except Exception as e:
        print(f"    SKIP mse_loss_64x10: {e}")

    # huber_loss (delta=1.0)
    try:
        def run_huber():
            diff = pred - target
            abs_diff = mx.abs(diff)
            delta = 1.0
            c = mx.mean(mx.where(abs_diff <= delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta)))
            mx.eval(c)

        times = bench(run_huber, ITERS_FAST)
        results.append({"op": "huber_loss_64x10", **stats(times)})
    except Exception as e:
        print(f"    SKIP huber_loss_64x10: {e}")

    # smooth_l1_loss (beta=1.0)
    try:
        def run_smooth_l1():
            diff = pred - target
            abs_diff = mx.abs(diff)
            beta = 1.0
            c = mx.mean(mx.where(abs_diff < beta, 0.5 * diff ** 2 / beta, abs_diff - 0.5 * beta))
            mx.eval(c)

        times = bench(run_smooth_l1, ITERS_FAST)
        results.append({"op": "smooth_l1_loss_64x10", **stats(times)})
    except Exception as e:
        print(f"    SKIP smooth_l1_loss_64x10: {e}")

    # kl_div_loss
    try:
        def run_kl():
            c = mx.mean(target_prob * (mx.log(target_prob) - pred))
            mx.eval(c)

        times = bench(run_kl, ITERS_FAST)
        results.append({"op": "kl_div_loss_64x10", **stats(times)})
    except Exception as e:
        print(f"    SKIP kl_div_loss_64x10: {e}")

    # cosine_similarity_loss (64x64 vectors)
    try:
        def run_cosine():
            dot = mx.sum(cos_a * cos_b, axis=1)
            norm_a = mx.sqrt(mx.sum(cos_a ** 2, axis=1))
            norm_b = mx.sqrt(mx.sum(cos_b ** 2, axis=1))
            c = mx.mean(dot / (norm_a * norm_b))
            mx.eval(c)

        times = bench(run_cosine, ITERS_FAST)
        results.append({"op": "cosine_sim_loss_64x64", **stats(times)})
    except Exception as e:
        print(f"    SKIP cosine_sim_loss_64x64: {e}")

    return results


# ---------------------------------------------------------------------------
# Phase 3B: NN layers
# ---------------------------------------------------------------------------


def bench_nn_layers():
    results = []

    # RMSNorm 64x512
    try:
        rms = nn.RMSNorm(512)
        x_rms = mx.random.normal((64, 512))
        mx.eval(x_rms)

        def run_rmsnorm():
            c = rms(x_rms)
            mx.eval(c)

        times = bench(run_rmsnorm, ITERS_FAST)
        results.append({"op": "rmsnorm_64x512", **stats(times)})
    except Exception as e:
        print(f"    SKIP rmsnorm_64x512: {e}")

    # Conv1d: input (1, 128, 32) -> Conv1d(32, 64, kernel_size=3)
    try:
        conv1d = nn.Conv1d(32, 64, 3)
        x_conv1d = mx.random.normal((1, 128, 32))
        mx.eval(x_conv1d)

        def run_conv1d():
            c = conv1d(x_conv1d)
            mx.eval(c)

        times = bench(run_conv1d, ITERS_FAST)
        results.append({"op": "conv1d_1x32x128_k3", **stats(times)})
    except Exception as e:
        print(f"    SKIP conv1d_1x32x128_k3: {e}")

    # AvgPool2d: input (1, 32, 32, 16) - manual mean pooling with 2x2 windows
    try:
        x_pool = mx.random.normal((1, 16, 32, 32))
        mx.eval(x_pool)

        def run_avgpool():
            # Reshape to (1, 16, 16, 2, 16, 2) then mean over pool dims
            b, c_in, h, w = x_pool.shape
            reshaped = mx.reshape(x_pool, (b, c_in, h // 2, 2, w // 2, 2))
            c = mx.mean(reshaped, axis=(3, 5))
            mx.eval(c)

        times = bench(run_avgpool, ITERS_FAST)
        results.append({"op": "avgpool2d_1x16x32x32", **stats(times)})
    except Exception as e:
        print(f"    SKIP avgpool2d_1x16x32x32: {e}")

    # GroupNorm: input (4, 16, 16, 64) with 8 groups
    try:
        gn = nn.GroupNorm(8, 64)
        # MLX GroupNorm expects (N, ..., C) layout
        x_gn = mx.random.normal((4, 16, 16, 64))
        mx.eval(x_gn)

        def run_groupnorm():
            c = gn(x_gn)
            mx.eval(c)

        times = bench(run_groupnorm, ITERS_FAST)
        results.append({"op": "groupnorm_4x64x16x16", **stats(times)})
    except Exception as e:
        print(f"    SKIP groupnorm_4x64x16x16: {e}")

    return results


# ---------------------------------------------------------------------------
# Phase 5: Random number generation
# ---------------------------------------------------------------------------


def bench_random():
    results = []

    random_ops = [
        ("rand_uniform_100k", 100_000, lambda n: mx.random.uniform(shape=(n,))),
        ("rand_normal_100k", 100_000, lambda n: mx.random.normal((n,))),
        ("rand_bernoulli_100k", 100_000, lambda n: mx.random.bernoulli(p=0.5, shape=(n,))),
        ("rand_uniform_1M", 1_000_000, lambda n: mx.random.uniform(shape=(n,))),
        ("rand_normal_1M", 1_000_000, lambda n: mx.random.normal((n,))),
    ]

    for op_name, n, fn in random_ops:
        try:
            def run(fn=fn, n=n):
                c = fn(n)
                mx.eval(c)

            times = bench(run, ITERS_FAST)
            results.append({"op": op_name, **stats(times)})
        except Exception as e:
            print(f"    SKIP {op_name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Phase 6: FFT operations
# ---------------------------------------------------------------------------


def bench_fft():
    results = []

    # rfft at various sizes
    for size in [1024, 4096, 16384]:
        label = f"rfft_{size // 1000}k" if size >= 1000 else f"rfft_{size}"
        x = mx.random.normal((size,))
        mx.eval(x)

        try:
            def run(x=x):
                c = mx.fft.rfft(x)
                mx.eval(c)

            times = bench(run, ITERS_FAST)
            results.append({"op": label, **stats(times)})
        except Exception as e:
            print(f"    SKIP {label}: {e}")

    # fft at various sizes
    for size in [1024, 4096]:
        label = f"fft_{size // 1000}k" if size >= 1000 else f"fft_{size}"
        # fft needs complex or real input
        x = mx.random.normal((size,))
        mx.eval(x)

        try:
            def run(x=x):
                c = mx.fft.fft(x)
                mx.eval(c)

            times = bench(run, ITERS_FAST)
            results.append({"op": label, **stats(times)})
        except Exception as e:
            print(f"    SKIP {label}: {e}")

    return results


# ---------------------------------------------------------------------------
# Phase 7: Linear algebra
# ---------------------------------------------------------------------------


def bench_linalg():
    results = []

    # L2 norm of a 1k vector
    try:
        x_norm = mx.random.normal((1000,))
        mx.eval(x_norm)

        def run_norm():
            c = mx.linalg.norm(x_norm)
            mx.eval(c)

        times = bench(run_norm, ITERS_FAST)
        results.append({"op": "norm_l2_1k", **stats(times)})
    except Exception as e:
        print(f"    SKIP norm_l2_1k: {e}")

    # Matrix operations at various sizes
    for n in [64, 128, 256]:
        iters = ITERS_FAST if n <= 128 else ITERS_SLOW

        # Create a random matrix and a positive-definite matrix
        b_mat = mx.random.normal((n, n))
        mx.eval(b_mat)
        # A = B^T @ B + n*I for positive definiteness
        pd_mat = b_mat.T @ b_mat + n * mx.eye(n)
        rhs = mx.random.normal((n, n))
        mx.eval(pd_mat, rhs)

        # solve
        try:
            def run_solve(pd=pd_mat, r=rhs):
                c = mx.linalg.solve(pd, r)
                mx.eval(c)

            times = bench(run_solve, iters)
            results.append({"op": f"solve_{n}x{n}", **stats(times)})
        except Exception as e:
            print(f"    SKIP solve_{n}x{n}: {e}")

        # inv
        try:
            def run_inv(pd=pd_mat):
                c = mx.linalg.inv(pd)
                mx.eval(c)

            times = bench(run_inv, iters)
            results.append({"op": f"inv_{n}x{n}", **stats(times)})
        except Exception as e:
            print(f"    SKIP inv_{n}x{n}: {e}")

        # cholesky
        try:
            def run_chol(pd=pd_mat):
                c = mx.linalg.cholesky(pd)
                mx.eval(c)

            times = bench(run_chol, iters)
            results.append({"op": f"cholesky_{n}x{n}", **stats(times)})
        except Exception as e:
            print(f"    SKIP cholesky_{n}x{n}: {e}")

        # svd
        try:
            def run_svd(m=b_mat):
                u, s, vt = mx.linalg.svd(m)
                mx.eval(u, s, vt)

            times = bench(run_svd, iters)
            results.append({"op": f"svd_{n}x{n}", **stats(times)})
        except Exception as e:
            print(f"    SKIP svd_{n}x{n}: {e}")

        # qr
        try:
            def run_qr(m=b_mat):
                q, r = mx.linalg.qr(m)
                mx.eval(q, r)

            times = bench(run_qr, iters)
            results.append({"op": f"qr_{n}x{n}", **stats(times)})
        except Exception as e:
            print(f"    SKIP qr_{n}x{n}: {e}")

        # eigh (eigenvalues of symmetric/hermitian matrix)
        try:
            def run_eigh(pd=pd_mat):
                eigvals, eigvecs = mx.linalg.eigh(pd)
                mx.eval(eigvals, eigvecs)

            times = bench(run_eigh, iters)
            results.append({"op": f"eigh_{n}x{n}", **stats(times)})
        except Exception as e:
            print(f"    SKIP eigh_{n}x{n}: {e}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    bench_fns = [
        # Existing benchmarks
        ("bench_matmul", bench_matmul),
        ("bench_add", bench_add),
        ("bench_mul", bench_mul),
        ("bench_exp", bench_exp),
        ("bench_relu", bench_relu),
        ("bench_softmax", bench_softmax),
        ("bench_mlp_forward", bench_mlp_forward),
        ("bench_training_step", bench_training_step),
        # Phase 1A: Unary math
        ("bench_unary_math", bench_unary_math),
        # Phase 1B-D: Binary/clip/compare
        ("bench_binary_ops", bench_binary_ops),
        # Phase 1E: Axis reductions
        ("bench_reductions", bench_reductions),
        # Phase 1F: Shape ops
        ("bench_shape_ops", bench_shape_ops),
        # Phase 2: Activations
        ("bench_activations", bench_activations),
        # Phase 3A: Loss functions
        ("bench_losses", bench_losses),
        # Phase 3B: NN layers
        ("bench_nn_layers", bench_nn_layers),
        # Phase 5: Random
        ("bench_random", bench_random),
        # Phase 6: FFT
        ("bench_fft", bench_fft),
        # Phase 7: Linear algebra
        ("bench_linalg", bench_linalg),
    ]

    for name, fn in bench_fns:
        print(f"  MLX: {name} ...")
        try:
            all_results.extend(fn())
        except Exception as e:
            print(f"  ERROR in {name}: {e}")

    output = {
        "framework": "mlx",
        "mlx_version": mx.__version__ if hasattr(mx, "__version__") else "unknown",
        "device": "cpu",
        "results": all_results,
    }

    out_path = OUT_DIR / "mlx.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved {out_path}  ({len(all_results)} benchmarks)")


if __name__ == "__main__":
    main()
