#!/usr/bin/env python3
"""Wall-clock benchmark for TensorFlow operations (CPU mode).

Outputs JSON to target/bench_compare/tensorflow.json with timing stats
(median, std, min, max in microseconds) for each operation.
"""

import json
import os
import time
from pathlib import Path
from statistics import median, stdev

# Force CPU before importing TF
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

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
# Existing benchmarks
# ---------------------------------------------------------------------------


def bench_matmul():
    results = []
    for size in [128, 256, 512]:
        iters = ITERS_SLOW if size == 512 else ITERS_FAST
        a = tf.random.normal((size, size))
        b = tf.random.normal((size, size))
        times = bench(lambda a=a, b=b: tf.matmul(a, b), iters)
        results.append({"op": f"matmul_{size}x{size}", **stats(times)})
    return results


def bench_add():
    results = []
    for n in [100_000, 500_000]:
        a = tf.random.normal((n,))
        b = tf.random.normal((n,))
        times = bench(lambda a=a, b=b: tf.add(a, b), iters=ITERS_FAST)
        results.append({"op": f"add_{n // 1000}k", **stats(times)})
    return results


def bench_mul():
    results = []
    for n in [100_000, 500_000]:
        a = tf.random.normal((n,))
        b = tf.random.normal((n,))
        times = bench(lambda a=a, b=b: tf.multiply(a, b), iters=ITERS_FAST)
        results.append({"op": f"mul_{n // 1000}k", **stats(times)})
    return results


def bench_exp():
    results = []
    for n in [100_000, 500_000]:
        a = tf.random.normal((n,))
        times = bench(lambda a=a: tf.exp(a), iters=ITERS_FAST)
        results.append({"op": f"exp_{n // 1000}k", **stats(times)})
    return results


def bench_relu():
    a = tf.random.normal((100_000,))
    times = bench(lambda: tf.nn.relu(a), ITERS_FAST)
    return [{"op": "relu_100k", **stats(times)}]


def bench_softmax():
    results = []
    for seq in [128, 512]:
        x = tf.random.normal((8, seq))
        times = bench(lambda x=x: tf.nn.softmax(x, axis=-1), ITERS_FAST)
        results.append({"op": f"softmax_8x{seq}", **stats(times)})
    return results


def bench_mlp_forward():
    w1 = tf.random.normal((784, 128))
    b1 = tf.random.normal((1, 128))
    w2 = tf.random.normal((128, 64))
    b2 = tf.random.normal((1, 64))
    w3 = tf.random.normal((64, 10))
    b3 = tf.random.normal((1, 10))
    x = tf.random.normal((64, 784))

    def fwd():
        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        return h2 @ w3 + b3

    times = bench(fwd, ITERS_FAST)
    return [{"op": "mlp_fwd_64x784", **stats(times)}]


def bench_training_step():
    w1 = tf.Variable(tf.random.normal((784, 128)))
    b1 = tf.Variable(tf.zeros((1, 128)))
    w2 = tf.Variable(tf.random.normal((128, 64)))
    b2 = tf.Variable(tf.zeros((1, 64)))
    w3 = tf.Variable(tf.random.normal((64, 10)))
    b3 = tf.Variable(tf.zeros((1, 10)))

    params = [w1, b1, w2, b2, w3, b3]
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    targets = tf.constant(list(range(64)), dtype=tf.int32) % 10

    def step():
        x = tf.random.normal((64, 784))
        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            logits = h2 @ w3 + b3
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets, logits=logits
                )
            )
        grads = tape.gradient(loss, params)
        opt.apply_gradients(zip(grads, params))

    times = bench(step, ITERS_SLOW)
    return [{"op": "train_step_64", **stats(times)}]


# ---------------------------------------------------------------------------
# Phase 1A: Unary math (100k elements)
# ---------------------------------------------------------------------------


def bench_unary_math():
    results = []
    a = tf.random.normal((100_000,))
    a_pos = tf.abs(tf.random.normal((100_000,))) + 1e-5
    a_unit = tf.constant(np.linspace(-0.9, 0.9, 100_000).astype(np.float32))

    ops = [
        ("reciprocal_100k", lambda: tf.math.reciprocal(a_pos)),
        ("square_100k", lambda: tf.math.square(a)),
        ("rsqrt_100k", lambda: tf.math.rsqrt(a_pos)),
        ("floor_100k", lambda: tf.math.floor(a)),
        ("ceil_100k", lambda: tf.math.ceil(a)),
        ("round_100k", lambda: tf.math.round(a)),
        ("sign_100k", lambda: tf.math.sign(a)),
        ("expm1_100k", lambda: tf.math.expm1(a)),
        ("log2_100k", lambda: tf.math.log(a_pos) / tf.math.log(2.0)),
        ("log10_100k", lambda: tf.math.log(a_pos) / tf.math.log(10.0)),
        ("log1p_100k", lambda: tf.math.log1p(a_pos)),
        ("erf_100k", lambda: tf.math.erf(a)),
        ("sinh_100k", lambda: tf.math.sinh(a)),
        ("cosh_100k", lambda: tf.math.cosh(a)),
        ("arcsin_100k", lambda: tf.math.asin(a_unit)),
        ("arccos_100k", lambda: tf.math.acos(a_unit)),
        ("arctan_100k", lambda: tf.math.atan(a)),
        ("arcsinh_100k", lambda: tf.math.asinh(a)),
    ]

    for name, fn in ops:
        times = bench(fn, ITERS_FAST)
        results.append({"op": name, **stats(times)})
    return results


# ---------------------------------------------------------------------------
# Phase 1B-D: Binary, clip, compare (100k)
# ---------------------------------------------------------------------------


def bench_binary_clip_compare():
    results = []
    a = tf.random.normal((100_000,))
    b = tf.random.normal((100_000,))
    a_pos = tf.abs(tf.random.normal((100_000,))) + 1e-5
    b_pos = tf.abs(tf.random.normal((100_000,))) + 1e-5
    cond = tf.random.normal((100_000,)) > 0.0

    ops = [
        ("maximum_100k", lambda: tf.math.maximum(a, b)),
        ("minimum_100k", lambda: tf.math.minimum(a, b)),
        ("power_100k", lambda: tf.math.pow(a_pos, b_pos)),
        ("arctan2_100k", lambda: tf.math.atan2(a, b)),
        (
            "logaddexp_100k",
            lambda: tf.math.maximum(a, b)
            + tf.math.log1p(tf.math.exp(-tf.abs(a - b))),
        ),
        ("clip_100k", lambda: tf.clip_by_value(a, -0.5, 0.5)),
        ("where_100k", lambda: tf.where(cond, a, b)),
        ("greater_100k", lambda: tf.math.greater(a, b)),
        ("equal_100k", lambda: tf.math.equal(a, b)),
    ]

    for name, fn in ops:
        times = bench(fn, ITERS_FAST)
        results.append({"op": name, **stats(times)})
    return results


# ---------------------------------------------------------------------------
# Phase 1E: Axis reductions (256x512, axis=1)
# ---------------------------------------------------------------------------


def bench_reductions():
    results = []
    x = tf.random.normal((256, 512))
    x_small_pos = tf.abs(tf.random.normal((256, 512))) * 0.01 + 0.99  # near 1.0

    ops = [
        ("sum_axis_256x512", lambda: tf.reduce_sum(x, axis=1)),
        ("mean_axis_256x512", lambda: tf.reduce_mean(x, axis=1)),
        ("max_axis_256x512", lambda: tf.reduce_max(x, axis=1)),
        ("min_axis_256x512", lambda: tf.reduce_min(x, axis=1)),
        ("var_256x512", lambda: tf.math.reduce_variance(x, axis=1)),
        ("prod_axis_256x512", lambda: tf.reduce_prod(x_small_pos, axis=1)),
        ("logsumexp_256x512", lambda: tf.reduce_logsumexp(x, axis=1)),
        ("cumsum_256x512", lambda: tf.cumsum(x, axis=1)),
        ("argmax_axis_256x512", lambda: tf.argmax(x, axis=1)),
    ]

    for name, fn in ops:
        times = bench(fn, ITERS_FAST)
        results.append({"op": name, **stats(times)})
    return results


# ---------------------------------------------------------------------------
# Phase 1F: Shape ops
# ---------------------------------------------------------------------------


def bench_shape_ops():
    results = []

    x_256 = tf.random.normal((256, 256))
    x_64x128 = tf.random.normal((64, 128))
    x_512 = tf.random.normal((512, 512))
    xs_stack = [tf.random.normal((64, 128)) for _ in range(8)]

    ops = [
        ("tril_256x256", lambda: tf.linalg.band_part(x_256, -1, 0)),
        ("triu_256x256", lambda: tf.linalg.band_part(x_256, 0, -1)),
        ("repeat_64x128_2x3", lambda: tf.tile(x_64x128, [2, 3])),
        (
            "pad_64x128",
            lambda: tf.pad(x_64x128, [[1, 1], [2, 2]], mode="CONSTANT"),
        ),
        ("stack_8x64x128", lambda: tf.stack(xs_stack)),
        ("diagonal_512x512", lambda: tf.linalg.diag_part(x_512)),
    ]

    for name, fn in ops:
        times = bench(fn, ITERS_FAST)
        results.append({"op": name, **stats(times)})
    return results


# ---------------------------------------------------------------------------
# Phase 2: Activations (100k)
# ---------------------------------------------------------------------------


def bench_activations():
    results = []
    a = tf.random.normal((100_000,))

    ops = [
        ("silu_100k", lambda: tf.nn.silu(a)),
        ("softplus_100k", lambda: tf.math.softplus(a)),
        ("mish_100k", lambda: a * tf.math.tanh(tf.math.softplus(a))),
        ("leaky_relu_100k", lambda: tf.nn.leaky_relu(a, alpha=0.01)),
        ("elu_100k", lambda: tf.nn.elu(a)),
        ("hard_tanh_100k", lambda: tf.clip_by_value(a, -1.0, 1.0)),
        ("relu6_100k", lambda: tf.nn.relu6(a)),
        ("hardswish_100k", lambda: a * tf.nn.relu6(a + 3.0) / 6.0),
        ("gelu_100k", lambda: tf.nn.gelu(a)),
        ("selu_100k", lambda: tf.nn.selu(a)),
        ("softsign_100k", lambda: tf.nn.softsign(a)),
    ]

    for name, fn in ops:
        times = bench(fn, ITERS_FAST)
        results.append({"op": name, **stats(times)})
    return results


# ---------------------------------------------------------------------------
# Phase 3A: Loss functions (batch=64, classes=10)
# ---------------------------------------------------------------------------


def bench_losses():
    results = []

    logits = tf.random.normal((64, 10))
    targets_oh = tf.one_hot(tf.range(64) % 10, 10)
    preds = tf.nn.softmax(logits, axis=-1)
    preds_clamp = tf.clip_by_value(preds, 1e-7, 1.0)
    targets_float = tf.random.normal((64, 10))
    preds_float = tf.random.normal((64, 10))

    # Cosine similarity inputs
    a_cos = tf.random.normal((64, 64))
    b_cos = tf.random.normal((64, 64))

    ops = [
        (
            "cross_entropy_64x10",
            lambda: tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=targets_oh, logits=logits
                )
            ),
        ),
        (
            "l1_loss_64x10",
            lambda: tf.reduce_mean(tf.abs(preds_float - targets_float)),
        ),
        (
            "mse_loss_64x10",
            lambda: tf.reduce_mean(tf.square(preds_float - targets_float)),
        ),
        (
            "huber_loss_64x10",
            lambda: tf.reduce_mean(
                tf.keras.losses.huber(targets_float, preds_float)
            ),
        ),
        (
            "smooth_l1_loss_64x10",
            lambda: tf.reduce_mean(
                tf.keras.losses.huber(targets_float, preds_float, delta=1.0)
            ),
        ),
        (
            "kl_div_loss_64x10",
            lambda: tf.reduce_mean(
                tf.keras.losses.KLDivergence()(targets_oh, preds_clamp)
            ),
        ),
        (
            "cosine_sim_loss_64x64",
            lambda: tf.reduce_mean(
                tf.keras.losses.cosine_similarity(a_cos, b_cos, axis=-1)
            ),
        ),
    ]

    for name, fn in ops:
        times = bench(fn, ITERS_FAST)
        results.append({"op": name, **stats(times)})
    return results


# ---------------------------------------------------------------------------
# Phase 3B: NN layers
# ---------------------------------------------------------------------------


def bench_nn_layers():
    results = []

    # RMSNorm
    x_rms = tf.random.normal((64, 512))

    def rmsnorm_fn():
        return x_rms * tf.math.rsqrt(
            tf.reduce_mean(x_rms ** 2, axis=-1, keepdims=True) + 1e-5
        )

    times = bench(rmsnorm_fn, ITERS_FAST)
    results.append({"op": "rmsnorm_64x512", **stats(times)})

    # Conv1D
    conv1d_layer = tf.keras.layers.Conv1D(64, 3, padding="valid")
    x_conv1d = tf.random.normal((1, 128, 32))
    # Build the layer
    conv1d_layer(x_conv1d)

    times = bench(lambda: conv1d_layer(x_conv1d), ITERS_FAST)
    results.append({"op": "conv1d_1x32x128_k3", **stats(times)})

    # AvgPool2D
    avgpool_layer = tf.keras.layers.AveragePooling2D(pool_size=2)
    x_pool = tf.random.normal((1, 32, 32, 16))
    avgpool_layer(x_pool)

    times = bench(lambda: avgpool_layer(x_pool), ITERS_FAST)
    results.append({"op": "avgpool2d_1x16x32x32", **stats(times)})

    # GroupNorm (manual implementation)
    x_gn = tf.random.normal((4, 16, 16, 64))
    num_groups = 8
    gamma_gn = tf.ones((64,))
    beta_gn = tf.zeros((64,))

    def groupnorm_fn():
        # Reshape to (N, H, W, G, C//G)
        N, H, W, C = 4, 16, 16, 64
        G = num_groups
        x_reshaped = tf.reshape(x_gn, (N, H, W, G, C // G))
        mean = tf.reduce_mean(x_reshaped, axis=[1, 2, 4], keepdims=True)
        var = tf.math.reduce_variance(x_reshaped, axis=[1, 2, 4], keepdims=True)
        x_norm = (x_reshaped - mean) / tf.math.sqrt(var + 1e-5)
        x_norm = tf.reshape(x_norm, (N, H, W, C))
        return x_norm * gamma_gn + beta_gn

    times = bench(groupnorm_fn, ITERS_FAST)
    results.append({"op": "groupnorm_4x64x16x16", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 5: Random
# ---------------------------------------------------------------------------


def bench_random():
    results = []

    ops = [
        ("rand_uniform_100k", lambda: tf.random.uniform((100_000,))),
        ("rand_normal_100k", lambda: tf.random.normal((100_000,))),
        (
            "rand_bernoulli_100k",
            lambda: tf.cast(
                tf.random.uniform((100_000,)) < 0.5, tf.float32
            ),
        ),
        ("rand_uniform_1M", lambda: tf.random.uniform((1_000_000,))),
        ("rand_normal_1M", lambda: tf.random.normal((1_000_000,))),
    ]

    for name, fn in ops:
        times = bench(fn, ITERS_FAST)
        results.append({"op": name, **stats(times)})
    return results


# ---------------------------------------------------------------------------
# Phase 6: FFT
# ---------------------------------------------------------------------------


def bench_fft():
    results = []

    # RFFT
    for n in [1024, 4096, 16384]:
        label = f"{n // 1024}k" if n >= 1024 else str(n)
        x = tf.random.normal((n,))
        times = bench(lambda x=x: tf.signal.rfft(x), ITERS_FAST)
        results.append({"op": f"rfft_{label}", **stats(times)})

    # FFT (complex input)
    for n in [1024, 4096]:
        label = f"{n // 1024}k"
        x_real = tf.random.normal((n,))
        x_imag = tf.random.normal((n,))
        x_complex = tf.complex(x_real, x_imag)
        times = bench(lambda x=x_complex: tf.signal.fft(x), ITERS_FAST)
        results.append({"op": f"fft_{label}", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 7: Linear algebra
# ---------------------------------------------------------------------------


def bench_linalg():
    results = []

    # L2 norm
    x_1k = tf.random.normal((1000,))
    times = bench(lambda: tf.norm(x_1k), ITERS_FAST)
    results.append({"op": "norm_l2_1k", **stats(times)})

    for N in [64, 128, 256]:
        iters = ITERS_FAST if N <= 128 else ITERS_SLOW
        suffix = f"_{N}x{N}"

        # Make a positive-definite matrix for cholesky
        A_raw = tf.random.normal((N, N))
        A_sym = A_raw @ tf.transpose(A_raw) + tf.eye(N) * float(N)

        # General square matrix
        A = tf.random.normal((N, N))
        b = tf.random.normal((N, 1))

        # solve
        times = bench(
            lambda A=A_sym, b=b: tf.linalg.solve(A, b), iters
        )
        results.append({"op": f"solve{suffix}", **stats(times)})

        # inv
        times = bench(lambda A=A_sym: tf.linalg.inv(A), iters)
        results.append({"op": f"inv{suffix}", **stats(times)})

        # cholesky
        times = bench(lambda A=A_sym: tf.linalg.cholesky(A), iters)
        results.append({"op": f"cholesky{suffix}", **stats(times)})

        # svd
        times = bench(lambda A=A: tf.linalg.svd(A), iters)
        results.append({"op": f"svd{suffix}", **stats(times)})

        # qr
        times = bench(lambda A=A: tf.linalg.qr(A), iters)
        results.append({"op": f"qr{suffix}", **stats(times)})

        # eigh (needs symmetric matrix)
        times = bench(lambda A=A_sym: tf.linalg.eigh(A), iters)
        results.append({"op": f"eigh{suffix}", **stats(times)})

        # det
        times = bench(lambda A=A: tf.linalg.det(A), iters)
        results.append({"op": f"det{suffix}", **stats(times)})

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
        x = tf.random.normal((m, k))
        w = tf.random.normal((k, n))
        b = tf.random.normal((1, n))

        def run(x=x, w=w, b=b):
            h = tf.nn.gelu(x @ w + b)
            return h

        times = bench(run, ITERS_SLOW)
        results.append({"op": f"matmul_bias_gelu_{label}", **stats(times)})

    # add + layernorm at transformer sizes
    for batch, dim, label in [
        (196, 768, "196x768"),
        (196, 1024, "196x1024"),
    ]:
        x = tf.random.normal((batch, dim))
        r = tf.random.normal((batch, dim))

        # Use tf.keras.layers.LayerNormalization
        ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        ln.build((batch, dim))

        def run(x=x, r=r, ln=ln):
            out = ln(x + r)
            return out

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
        # Existing
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
        # Phase 1B-D: Binary, clip, compare
        bench_binary_clip_compare,
        # Phase 1E: Axis reductions
        bench_reductions,
        # Phase 1F: Shape ops
        bench_shape_ops,
        # Phase 2: Activations
        bench_activations,
        # Phase 3A: Losses
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
        print(f"  TensorFlow: {fn.__name__} ...")
        all_results.extend(fn())

    output = {
        "framework": "tensorflow",
        "tf_version": tf.__version__,
        "device": "cpu",
        "results": all_results,
    }

    out_path = OUT_DIR / "tensorflow.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
