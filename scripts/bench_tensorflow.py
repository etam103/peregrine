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
# Benchmarks
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
