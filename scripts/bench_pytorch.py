#!/usr/bin/env python3
"""Wall-clock benchmark for PyTorch operations.

Outputs JSON to target/bench_compare/pytorch.json with timing stats
(median, std, min, max in microseconds) for each operation.

Covers ALL operations benchmarked in Peregrine's wallclock.rs.
"""

import json
import os
import time
from pathlib import Path
from statistics import median, stdev

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP = 5
ITERS_FAST = 50
ITERS_SLOW = 20  # matmul 512, training step, linalg 256x256

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
# Existing benchmarks
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
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        opt.step()

    times = bench(step, ITERS_SLOW)
    return [{"op": "train_step_64", **stats(times)}]


# ---------------------------------------------------------------------------
# Phase 1A: Unary math ops (100k elements)
# ---------------------------------------------------------------------------


def bench_unary_math():
    results = []
    n = 100_000

    a = torch.randn(n)
    a_pos = torch.abs(torch.randn(n)) + 1e-5
    a_unit = torch.linspace(-0.9, 0.9, n)

    ops = [
        ("reciprocal_100k", lambda: torch.reciprocal(a_pos)),
        ("square_100k",     lambda: torch.square(a)),
        ("rsqrt_100k",      lambda: torch.rsqrt(a_pos)),
        ("floor_100k",      lambda: torch.floor(a)),
        ("ceil_100k",       lambda: torch.ceil(a)),
        ("round_100k",      lambda: torch.round(a)),
        ("sign_100k",       lambda: torch.sign(a)),
        ("expm1_100k",      lambda: torch.expm1(a)),
        ("log2_100k",       lambda: torch.log2(a_pos)),
        ("log10_100k",      lambda: torch.log10(a_pos)),
        ("log1p_100k",      lambda: torch.log1p(a_pos)),
        ("erf_100k",        lambda: torch.erf(a)),
        ("sinh_100k",       lambda: torch.sinh(a)),
        ("cosh_100k",       lambda: torch.cosh(a)),
        ("arcsin_100k",     lambda: torch.arcsin(a_unit)),
        ("arccos_100k",     lambda: torch.arccos(a_unit)),
        ("arctan_100k",     lambda: torch.arctan(a)),
        ("arcsinh_100k",    lambda: torch.arcsinh(a)),
    ]

    for name, fn in ops:
        times = bench(fn, ITERS_FAST)
        results.append({"op": name, **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 1B-D: Binary math / clip / compare (100k elements)
# ---------------------------------------------------------------------------


def bench_binary_math():
    results = []
    n = 100_000

    a = torch.randn(n)
    b = torch.randn(n)
    a_pos = torch.abs(torch.randn(n)) + 1e-5
    b_pos = torch.abs(torch.randn(n)) + 1e-5
    cond = torch.randn(n) > 0  # boolean mask for where

    ops = [
        ("maximum_100k",   lambda: torch.maximum(a, b)),
        ("minimum_100k",   lambda: torch.minimum(a, b)),
        ("power_100k",     lambda: torch.pow(a_pos, b_pos)),
        ("arctan2_100k",   lambda: torch.arctan2(a, b)),
        ("logaddexp_100k", lambda: torch.logaddexp(a, b)),
        ("clip_100k",      lambda: torch.clip(a, min=-0.5, max=0.5)),
        ("where_100k",     lambda: torch.where(cond, a, b)),
        ("greater_100k",   lambda: torch.greater(a, b)),
        ("equal_100k",     lambda: torch.equal(a, b)),
    ]

    for name, fn in ops:
        times = bench(fn, ITERS_FAST)
        results.append({"op": name, **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 1E: Axis reductions (256x512, reduce along axis=1)
# ---------------------------------------------------------------------------


def bench_axis_reductions():
    results = []

    x = torch.randn(256, 512)
    # Positive small values for prod to avoid overflow
    x_pos_small = torch.abs(torch.randn(256, 512)) * 0.0001 + 1e-4

    ops = [
        ("sum_axis_256x512",     lambda: torch.sum(x, dim=1)),
        ("mean_axis_256x512",    lambda: torch.mean(x, dim=1)),
        ("max_axis_256x512",     lambda: torch.max(x, dim=1)),
        ("min_axis_256x512",     lambda: torch.min(x, dim=1)),
        ("var_256x512",          lambda: torch.var(x, dim=1)),
        ("prod_axis_256x512",    lambda: torch.prod(x_pos_small, dim=1)),
        ("logsumexp_256x512",    lambda: torch.logsumexp(x, dim=1)),
        ("cumsum_256x512",       lambda: torch.cumsum(x, dim=1)),
        ("argmax_axis_256x512",  lambda: torch.argmax(x, dim=1)),
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

    # tril / triu on 256x256
    x_sq = torch.randn(256, 256)
    times = bench(lambda: torch.tril(x_sq), ITERS_FAST)
    results.append({"op": "tril_256x256", **stats(times)})

    times = bench(lambda: torch.triu(x_sq), ITERS_FAST)
    results.append({"op": "triu_256x256", **stats(times)})

    # repeat 64x128 by (2, 3)
    x_small = torch.randn(64, 128)
    times = bench(lambda: x_small.repeat(2, 3), ITERS_FAST)
    results.append({"op": "repeat_64x128_2x3", **stats(times)})

    # pad 64x128 with (2,2,1,1) -> left=2,right=2,top=1,bottom=1
    times = bench(lambda: F.pad(x_small, (2, 2, 1, 1)), ITERS_FAST)
    results.append({"op": "pad_64x128", **stats(times)})

    # stack 8 tensors of [64, 128]
    tensors = [torch.randn(64, 128) for _ in range(8)]
    times = bench(lambda: torch.stack(tensors, dim=0), ITERS_FAST)
    results.append({"op": "stack_8x64x128", **stats(times)})

    # diagonal of 512x512
    x_diag = torch.randn(512, 512)
    times = bench(lambda: torch.diagonal(x_diag), ITERS_FAST)
    results.append({"op": "diagonal_512x512", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 2: Activations (100k elements)
# ---------------------------------------------------------------------------


def bench_activations():
    results = []
    n = 100_000
    a = torch.randn(n)

    ops = [
        ("silu_100k",       lambda: F.silu(a)),
        ("softplus_100k",   lambda: F.softplus(a)),
        ("mish_100k",       lambda: F.mish(a)),
        ("leaky_relu_100k", lambda: F.leaky_relu(a, negative_slope=0.01)),
        ("elu_100k",        lambda: F.elu(a, alpha=1.0)),
        ("hard_tanh_100k",  lambda: F.hardtanh(a, min_val=-1.0, max_val=1.0)),
        ("relu6_100k",      lambda: F.relu6(a)),
        ("hardswish_100k",  lambda: F.hardswish(a)),
        ("gelu_100k",       lambda: F.gelu(a)),
        ("selu_100k",       lambda: F.selu(a)),
        ("softsign_100k",   lambda: F.softsign(a)),
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
    batch = 64
    classes = 10

    pred = torch.randn(batch, classes)
    target = torch.randn(batch, classes)
    target_abs = torch.abs(torch.randn(batch, classes)) + 1e-5
    targets_idx = torch.arange(batch) % classes

    # Cross-entropy
    times = bench(lambda: F.cross_entropy(pred, targets_idx), ITERS_FAST)
    results.append({"op": "cross_entropy_64x10", **stats(times)})

    # L1 loss
    times = bench(lambda: F.l1_loss(pred, target), ITERS_FAST)
    results.append({"op": "l1_loss_64x10", **stats(times)})

    # MSE loss
    times = bench(lambda: F.mse_loss(pred, target), ITERS_FAST)
    results.append({"op": "mse_loss_64x10", **stats(times)})

    # Huber loss
    times = bench(lambda: F.huber_loss(pred, target, delta=1.0), ITERS_FAST)
    results.append({"op": "huber_loss_64x10", **stats(times)})

    # Smooth L1 loss
    times = bench(lambda: F.smooth_l1_loss(pred, target, beta=1.0), ITERS_FAST)
    results.append({"op": "smooth_l1_loss_64x10", **stats(times)})

    # KL divergence (log input, positive target)
    pred_log = F.log_softmax(pred, dim=-1)
    times = bench(lambda: F.kl_div(pred_log, target_abs, reduction="batchmean"), ITERS_FAST)
    results.append({"op": "kl_div_loss_64x10", **stats(times)})

    # Cosine similarity loss
    a_emb = torch.randn(batch, 64)
    b_emb = torch.randn(batch, 64)
    times = bench(lambda: F.cosine_similarity(a_emb, b_emb, dim=1).mean(), ITERS_FAST)
    results.append({"op": "cosine_sim_loss_64x64", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 3B: NN layers
# ---------------------------------------------------------------------------


def bench_nn_layers():
    results = []

    # RMSNorm: x is [64, 512]
    rmsnorm = torch.nn.RMSNorm(512)
    x_rms = torch.randn(64, 512)
    times = bench(lambda: rmsnorm(x_rms), ITERS_FAST)
    results.append({"op": "rmsnorm_64x512", **stats(times)})

    # Conv1d: in=32, out=64, kernel=3; x is [1, 32, 128]
    conv1d = torch.nn.Conv1d(32, 64, 3)
    x_conv = torch.randn(1, 32, 128)
    times = bench(lambda: conv1d(x_conv), ITERS_FAST)
    results.append({"op": "conv1d_1x32x128_k3", **stats(times)})

    # AvgPool2d: kernel=2, stride=2; x is [1, 16, 32, 32]
    avgpool = torch.nn.AvgPool2d(2, 2)
    x_pool = torch.randn(1, 16, 32, 32)
    times = bench(lambda: avgpool(x_pool), ITERS_FAST)
    results.append({"op": "avgpool2d_1x16x32x32", **stats(times)})

    # GroupNorm: 8 groups, 64 channels; x is [4, 64, 16, 16]
    groupnorm = torch.nn.GroupNorm(8, 64)
    x_gn = torch.randn(4, 64, 16, 16)
    times = bench(lambda: groupnorm(x_gn), ITERS_FAST)
    results.append({"op": "groupnorm_4x64x16x16", **stats(times)})

    # RNN: input=128, hidden=256; x is [32, 1, 128] (seq_len=32, batch=1)
    rnn = torch.nn.RNN(128, 256, batch_first=False)
    x_rnn = torch.randn(32, 1, 128)
    h0_rnn = torch.zeros(1, 1, 256)
    times = bench(lambda: rnn(x_rnn, h0_rnn), ITERS_SLOW)
    results.append({"op": "rnn_seq32_128_256", **stats(times)})

    # LSTM: input=128, hidden=256; x is [32, 1, 128]
    lstm = torch.nn.LSTM(128, 256, batch_first=False)
    h0_lstm = torch.zeros(1, 1, 256)
    c0_lstm = torch.zeros(1, 1, 256)
    times = bench(lambda: lstm(x_rnn, (h0_lstm, c0_lstm)), ITERS_SLOW)
    results.append({"op": "lstm_seq32_128_256", **stats(times)})

    # GRU: input=128, hidden=256; x is [32, 1, 128]
    gru = torch.nn.GRU(128, 256, batch_first=False)
    h0_gru = torch.zeros(1, 1, 256)
    times = bench(lambda: gru(x_rnn, h0_gru), ITERS_SLOW)
    results.append({"op": "gru_seq32_128_256", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 4: Optimizers (full training step)
# ---------------------------------------------------------------------------


def _make_training_step(optimizer_cls, lr, **opt_kwargs):
    """Helper: create MLP + optimizer, return a step function."""
    w1 = torch.randn(784, 128, requires_grad=True)
    b1 = torch.zeros(1, 128, requires_grad=True)
    w2 = torch.randn(128, 64, requires_grad=True)
    b2 = torch.zeros(1, 64, requires_grad=True)
    w3 = torch.randn(64, 10, requires_grad=True)
    b3 = torch.zeros(1, 10, requires_grad=True)

    params = [w1, b1, w2, b2, w3, b3]
    opt = optimizer_cls(params, lr=lr, **opt_kwargs)
    targets = torch.arange(64) % 10

    def step():
        opt.zero_grad()
        x = torch.randn(64, 784)
        h1 = torch.relu(x @ w1 + b1)
        h2 = torch.relu(h1 @ w2 + b2)
        logits = h2 @ w3 + b3
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        opt.step()

    return step


def bench_optimizers():
    results = []

    # Adam (already covered by train_step_64, but with explicit optimizer name)
    step_adam = _make_training_step(torch.optim.Adam, lr=1e-3)
    times = bench(step_adam, ITERS_SLOW)
    results.append({"op": "optim_adam_64", **stats(times)})

    # RMSprop
    step_rmsprop = _make_training_step(torch.optim.RMSprop, lr=1e-3)
    times = bench(step_rmsprop, ITERS_SLOW)
    results.append({"op": "optim_rmsprop_64", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 5: Random
# ---------------------------------------------------------------------------


def bench_random():
    results = []

    times = bench(lambda: torch.rand(100_000), ITERS_FAST)
    results.append({"op": "rand_uniform_100k", **stats(times)})

    times = bench(lambda: torch.randn(100_000), ITERS_FAST)
    results.append({"op": "rand_normal_100k", **stats(times)})

    prob = torch.full((100_000,), 0.5)
    times = bench(lambda: torch.bernoulli(prob), ITERS_FAST)
    results.append({"op": "rand_bernoulli_100k", **stats(times)})

    times = bench(lambda: torch.rand(1_000_000), ITERS_FAST)
    results.append({"op": "rand_uniform_1M", **stats(times)})

    times = bench(lambda: torch.randn(1_000_000), ITERS_FAST)
    results.append({"op": "rand_normal_1M", **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 6: FFT
# ---------------------------------------------------------------------------


def bench_fft():
    results = []

    # rfft on real input
    for n in [1024, 4096, 16384]:
        x = torch.randn(n)
        label = f"rfft_{n // 1000}k"
        times = bench(lambda: torch.fft.rfft(x), ITERS_FAST)
        results.append({"op": label, **stats(times)})

    # fft on complex input
    for n in [1024, 4096]:
        x_real = torch.randn(n)
        x_imag = torch.randn(n)
        x_complex = torch.complex(x_real, x_imag)
        label = f"fft_{n // 1000}k"
        times = bench(lambda: torch.fft.fft(x_complex), ITERS_FAST)
        results.append({"op": label, **stats(times)})

    return results


# ---------------------------------------------------------------------------
# Phase 7: Linear algebra
# ---------------------------------------------------------------------------


def bench_linalg():
    results = []

    # Norm L2 on 1k vector
    x_norm = torch.randn(1000)
    times = bench(lambda: torch.linalg.norm(x_norm, 2), ITERS_FAST)
    results.append({"op": "norm_l2_1k", **stats(times)})

    for n in [64, 128, 256]:
        iters = ITERS_SLOW if n >= 256 else ITERS_FAST

        # Create positive-definite matrix: A_pd = B^T B + n*I
        b_mat = torch.randn(n, n)
        eye = torch.eye(n)
        a_pd = b_mat.T @ b_mat + n * eye

        # Create symmetric matrix for eigh: A_sym = B^T B
        a_sym = b_mat.T @ b_mat

        # General matrix for SVD, QR, det
        a_gen = torch.randn(n, n)

        # b vector for solve
        b_vec = torch.randn(n, 1)

        # Solve
        times = bench(lambda: torch.linalg.solve(a_pd, b_vec), iters)
        results.append({"op": f"solve_{n}x{n}", **stats(times)})

        # Inverse
        times = bench(lambda: torch.linalg.inv(a_pd), iters)
        results.append({"op": f"inv_{n}x{n}", **stats(times)})

        # Cholesky
        times = bench(lambda: torch.linalg.cholesky(a_pd), iters)
        results.append({"op": f"cholesky_{n}x{n}", **stats(times)})

        # SVD
        times = bench(lambda: torch.linalg.svd(a_gen), iters)
        results.append({"op": f"svd_{n}x{n}", **stats(times)})

        # QR
        times = bench(lambda: torch.linalg.qr(a_gen), iters)
        results.append({"op": f"qr_{n}x{n}", **stats(times)})

        # Eigendecomposition (symmetric)
        times = bench(lambda: torch.linalg.eigh(a_sym), iters)
        results.append({"op": f"eigh_{n}x{n}", **stats(times)})

        # Determinant
        times = bench(lambda: torch.linalg.det(a_gen), iters)
        results.append({"op": f"det_{n}x{n}", **stats(times)})

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
        # Phase 3B: NN layers
        bench_nn_layers,
        # Phase 4: Optimizers
        bench_optimizers,
        # Phase 5: Random
        bench_random,
        # Phase 6: FFT
        bench_fft,
        # Phase 7: Linear algebra
        bench_linalg,
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
    print(f"  Total operations benchmarked: {len(all_results)}")


if __name__ == "__main__":
    main()
