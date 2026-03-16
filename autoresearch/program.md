# Peregrine Autoresearch — Autonomous Op Optimization

This is an autonomous optimization loop for Peregrine's SIMD kernels and op dispatch. You modify code, benchmark, keep improvements, discard regressions, and repeat.

## Setup

To set up a new optimization run:

1. **Agree on a run tag** with the user (e.g. `mar15`). Create branch `autoresearch/<tag>`.
2. **Read the in-scope files** for full context:
   - `src/simd_kernels.rs` — NEON SIMD kernels (the primary file you modify)
   - `src/tensor.rs` — Op dispatch, vForce paths, fused ops (secondary file you may modify)
   - `src/nn.rs` — Neural network layers with gate computations (tertiary)
   - `autoresearch/bench_quick.sh` — Quick benchmark script
3. **Establish baseline**: Run `./autoresearch/bench_quick.sh` to get current results.
4. **Initialize results.tsv**: Create `autoresearch/results.tsv` with the header row.
5. **Identify targets**: Run `./autoresearch/bench_quick.sh | grep LOSE` to see all losing ops.

## Target Ops (current losses sorted by gap)

Focus on these ops where Peregrine loses to another framework:

| Priority | Op | Peregrine | Best | Gap |
|----------|---|-----------|------|-----|
| 1 | silu_100k | 63µs | JAX 52µs | 21% |
| 2 | softplus_100k | 134µs | TF 132µs | 1.5% |
| 3 | arccos_100k | 62µs | TF 57µs | 8.7% |
| 4 | arcsinh_100k | 129µs | JAX 124µs | 4.1% |
| 5 | exp_100k | 50µs | JAX 46µs | 8.6% |
| 6 | exp_500k | 100µs | TF 98µs | 2% |
| 7 | train_step_64 | 831µs | MLX 772µs | 7.6% |
| 8 | lstm_seq32 | 1030µs | PyTorch 803µs | 28% |

Also: matmul_bias_gelu (2 sizes), mul_500k, rand_uniform/normal_1M.
Skip: LAPACK ops (eigh, inv, solve, cholesky, svd, qr at 128+ — same underlying library).

## What You CAN Modify

- `src/simd_kernels.rs` — NEON polynomial coefficients, loop unrolling, instruction scheduling, new kernels
- `src/tensor.rs` — Op dispatch paths, vForce vs NEON selection, fused op implementations, threshold tuning
- `src/nn.rs` — LSTM/GRU gate computation, layer implementations

## What You CANNOT Modify

- `benches/wallclock.rs` — The benchmark harness is the ground truth
- `scripts/bench_*.py` — Framework benchmark scripts
- `target/bench_compare/*.json` — Saved competitor results (except peregrine.json which is regenerated)
- Any test files — Tests must continue to pass

## Constraints

- **All 532+ tests must pass**: Run `cargo test --release --lib 2>&1 | tail -3` after each change
- **Accuracy**: Polynomial approximations must maintain f32 precision (typically 1e-4 to 1e-5 tolerance)
- **No new dependencies**: Only use what's already available (std, NEON intrinsics, Apple Accelerate)

## Known Dead Ends (DO NOT retry)

These approaches have been proven slower on Apple M-series:

1. **fast_recip_f32x4 (Newton-Raphson reciprocal)** — Apple vdivq_f32 is 3-4 cycles, Newton-Raphson is 5-6 cycles. SLOWER on Apple Silicon.
2. **vvexpf for silu** — The negate+copy+vvexpf overhead exceeds the benefit of Apple's bulk exp. SLOWER.
3. **NEON vec_arccos_f32 polynomial** — Our polynomial is slower than Apple's vvacosf (83µs vs 62µs). SLOWER.
4. **NEON vec_softplus_f32** — Our fast_log polynomial is slower than Apple's vvexpf+vvlog1pf for softplus. SLOWER.
5. **Polynomial sigmoid on [-8,8]** — Sigmoid doesn't converge well for polynomial approximation over this range. Taylor series diverges badly. DOESN'T WORK.

## Optimization Techniques to Try

### Promising approaches:
- **2x/4x loop unrolling** for better ILP on dual-issue M-series pipeline
- **Algebraic simplification** (e.g., x*(A + B*x²) instead of sqrt(2/pi)*(x + 0.044715*x³))
- **Hybrid vForce/NEON paths** with size-dependent dispatch (vForce better for large arrays, NEON for small)
- **Fused multi-op kernels** (e.g., bias+activation in single pass after matmul)
- **Apple Accelerate vForce** functions for ops where our polynomial is slower (vvtanhf, vvexpf, vvlog1pf)
- **Buffer pre-filling** with sgemm beta=1.0 to eliminate bias addition passes
- **Chebyshev minimax polynomials** for custom approximations (worked great for erf)
- **Instruction reordering** to hide latency (interleave independent FMA chains)

### For specific ops:
- **silu**: Try different exp approximations, range-reduced sigmoid, or erf-based sigmoid approximation
- **lstm**: Batch the h@W_hh projections across timesteps, or fuse gate activations into sgemm output
- **softplus**: Try computing log(1+exp(x)) = x + log(1+exp(-|x|)) to reduce dynamic range
- **train_step**: Profile which sub-ops dominate and optimize those

## The Metric

**Time in microseconds (lower is better).** The benchmark uses median of timed iterations with warmup excluded.

A change is an improvement if it makes ANY losing op faster while not regressing any winning op by more than 2%.

## Output Format

The benchmark script outputs:
```
op_name                  peregrine_us    best_us    framework    ratio    status
silu_100k                      63.6       52.1          JAX     1.221    LOSE
```

## Logging Results

Log each experiment to `autoresearch/results.tsv` (tab-separated):

```
commit	wins	total	status	description
```

1. git commit hash (short, 7 chars)
2. Number of benchmark wins (e.g. 113)
3. Total ops (141)
4. Status: `keep`, `discard`, or `crash`
5. Short description of what was tried

## The Experiment Loop

LOOP FOREVER:

1. Look at the current losing ops: `./autoresearch/bench_quick.sh | grep LOSE`
2. Pick a target op and form a hypothesis for how to improve it
3. Modify the relevant source file(s)
4. Build and test: `cargo test --release --lib 2>&1 | tail -3`
5. If tests fail, fix or revert
6. Benchmark: `./autoresearch/bench_quick.sh [op_name] 2>/dev/null`
7. If the target op improved AND no other ops regressed significantly: `git commit`, log as `keep`
8. If the target op didn't improve or other ops regressed: `git checkout -- src/`, log as `discard`
9. Go to step 1

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. If you run out of ideas, re-read the source code for new angles, try combining previous approaches, try more radical changes. The loop runs until the human interrupts you.

**Timeout**: Each experiment (build + test + benchmark) should take < 2 minutes. If something takes longer, kill it and move on.
