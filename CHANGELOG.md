# Changelog

All notable changes to Peregrine are documented here.
Benchmark numbers included for performance-related changes.

**Hardware:** Apple Silicon (M-series), macOS, f32 precision

---

## [0.16.0] - 2026-03-02

### Added — Reinforcement learning module and interactive demos

Full RL infrastructure built on Peregrine's tensor and autograd system: algorithms, environments, and interactive HTML visualizations.

**RL algorithms** (`src/rl.rs` — ~1,750 lines)
- `Environment` trait with `reset()`, `step()`, `render()`, observation/action spaces
- `ReasoningEnv` trait for math/logic environments with `question()`, `answer()`, `score_answer()`
- `Space` enum: `Discrete`, `Box`, `MultiDiscrete`, `MultiBinary` with `sample()` and `n()`
- `ReplayBuffer` for off-policy methods (DQN) — ring buffer with batch sampling
- `RolloutBuffer` for on-policy methods (PPO) — GAE advantage estimation, minibatch iteration
- **REINFORCE** — Monte Carlo policy gradient with optional baseline (running mean return)
  - `train_batch()` for batched episode collection and gradient updates
- **PPO** (Proximal Policy Optimization) — clipped surrogate objective with GAE
  - Configurable rollout steps, batch size, epochs, clip epsilon, entropy/value coefficients, max grad norm
- **DQN** (Deep Q-Network) — epsilon-greedy exploration with target network
  - Configurable epsilon decay, buffer size, batch size, gamma, target update frequency

**RL environments** (`src/envs.rs` — ~2,150 lines)
- Classic control: `CartPole`, `MountainCar`
- Grid/navigation: `GridWorld` (configurable size, obstacles), `FrozenLake` (slippery grid)
- Reasoning/math: `BasicArithmetic` (addition/subtraction/multiplication), `ChainArithmetic`, `NumberSorting`, `SequenceCompletion`
- Logic: `PropositionalLogic` (boolean expression evaluation)
- Game: `TicTacToe` (self-play)

**Interactive demo** (`examples/rl_demo/main.rs`)
- PPO on CartPole: pole-balancing animation (canvas), learning curves (Chart.js)
- DQN on GridWorld: agent pathfinding animation with trail, grid visualization, step-by-step controls
- REINFORCE on BasicArithmetic: flash-card style quiz animation with score tracking
- All animations: play/pause, restart, speed controls (0.5x/1x/2x), dark theme
- HTML output: `rl_cartpole.html`, `rl_cartpole_anim.html`, `rl_gridworld.html`, `rl_gridworld_anim.html`, `rl_arithmetic.html`, `rl_arithmetic_anim.html`

```
cargo run --example rl_demo --release               # PPO on CartPole
cargo run --example rl_demo --release -- gridworld   # DQN on GridWorld
cargo run --example rl_demo --release -- arithmetic  # REINFORCE on BasicArithmetic
```

### Stats

- ~30,000 lines of Rust (up from ~25,000)
- 3 RL algorithms (PPO, DQN, REINFORCE)
- 10 RL environments
- 6 interactive HTML visualizations (3 learning curves + 3 animations)

---

## [0.15.0] - 2026-02-28

### Added — MUSt3R server mode, parallel workers, and Metal GPU inference

Three performance features for the multi-view 3D reconstruction pipeline.

**Server mode** (`examples/must3r/main.rs`)
- `--server` flag: load weights once, read image pairs from stdin, write binary pointmaps to stdout
- Protocol: tab-separated `<img1>\t<img2>\t<W>\t<H>\n` requests, binary response (`8 + 32*H*W` bytes)
- Eliminates ~0.5s weight-loading overhead per pair (33s saved across 67 pairs)

**Parallel workers** (`reconstruct_video.py`)
- `--workers N` flag: spawns N server processes, distributes pairs across them via ThreadPoolExecutor
- Each worker has its own model copy (~1.7GB); 4 workers = ~6.8GB RAM
- `MUSt3RServer` class wraps `subprocess.Popen` for persistent stdin/stdout communication

**Metal GPU inference** (`--features metal`, `--gpu` flag)
- New `gelu_f32` Metal compute kernel for GPU-resident GELU activation
- `to_gpu()` methods on all model components (encoder, decoder, head) to upload weights to GPU
- `use_gpu: bool` threaded through forward calls; activations uploaded at encoder/decoder entry points
- GPU dispatch for GELU in `tensor.rs` via `dispatch_unary` (same pattern as relu/sigmoid)

### Fixed

- **Metal GELU kernel NaN**: Metal's fast-math `tanh()` produced NaN for certain input values (~10.5) when used in the GELU computation pattern. Changed to `precise::tanh()` in both `gelu_f32` and `gelu_backward_f32` kernels. Root cause: the Metal compiler's fast GELU approximation has precision issues for large-ish inputs that compound through the `0.5 * x * (1 + tanh(inner))` expression.

### Benchmark Results

**Server mode vs subprocess-per-pair (CPU, Apple Silicon):**

| Resolution | Subprocess | Server (warm) | Speedup |
|-----------|----------:|-------------:|--------:|
| 224x224 | ~0.57s/pair | ~0.51s/pair | 1.1x per pair |
| 512x384 | ~1.90s/pair | ~1.81s/pair | 1.05x per pair |

Server mode eliminates weight loading overhead. With parallel workers, wall-clock time scales near-linearly with worker count.

**GPU vs CPU (server mode, Apple Silicon):**

| Resolution | CPU | GPU | Note |
|-----------|----:|----:|------|
| 224x224 | 0.51s | 0.52s | Parity |
| 512x384 | 1.83s | 1.82s | Parity |

GPU mode produces byte-identical output to CPU. Performance is at parity due to decoder GPU↔CPU roundtrips in stack/split features and Apple's AMX-based sgemm matching the 16x16 tiled Metal matmul.

---

## [0.14.0] - 2026-02-28

### Added — Multi-view global pose optimization for MUSt3R reconstruction

New `reconstruct_video.py` script for multi-view 3D reconstruction from video using Peregrine's MUSt3R inference. Replaces pairwise Procrustes chaining with joint global optimization, eliminating drift and enabling loop closures.

**Dense pair selection** (`reconstruct_video.py`)
- `--pairs` flag: `consecutive` (N-1 pairs, original), `dense` (~3N pairs), `all` (N*(N-1)/2 pairs)
- Dense mode adds skip-1 and skip-2 connections; all mode runs every pair
- Ensures every view appears as first image in at least one pair for canonical pointmaps

**Global pose optimization** (`reconstruct_video.py`)
- Per-view parameters: rotation (axis-angle), translation, log-scale — 7 × (N-1) variables
- Initialized from Procrustes chain on consecutive pairs
- Constraints: for each pair (i,j), T_i(pts2) should match T_j(canonical pts1) — subsampled ~500 pixels per constraint
- Solver: `scipy.optimize.least_squares(method='trf', loss='soft_l1')` — robust to outliers
- Canonical pair selection: picks pair with highest mean confidence for each view

**Point fusion** (`reconstruct_video.py`)
- Confidence-weighted average of all predictions for each view's pixels across all pairs
- Produces one clean fused pointmap per view, reducing noise from individual pair predictions

**Visualization**
- Plotly HTML output with scroll zoom enabled
- Configurable confidence filtering (`--conf-percentile`), face limits (`--max-faces`), point cloud mode (`--points`)

### Benchmark Results (MUSt3R multi-view, 12 frames, CPU, Apple Silicon)

| Mode | Pairs | Inference | Optimization | Total |
|------|------:|----------:|-------------:|------:|
| consecutive (224) | 11 | ~7s | — | ~8s |
| dense (224) | 31 | ~21s | ~1s | ~23s |
| all (224) | 67 | ~45s | ~2s | ~48s |
| all (512) | 67 | ~3min | ~3s | ~3.2min |

Global optimization converges in 20-42 function evaluations. Cost reduction: 8-9x from initial Procrustes estimate.

---

## [0.13.0] - 2026-02-27

### Added — MUSt3R inference performance sprint

End-to-end optimization of the MUSt3R 3D reconstruction example (423M param ViT-L encoder + ViT-B decoder). Peregrine now matches PyTorch at 224x224 and beats it by 13% at 512x384 on Apple Silicon CPU.

**Weight loading** (378x speedup) (`src/serial.rs`)
- BufReader wrapping for amortized I/O
- Bulk tensor read: single `read_exact` per tensor instead of per-element 4-byte reads
- Bulk tensor write: byte reinterpret slice for `save_model`

**Batched encoder** (`examples/must3r/model.rs`)
- Both images processed in a single encoder pass with batch=2
- Eliminates cold-cache warmup penalty (~40ms at 224)
- Doubles GEMM sizes for better AMX utilization

**Batched decoder** (`examples/must3r/decoder.rs`)
- Self-attention and FFN process both views together (batch=2)
- Cross-attention stays separate (different KV per view)
- Batched feature embedding projection and final LayerNorm

**Parallel multi-head attention** (`src/tensor.rs`)
- `multi_head_attention()`: rayon par_chunks_mut with pre-allocated output
- Each thread writes directly to its output slice (no Vec<Vec<f32>> collect+flatten)
- Sequential fallback for small sequences (seq_q * seq_kv < 4096)
- `sgemm_strided()`: zero-copy GEMM with element offsets into existing buffers

**Vectorized softmax** (`src/tensor.rs`)
- `softmax_rows_inplace()`: NEON vmaxq/vmaxvq max-reduction, Accelerate `vvexpf` for bulk exp, NEON vaddvq sum-reduction, NEON vmulq normalize
- Added `vvexpf` FFI declaration

**Fixed GELU for large tensors** (`src/tensor.rs`)
- Large-tensor path (>100K elements) was using scalar `.tanh()` via rayon, bypassing the vvtanhf+NEON fast path
- Now uses chunked parallel pipeline: each rayon thread runs prep -> vvtanhf -> NEON combine on its 32K-element chunk

**Fused QKV split+reshape** (`examples/must3r/encoder.rs`, `examples/must3r/decoder.rs`)
- Single pass from [batch*seq, 3*embed_dim] directly to Q,K,V in [batch, heads, seq, head_dim] layout
- Eliminates 3 intermediate Vec allocations per attention call

**Direct transpose loops** (`examples/must3r/encoder.rs`, `examples/must3r/decoder.rs`)
- Replaced Tensor::transpose(1,2).reshape() with direct memory copy loops
- Avoids allocating + copying full 4D tensors

**NEON LayerNorm** (`src/tensor.rs`)
- Single-pass Welford algorithm with NEON vaddq + vfmaq for sum/sum_sq
- Fused normalize+scale+shift in one NEON pass

### Benchmark Results (MUSt3R, CPU, Apple Silicon)

| Metric | Before | After | PyTorch | Improvement |
|--------|--------|-------|---------|-------------|
| Weight load | 264.7s | 0.6s | 1.6s | 441x (2.7x faster than PyTorch) |
| Inference 224 | 1.87s | 0.67s | 0.67s | 2.8x (matches PyTorch) |
| Inference 512 | 10.45s | 1.98s | 2.26s | 5.3x (13% faster than PyTorch) |

---

## [0.12.0] - 2026-02-26

### Added — PyTorch feature parity sprint

Closes both performance gaps and missing features identified in the 133-op PyTorch benchmark comparison. Adds 30+ new NN modules, 5 new random distributions, 4 new linalg functions, 2D FFT, and multi-dimensional indexing. All with full autograd backward support.

**Phase 1: Critical Performance Fixes** (`src/fft.rs`, `src/tensor.rs`, `src/simd_kernels.rs`, `src/nn.rs`)
- FFT setup caching: thread-local `HashMap<u64, *mut c_void>` cache for `vDSP_create_fftsetup()` — eliminates ~300us per-call overhead. Added `vDSP_fft_zip` FFI for complex-to-complex FFT via Accelerate.
- SiLU NEON dispatch: new `Op::Silu` with single-pass NEON kernel (`vec_silu_f32`), replacing 2-intermediate-tensor composition
- cosine_similarity: `pow(2.0)` → `square()` (avoids `exp(2*ln(x))` per element)
- logaddexp NEON kernel: `vec_logaddexp_f32` using `vmaxq_f32`, `vabsq_f32`, `fast_exp_f32x4`
- GELU via vForce: `vvtanhf` FFI + NEON combination for `0.5 * x * (1 + tanh(inner))`
- GroupNorm optimization: `pool_get()` allocation, fused mean+variance single pass, precomputed fused_scale/fused_bias
- Transcendentals via vForce: `vvsinhf`, `vvcoshf`, `vvasinf`, `vvatanf` fast paths on macOS

**Phase 2: Conv2d Module + Generalized MaxPool** (`src/tensor.rs`, `src/nn.rs`)
- `im2col_strided()` / `col2im_strided()` with configurable stride and padding
- `conv2d_strided()` forward + backward with `Op::Conv2dStrided`
- `nn::Conv2d` module (Kaiming init, configurable kernel/stride/padding)
- `max_pool2d_ext()` with configurable kernel_size/stride/padding + `Op::MaxPool2dExt`
- `nn::MaxPool2d`, `nn::MaxPool1d` modules

**Phase 3: Normalization Modules + Utility** (`src/tensor.rs`, `src/nn.rs`)
- `nn::LayerNorm` module (wraps functional `layer_norm`)
- `nn::BatchNorm2d` module with running_mean/running_var, EMA updates, train/eval modes
- `nn::BatchNorm1d` module (reshapes to 4D, delegates to BatchNorm2d)
- `Tensor::item()` — extract scalar value from single-element tensor
- `index_select(dim, indices)` with `Op::IndexSelect` backward (scatter-add)
- `index_add_(dim, indices, src)` in-place scatter-add helper

**Phase 4: ConvTranspose + Upsample + Transformer Containers** (`src/tensor.rs`, `src/nn.rs`)
- `conv_transpose2d()` / `conv_transpose1d()` with `Op::ConvTranspose2d` / `Op::ConvTranspose1d`, full backward
- `nn::ConvTranspose2d`, `nn::ConvTranspose1d` modules (Kaiming init)
- `upsample_nearest()` / `upsample_bilinear()` with `Op::UpsampleNearest` / `Op::UpsampleBilinear`, full backward
- `nn::Upsample` module with `UpsampleMode::Nearest` / `UpsampleMode::Bilinear`
- `nn::TransformerEncoder`, `nn::TransformerDecoder`, `nn::Transformer` container modules

**Phase 5: Tier 2 Features** (`src/nn.rs`, `src/linalg.rs`, `src/fft.rs`)
- Padding layers: `nn::ZeroPad2d`, `nn::ConstantPad2d`, `nn::ReflectionPad2d`, `nn::ReplicationPad2d`
- Dropout variants: `nn::Dropout2d` (channel-wise), `nn::AlphaDropout` (SELU-preserving)
- `nn::PixelShuffle`, `nn::PixelUnshuffle` (sub-pixel convolution rearrangement)
- `nn::AdaptiveAvgPool2d`, `nn::AdaptiveAvgPool1d` (dynamic window sizing)
- `linalg::matrix_rank` (via SVD), `linalg::cond` (condition number), `linalg::lstsq` (least-squares via LAPACK `sgels_`), `linalg::matrix_power` (exponentiation by squaring with inverse support)
- `fft::fft2`, `fft::ifft2`, `fft::rfft2`, `fft::irfft2` (2D FFT via row-wise + column-wise composition)

**Phase 6: Additional Distributions + InstanceNorm** (`src/random.rs`, `src/nn.rs`)
- `random::exponential` (inverse CDF), `random::gamma` (Marsaglia-Tsang), `random::beta` (via gamma), `random::poisson` (Knuth), `random::multinomial` (with/without replacement)
- `nn::InstanceNorm2d`, `nn::InstanceNorm1d` modules with running stats and train/eval modes

### Fixed
- `irfft` scaling bug: vDSP path used `1/N` instead of correct `1/(2N)` scale factor
- `test_seed_determinism` race: hold PRNG mutex across both seed+generate pairs to prevent concurrent test interference

### Stats

- 409 tests passing (up from 302)
- ~25,000 lines of Rust (up from ~19,500)
- 30+ new NN modules, 5 new distributions, 4 new linalg functions, 4 new 2D FFT functions
- 15 random distribution functions (up from 10)
- Tensor ops: ~220 (up from ~200)

### Benchmark Results (133 ops, CPU, Apple Silicon)

**Geometric mean ratio (Peregrine / Framework):**
- vs PyTorch: **0.95x** (Peregrine faster)
- vs MLX: **0.74x** (Peregrine 1.35x faster)
- vs JAX: **0.68x** (Peregrine 1.47x faster)
- vs TensorFlow: **0.55x** (Peregrine 1.82x faster)
- vs tinygrad: **0.09x** (Peregrine 11x faster)

**Wins: Peregrine 56/133, PyTorch 27, MLX 20, JAX 16, TensorFlow 14**

Key Phase 1 improvements:
| Operation | Before (us) | After (us) | Speedup |
|-----------|------------:|-----------:|--------:|
| rfft 1k | 302 | 2.2 | 137x |
| silu 100k | 125 | 64 | 2.0x |
| cosine_sim 64x64 | 73 | 14 | 5.2x |
| sinh 100k | 131 | 51 | 2.6x |
| cosh 100k | 128 | 46 | 2.8x |
| arcsin 100k | 72 | 52 | 1.4x |
| arctan 100k | 96 | 53 | 1.8x |

---

## [0.11.0] - 2026-02-26

### Added — MLX feature parity sprint

Massive expansion of the tensor op library, NN modules, optimizers, and supporting infrastructure. Brings Peregrine from ~60 ops to ~200 ops, closing the gap with MLX/PyTorch.

**Phase 1A: 21 unary math ops** (`src/tensor.rs`, `src/metal/shaders.rs`)
- reciprocal, square, rsqrt, floor, ceil, round, sign, expm1, log2, log10, log1p, erf, erfinv, sinh, cosh, arcsin, arccos, arctan, arcsinh, arccosh, arctanh
- Compositions: degrees, radians (no new kernel needed)
- 17 new Op variants with full autograd backward
- 21 new Metal compute kernels

**Phase 1B-1D: binary, clip/where, comparison ops** (`src/tensor.rs`, `src/metal/shaders.rs`, `src/metal/context.rs`)
- Binary math: maximum, minimum, power, arctan2, logaddexp
- Conditional: clip (with ClipParams), where (ternary), nan_to_num
- 12 comparison/logical ops: equal, not_equal, greater, greater_equal, less, less_equal, logical_and, logical_or, logical_not, isnan, isinf, isfinite
- Utility: allclose, array_equal
- New dispatch methods: `dispatch_clip`, `dispatch_ternary`, `dispatch_nan_to_num`

**Phase 1E: 18 axis reduction ops** (`src/tensor.rs`, `src/metal/shaders.rs`, `src/metal/context.rs`)
- Differentiable: sum_axis, mean_axis, max_axis, min_axis, var, std, prod_axis, logsumexp, cumsum, cumprod
- Non-differentiable: any, all, argmax_axis, argmin_axis, sort, argsort, topk
- ReduceAxisParams + VarAxisParams structs for GPU dispatch
- New dispatch methods: `dispatch_reduce_axis`, `dispatch_var_axis`
- Helper methods: resolve_axis (negative indexing), axis_params, reduced_shape

**Phase 1F: 16 shape/indexing ops** (`src/tensor.rs`, `src/metal/shaders.rs`, `src/metal/context.rs`)
- tril, triu, repeat, tile, pad, roll, take, stack, split, broadcast_to, diagonal, diag, trace, outer, inner, expand_dims
- 4 new Metal kernels (tril, triu, pad, repeat) with TrilTriuParams/PadParams/RepeatParams
- New dispatch methods: `dispatch_tril`, `dispatch_triu`

**Phase 2: 18 activations + PReLU** (`src/tensor.rs`, `src/nn.rs`, `src/metal/shaders.rs`)
- Dedicated kernels: leaky_relu, elu (with forward + backward Metal kernels)
- Composed: silu, softplus, mish, hard_tanh, relu6, hardswish, softsign, log_sigmoid, selu, celu, gelu_fast, softmin, glu, hard_shrink, soft_shrink
- PReLU module with learnable weight parameter
- New dispatch methods: `dispatch_unary_param`, `dispatch_backward_unary_param`

**Phase 3: 11 loss functions + 12 NN layers** (`src/nn.rs`)
- Loss functions: l1, nll, smooth_l1, huber, kl_div, cosine_similarity, triplet, hinge, log_cosh, margin_ranking, gaussian_nll
- Module trait with forward(), params(), named_params(), train(), eval()
- Layers: RMSNorm, Dropout (train/eval mode), Identity, Sequential, RNN, LSTM, GRU, RoPE, Conv1d, AvgPool2d, GroupNorm, instance_norm
- Module impls for Linear, Embedding, PReLU, AvgPool2d

**Phase 4: 6 optimizers + 3 LR schedulers** (`src/optim.rs`)
- Optimizers: RmsProp (centered, momentum), Adagrad, Adamax, AdaDelta, Lion (sign-based), Adafactor (factored)
- LrSchedule trait (implemented for all existing schedulers)
- New schedulers: ExponentialDecayLr, LinearScheduleLr, JoinSchedules

**Phase 5: Random module** (`src/random.rs` — NEW)
- Xoshiro256++ PRNG engine (no external dependencies)
- 10 distributions: uniform, normal, randint, bernoulli, truncated_normal, gumbel, categorical, laplace, permutation
- Tensor::rand() and Tensor::rand_like() convenience methods

**Phase 6: FFT module** (`src/fft.rs` — NEW)
- CPU via Apple Accelerate vDSP (fft_zrip) with Cooley-Tukey radix-2 fallback
- fft, ifft, rfft, irfft, fftshift, ifftshift
- Complex representation: trailing dim of size 2

**Phase 7: Linear algebra module** (`src/linalg.rs` — NEW)
- CPU via LAPACK (Accelerate framework)
- norm, solve (sgesv), inv (sgetrf+sgetri), cholesky (spotrf), svd (sgesdd), qr (sgeqrf+sorgqr), eigh (ssyev), lu (sgetrf), det, pinv, cross, triangular_solve (strtrs)

**Phase 8: Transforms + Init** (`src/transforms.rs`, `src/init.rs` — NEW)
- grad(f, inputs), value_and_grad(f, inputs), checkpoint (stub)
- Weight init: glorot_uniform, glorot_normal, he_normal, he_uniform, lecun_normal, constant, orthogonal

### Performance optimizations

**10 new NEON SIMD kernels** (`src/simd_kernels.rs`)
- `vec_leaky_relu_f32`, `vec_leaky_relu_backward_f32` — `vcgtq_f32` + `vbslq_f32` conditional select
- `vec_elu_f32`, `vec_elu_backward_f32` — `fast_exp_f32x4` + conditional select
- `vec_silu_f32` — fused `x * sigmoid(x)` via `fast_exp_f32x4`
- `vec_maximum_f32`, `vec_minimum_f32` — `vmaxq_f32` / `vminq_f32`
- `vec_clip_f32` — `vmaxq_f32` + `vminq_f32` clamp
- `vec_square_f32` — `vmulq_f32(v, v)`
- `vec_reciprocal_f32` — `vrecpeq_f32` + Newton refinement

**NEON dispatch** (`src/tensor.rs`) — 8 forward ops (leaky_relu, elu, maximum, minimum, clip, square, reciprocal) and 2 backward ops (LeakyRelu, Elu) now dispatch to NEON kernels in the single-threaded path.

**Conv1d im2col + BLAS** (`src/nn.rs`) — Replaced 5-level nested loop with im2col matrix construction + `cblas_sgemm` for Conv1d::forward(). Expected ~10-20x speedup for typical Conv1d workloads.

**FFT buffer pool reuse** (`src/fft.rs`) — Replaced `vec!` allocations in rfft/irfft/fft/ifft with `pool_get`/`pool_recycle`. Reduces allocation overhead for repeated FFT calls.

**Random buffer pool reuse** (`src/random.rs`) — Replaced `.collect()` in `uniform()` and `normal()` with `pool_get` + fill loop. Avoids per-call allocation for large random tensors.

**Adafactor inner loop optimization** (`src/optim.rs`) — Precomputed `grad_sq` once for row/col factor updates, replaced `vec!` temporaries with pool buffers, precomputed reciprocals and constants.

### Stats

- 98 Metal compute shaders (up from 38), 30 dispatch methods (up from 24)
- 24 NEON SIMD kernels (up from 14) + Adam step kernel
- 302 tests passing (245 unit + 34 activation + 23 parity)
- ~19,500 lines of Rust (up from ~8,000)
- 5 new modules: random, fft, linalg, transforms, init

---

## [0.10.0] - 2026-02-23

### Added — GPU training pipeline optimization

**Command batching** (`src/metal/context.rs`)
- Accumulate dispatches into a single `MTLCommandBuffer` instead of per-op synchronous `waitUntilCompleted()`
- `gpu_sync()` commits and waits only at boundaries (reduce, data read-back)
- Individual op dispatch overhead reduced from ~200-300us to ~5us

**7 new Metal compute shaders** (`src/metal/shaders.rs`)
- `bias_add_f32` — broadcasts bias across rows for add_bias forward
- `bias_grad_sum_f32` — column-wise reduction for add_bias backward
- `log_softmax_f32` — numerically stable log_softmax forward (threadgroup shared memory reductions)
- `log_softmax_backward_f32` — `grad - exp(output) * sum(grad)` per row
- `scale_fill_f32` — broadcast `src[0] * scalar` to all elements (Mean/Sum backward)
- `gather_f32` — index gather for select forward
- `scatter_add_f32` — atomic scatter-add for select backward

**GPU forward paths** (`src/tensor.rs`)
- `add_bias` — dispatches `bias_add_f32` when both tensors are GPU-resident; eliminates the `sync_gpu_to_cpu()` that previously forced the entire MLP forward chain off GPU after the first matmul
- `log_softmax` — dispatches `log_softmax_f32` for 2D last-dim case; stores empty `output_data` (backward uses `gpu_data` directly)
- `select` — dispatches `gather_f32` on GPU, reads small result to CPU (keeps backward on CPU where it's faster at batch=64)

**GPU backward paths** (`src/tensor.rs`)
- `Op::AddBias` — `bias_grad_sum_f32` kernel replaces CPU row-sum (called 3x per training step)
- `Op::Add` — `dispatch_scale` copy (scale by 1.0) instead of GPU→CPU→GPU round-trip
- `Op::Mean` / `Op::Sum` — `scale_fill_f32` kernel instead of sync + scalar read + CPU fill
- `Op::Select` — fill zeros + `scatter_add_f32`
- `Op::LogSoftmax` — `log_softmax_backward_f32` kernel using `gpu_data` directly

**Eliminated softmax/log_softmax output cache sync**
- Softmax and log_softmax forward no longer sync GPU output to CPU for backward storage
- Backward uses the tensor's own `gpu_data` directly, avoiding a pointless GPU→CPU→GPU round-trip
- CPU backward fallback handles empty `output_data` via `self.data()` (calls `ensure_cpu_data`)

### Fixed

- Added `gpu_sync()` before reading `gpu_grad` in CPU backward fallback — prevents reading unsynced buffer data when GPU commands are still pending

### Stats

- 38 Metal compute shaders (up from 31)
- 140 tests passing

### GPU Benchmark Results (Metal, Apple Silicon, all times in microseconds)

| Operation | CPU (us) | GPU (us) | Note |
|-----------|----------:|----------:|------|
| matmul 128x128 | 6.1 | **5.0** | GPU wins (command batching) |
| matmul 256x256 | 33.0 | **4.4** | GPU 7.5x faster |
| matmul 512x512 | 165.1 | **4.4** | GPU 37x faster |
| add 100k | 13.4 | **4.7** | GPU 2.9x faster |
| mul 100k | 12.5 | **4.6** | GPU 2.7x faster |
| exp 100k | 111.0 | **4.2** | GPU 26x faster |
| relu 100k | 8.7 | **5.0** | GPU 1.7x faster |
| softmax 8x128 | 3.7 | **2.3** | GPU 1.6x faster |
| MLP fwd 64x784 | **33.4** | 34.3 | Near parity |
| train step 64 | **801.3** | 1632.8 | CPU wins (dispatch_reduce sync) |

GPU now wins on all individual ops thanks to command batching. The train_step gap is caused by `dispatch_reduce` in `mean()` forcing a mid-forward `commitAndWait()`. At larger batch sizes, GPU would win end-to-end.

---

## [0.9.0] - 2026-02-23

### Added — Metal autograd integration for end-to-end GPU training

**GPU backward kernels** (`src/metal/shaders.rs` — 10 new compute shaders)
- `relu_backward_f32`, `sigmoid_backward_f32`, `tanh_backward_f32`, `gelu_backward_f32` — activation backward passes
- `softmax_backward_f32` — fused per-row softmax Jacobian-vector product
- `layernorm_backward_f32` — fused grad_input/gamma/beta computation
- `adam_step_f32` — fused Adam optimizer update on GPU
- `accumulate_f32` — in-place gradient accumulation
- `fill_f32` — scalar broadcast fill
- `matmul_f32` extended with `trans_a`/`trans_b` parameters for backward matmul

**Dual storage in TensorInner** (`src/tensor.rs`)
- Optional `gpu_data` and `gpu_grad` fields on every tensor (behind `#[cfg(feature = "metal")]`)
- `to_gpu()`, `to_cpu()`, `is_gpu()` methods for explicit device placement
- `from_gpu_op()` constructor for tensors produced by GPU forward ops
- Lazy sync: `data()` and `grad()` auto-sync GPU→CPU on demand
- `sync_gpu_to_cpu()` ensures CPU data available for fallback paths

**GPU forward dispatch** (`src/tensor.rs` — 19 ops)
- All forward ops check GPU residence and dispatch to Metal: matmul, add, sub, mul, div, neg, scale, exp, log, sqrt, abs, relu, sigmoid, tanh, gelu, sin, cos, softmax, sum, mean, transpose, layer_norm
- Mixed CPU/GPU inputs gracefully fall back to CPU path
- GPU-produced tensors remain GPU-resident (no unnecessary CPU sync)

**GPU backward dispatch** (`src/tensor.rs`)
- Backward pass dispatches to Metal kernels when gradients and tensors are GPU-resident
- Covers: MatMul (two transposed matmuls), Add/Sub, Mul, Div, Relu, Sigmoid, Tanh, Gelu, Exp, Scale, Neg, Softmax, LayerNorm, Sum/Mean
- `gpu_accumulate_grad()` for in-place gradient addition on GPU
- Automatic fallback to CPU backward when GPU dispatch isn't available

**GPU optimizer step** (`src/optim.rs`)
- Adam optimizer dispatches fused `adam_step_f32` kernel when params have GPU data + grad
- Lazy-init GPU moment buffers on first step
- `zero_grad()` clears both CPU and GPU gradients

**GPU benchmark variants** (`benches/wallclock.rs`)
- All 14 benchmarks now have GPU counterparts (behind `--features metal`)
- GPU benchmarks: matmul, add, mul, exp, relu, softmax, MLP forward, training step

**Integration tests** (`tests/metal_autograd.rs` — 17 tests)
- GPU forward-backward gradient parity for 11 ops (add, sub, mul, relu, sigmoid, tanh, scale, neg, exp, matmul, softmax)
- GPU MLP training convergence (4→8→1 MLP, Adam, 50 epochs, loss decreases >50%)
- Mixed CPU/GPU fallback correctness
- Lazy sync correctness (data, grad, chained ops, GPU↔CPU roundtrip)

### Stats

- 31 Metal compute shaders (up from 21)
- 140 tests passing (64 unit + 17 autograd + 12 basics + 23 parity + 23 pytorch + 1 doc)
- ~8,000 lines of Rust (up from ~5,000)

### GPU Benchmark Results (Metal, Apple Silicon, all times in microseconds)

| Operation | CPU (us) | GPU (us) | Note |
|-----------|----------:|----------:|------|
| matmul 128x128 | **6.2** | 324.3 | CPU wins (BLAS + small size) |
| matmul 512x512 | **172.9** | 1673.4 | CPU wins (dispatch overhead) |
| add 100k | **12.5** | 289.3 | CPU wins (NEON + small size) |
| relu 100k | **9.3** | 241.5 | CPU wins (NEON + small size) |
| MLP fwd 64x784 | **33.5** | 881.5 | CPU wins |
| train step 64 | **921.9** | 1669.7 | CPU wins |

GPU is slower at current tensor sizes due to per-op synchronous `waitUntilCompleted()` overhead (~200-300us per dispatch). Command batching (M7) will eliminate this overhead and shift the crossover point to smaller tensor sizes.

---

## [0.8.0] - 2026-02-22

### Added — NEON intrinsics & elementwise dominance

**Hand-tuned NEON kernels** (`src/simd_kernels.rs` — new, ~530 lines)
- 14 NEON intrinsic kernels processing 4 f32s per iteration via `float32x4_t`
- Forward: `vec_add_f32`, `vec_sub_f32`, `vec_mul_f32`, `vec_div_f32`, `vec_neg_f32`, `vec_abs_f32`, `vec_scale_f32`, `vec_relu_f32`, `vec_add_inplace_f32`
- Transcendental: `vec_exp_f32` (Cephes-style polynomial, ~1.2e-7 relative error), `vec_sigmoid_f32`, `vec_tanh_f32`, `vec_gelu_f32`
- Backward: `vec_relu_backward_f32`, `vec_abs_backward_f32`, `vec_tanh_backward_f32`, `vec_sigmoid_backward_f32`
- All gated with `#[cfg(target_arch = "aarch64")]`; non-ARM targets use scalar fallback

**NEON Adam optimizer** (`src/simd_kernels.rs`)
- `adam_step_f32` vectorizes the 10+ FLOPs/element Adam inner loop
- Uses `vrsqrteq_f32` + Newton step for fast approximate sqrt
- Integrated in `src/optim.rs` via cfg-gated dispatch

**Pool bypass for small tensors** (`src/cpu_pool.rs`)
- `MIN_POOL_SIZE = 1024`: tensors < 1024 elements skip the HashMap pool
- Eliminates ~18 HashMap lookups per MLP forward pass where pool overhead exceeds malloc savings

**Integration** (`src/tensor.rs` — ~25 call sites)
- All forward ops (add, sub, mul, div, neg, abs, scale, relu, exp, sigmoid, tanh, gelu) dispatch to NEON kernels in the single-threaded path
- All backward ops (mul, relu, sigmoid, exp, scale, sub/neg, abs, tanh) dispatch to NEON kernels
- `accumulate_grad` uses `vec_add_inplace_f32`

### Benchmark Results (CPU, Apple Silicon, all times in microseconds)

Before (v0.7.0) → After (v0.8.0):

| Operation | Before (us) | After (us) | Speedup |
|-----------|------------:|-----------:|--------:|
| add 100k | 73.2 | 12.5 | 5.9x |
| mul 100k | 73.8 | 12.5 | 5.9x |
| relu 100k | 41.0 | 8.8 | 4.7x |
| MLP fwd | 37.1 | 32.6 | 1.1x |
| train step | 1135.7 | 809.1 | 1.4x |

**Geometric mean (Peregrine / framework):**
- vs PyTorch: 1.01x → **0.70x** (Peregrine 1.4x faster)
- vs MLX: 0.97x → **0.57x** (Peregrine 1.8x faster)
- vs TensorFlow: 0.61x → **0.40x**
- vs tinygrad: 0.12x → **0.08x**
- vs JAX: 0.49x → **0.39x**

**Wins:** Peregrine 8/14, PyTorch 3/14, JAX 2/14, TensorFlow 1/14

---

## [0.7.0] - 2026-02-22

### Added — CPU performance optimizations (close gap vs PyTorch & MLX)

**CPU buffer pool** (`src/cpu_pool.rs`)
- Thread-local, size-bucketed `HashMap<usize, Vec<Vec<f32>>>` with power-of-2 keys
- `pool_get(len)` / `pool_recycle(buf)` — eliminates malloc on every forward & backward op
- `Drop` impl on `TensorInner` auto-recycles `data` and `grad` buffers
- Converted ~20 forward ops and ~15 backward ops from `.collect()` to pool-allocated loops

**SIMD auto-vectorization** (`.cargo/config.toml`)
- `target-cpu=apple-m1` enables full NEON/ASIMD instruction set
- Simple loops like `data[i] = a[i] + b[i]` now emit vectorized `fadd v0.4s`

**Rayon threshold tuning**
- Dual thresholds: `PAR_THRESHOLD_CHEAP = 500_000` (add, mul, relu, etc.), `PAR_THRESHOLD_EXPENSIVE = 100_000` (exp, log, sqrt, etc.)
- Avoids ~15us Rayon spawn overhead on ops that complete in <10us single-threaded

**Adam/SGD gradient borrow fix** (`src/optim.rs`)
- Borrow `grad` and `data` fields in-place via split `&mut` instead of cloning entire gradient Vec

**JAX benchmark** (`scripts/bench_jax.py`)
- Added JAX 0.9.0.1 to the 6-framework comparison suite

### Benchmark Results (CPU, Apple Silicon, all times in microseconds)

Before → After (Peregrine medians):

| Operation | Before (us) | After (us) | Speedup |
|-----------|------------:|-----------:|--------:|
| relu 100k | 87.4 | 41.0 | 2.13x |
| add 100k | 121.3 | 73.2 | 1.66x |
| mul 100k | 116.8 | 73.8 | 1.58x |
| mul 500k | 124.5 | 109.3 | 1.14x |
| add 500k | 159.8 | 118.9 | 1.34x |
| train step | 1030.5 | 1135.7 | 0.91x |

**Geometric mean (Peregrine / framework):**
- vs PyTorch: 1.12x → **1.01x** (parity)
- vs MLX: 1.16x → **0.97x** (Peregrine now faster)
- vs TensorFlow: 0.66x → **0.61x**
- vs tinygrad: 0.14x → **0.12x**
- vs JAX: **0.49x** (Peregrine 2x faster)

**Wins:** MLX 5/14, Peregrine 3/14, PyTorch 3/14, TensorFlow 2/14, JAX 1/14

---

## [0.6.0] - 2026-02-22

### Added — Multi-framework wall-clock benchmark suite
- `benches/wallclock.rs` — Peregrine benchmark (standalone binary, JSON output)
- `scripts/bench_pytorch.py` — PyTorch 2.10.0 benchmark
- `scripts/bench_mlx.py` — MLX 0.30.6 benchmark
- `scripts/bench_tensorflow.py` — TensorFlow 2.20.0 benchmark
- `scripts/bench_tinygrad.py` — tinygrad 0.12.0 benchmark
- `scripts/compare_bench.py` — multi-framework comparison table (markdown output)
- `scripts/bench_compare.sh` — orchestrator: builds, runs all 5 frameworks sequentially with `nice -n 10`

14 operations benchmarked: matmul (128/256/512), add, mul, exp (100K/500K), relu, softmax (128/512), MLP forward, training step.

### Benchmark Results (CPU, Apple Silicon, all times in microseconds)

| Operation | Peregrine (us) | PyTorch (us) | MLX (us) | TensorFlow (us) | tinygrad (us) |
|-----------|---------------:|-------------:|---------:|----------------:|--------------:|
| matmul 512x512 | **162** | 165 | 174 | 676 | 434 |
| softmax 8x128 | **3.9** | 39.7 | 17.0 | 10.2 | 700 |
| MLP fwd 64x784 | **28.5** | 28.4 | 52.8 | 250 | 1831 |
| train step 64 | **1031** | 1462 | 782 | 8414 | 24801 |

**Geometric mean (Peregrine / framework):**
- vs PyTorch: 1.12x (near parity)
- vs MLX: 1.16x (near parity)
- vs TensorFlow: 0.66x (Peregrine 1.5x faster)
- vs tinygrad: 0.14x (Peregrine 7x faster)

**Wins:** PyTorch 5/14, MLX 5/14, Peregrine 2/14, TensorFlow 2/14

---

## [0.5.0] - 2026-02-22

### Added — Metal GPU Backend (`--features metal`)
- objc2-metal FFI foundation with safe Rust wrappers (GpuContext, GpuBuffer)
- 21 Metal compute shaders: add, sub, mul, div, neg, exp, log, sqrt, relu, sigmoid, tanh, sin, cos, abs, scale, matmul (fused bias+relu), sum, max, min, softmax, transpose, layernorm
- 35 GPU tests (12 basics + 23 CPU vs Metal parity)

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| matmul 1024x1024 | 1.07 ms | 3.49 ms | CPU 3.3x (BLAS) |
| add 1M elements | 1.04 ms | 306 µs | GPU 3.4x |
| mul 1M elements | 1.17 ms | 294 µs | GPU 4.0x |
| exp 1M elements | 1.32 ms | 284 µs | GPU 4.6x |

### Added — Numerically stable log_softmax
- `x - max - log(sum(exp(x - max)))` with backward pass
- Fixes NaN crash in cross_entropy_loss after many epochs

### Added — MNIST end-to-end example
- MLP (784→128→64→10), Adam optimizer, 10 epochs, 97.5% test accuracy

### Added — PyTorch numerical parity (23 tests)
- Cross-validates matmul, softmax, log_softmax, layernorm, cross_entropy, 14 element-wise ops, Adam step, and full 10-step MLP training
- All within 1e-4 to 1e-7 absolute error

### Added — Criterion benchmark suite
- `cargo bench` / `cargo bench --features metal`
- Covers matmul, element-wise, softmax, MLP forward, training step

| Operation | Time |
|-----------|------|
| MLP forward batch=64 | 186 µs |
| Training step (fwd+bwd+Adam) batch=64 | ~3 ms |

## [0.4.0] - 2026-02-20

### Added — Core tensor ops and training infrastructure
- 11 element-wise ops with autograd: sub, div, neg, exp, log, sqrt, abs, pow, sin, cos, tanh
- Reduction/shape ops: mean, squeeze, unsqueeze, max, min, argmax, argmin
- Creation ops: ones, full, arange, linspace, eye
- NumPy-style broadcasting for add, sub, mul, div
- Neural network layers: Linear, Embedding, CrossEntropyLoss, MSELoss
- Optimizers: SGD (momentum, Nesterov, weight decay), Adam, AdamW
- LR schedulers: StepLR, CosineAnnealing, Warmup
- Gradient clipping (by norm and by value)

### Changed
- Restructured as `peregrine` library crate. RT-DETR moved to examples.
- Dropped dead YOLO code

## [0.3.0] - 2026-02-20

### Added
- RT-DETR architecture: multi-head attention, transformer encoder/decoder, ResNet backbone, learned object queries
- Hungarian matching and set-based loss for end-to-end detection training
- Global gradient clipping (max norm = 1.0)

### Changed
- Xavier fan-in weight initialization — fixes NaN loss in deep networks
- Multi-scale feature pooling (3,072 tokens vs 56,784), reducing attention memory from ~52 GB to ~150 MB

## [0.2.0] - 2026-02-20

### Performance
- BLAS acceleration via Apple Accelerate (matmul, conv2d 1x1)
- Rayon parallelism for element-wise ops (>10k threshold)
- Clone elimination in backward pass

## [0.1.0] - 2026-02-18

### Added
- Tensor with N-dimensional storage and shared ownership
- Reverse-mode autograd engine
- Forward/backward: add, mul, matmul, relu, sigmoid, sum, scale, add_bias
- SGD optimizer
- Object detection demo with ASCII visualization
