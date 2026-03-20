# Research Notes

## Apple AMX Coprocessor — March 2026

Investigated direct access to Apple's undocumented AMX (Apple Matrix Extension) coprocessor.

**Reference:** [corsix/amx](https://github.com/corsix/amx) — reverse-engineered instruction encodings by Dougall Johnson.

### Key Finding: AMX IS Accessible from Userspace on macOS 26

A previous attempt (commit 30a42d0) concluded AMX was blocked on macOS 26. **This was wrong** — the SIGILL was caused by incorrect instruction encodings, not an OS-level restriction.

| Bug | Wrong | Correct |
|-----|-------|---------|
| SET (enable) | op=0, no NOP sled | op=17, imm5=0, with 3-NOP sled |
| CLR (disable) | op=1, no NOP sled | op=17, imm5=1, with 3-NOP sled |
| LDX | op=2 | op=0 |
| LDY | op=3 | op=1 |
| LDZ | op=7 | op=4 |
| STZ | op=6 | op=5 |

The Z accumulator layout for FMA32 also uses stride-4 rows: output row `j` maps to Z row `j*4` (not `j`).

### AMX Instruction Encoding

```
Regular ops:  .word (0x00201000 | (op << 5) | gpr)
SET/CLR:      nop; nop; nop; .word (0x00201000 | (17 << 5) | imm5)
```

Opcodes: 0=LDX, 1=LDY, 2=STX, 3=STY, 4=LDZ, 5=STZ, 6=LDZI, 7=STZI, 12=FMA32, 17=SET/CLR.

Register file: 8 X regs (64B each), 8 Y regs (64B each), 64 Z regs (64B each = 4KB accumulator).

### Benchmark: Naive AMX vs Apple Accelerate (cblas_sgemm)

Both use the same AMX hardware. The difference is software optimization.

| Size | Naive AMX | cblas_sgemm | Ratio | cblas GFLOPS |
|------|-----------|-------------|-------|-------------|
| 16x16 | 1.9µs | 0.3µs | 6.7x slower | 30 |
| 64x64 | 65µs | 1.0µs | 64x slower | 513 |
| 128x128 | 495µs | 6.6µs | 74x slower | 631 |
| 256x256 | 3979µs | 30µs | 131x slower | 1107 |

Naive AMX: ~8.5 GFLOPS. Apple Accelerate: ~1000 GFLOPS. The gap is entirely Apple's optimized tiling, packing, register scheduling, and cache blocking.

### Why Direct AMX Doesn't Help Peregrine

- `cblas_sgemm` already uses AMX internally with deeply optimized tiling
- A naive outer-product loop is 70-130x slower than Accelerate
- Matching Accelerate's performance would require replicating years of cache-aware tiling, data packing, and instruction scheduling
- The only potential benefit (fused matmul+activation) is better served by Metal GPU

### Decision

**Keep `src/amx.rs` as working documentation** of the correct instruction encodings and a proof that AMX is accessible. Continue using `cblas_sgemm` for all matmul — it already dispatches to AMX with optimal performance.

### M4 and ARM SME

M4 introduced ARM SME (Scalable Matrix Extension), a standardized version of AMX. On M4, Accelerate likely uses SME instead of/in addition to raw AMX. SME is a public ARM standard — if Rust adds stable SME intrinsics, this could be a future optimization path.

---

## Apple Neural Engine (ANE) — March 2026

Investigated feasibility of adding an ANE compute backend to Peregrine.

**Reference:** Based on reverse-engineered private ANE APIs (Obj-C) demonstrating Stories110M training on ANE.

### What the ANE Offers

- Dedicated ML accelerator on Apple Silicon (M4: 15.8 TFLOPS theoretical)
- Separate from GPU — could run inference on ANE while GPU handles other work
- Supports: conv (linear as 1x1), matmul, softmax, cast, add, mul
- In-memory MIL (Model Intermediate Language) compilation — no .mlmodelc needed

### Why It's Not Viable Now

| Issue | Detail |
|-------|--------|
| **Private APIs** | Uses `_ANEClient`, `_ANECompiler`, `_ANEInMemoryModel` — undocumented, can break on any macOS update, not App Store compatible |
| **Low utilization** | Only 1.78 TFLOPS sustained (11.2% of theoretical) on M4 — Metal GPU is faster in practice |
| **Weights baked at compile** | Can't update weights without recompiling the kernel (~33ms each) |
| **Compile limit** | ~119 ANE compiles per process before resource leak; workaround is `exec()` restart |
| **SRAM ~8-10MB** | Layers with weights exceeding this spill to DRAM, killing performance |
| **fp16 only** | ANE operates in fp16; Peregrine uses f32 — constant conversion overhead |
| **No causal SDPA** | Hardware ignores attention masks; must decompose into Q@K^T → mask → softmax → @V |
| **Obj-C FFI** | All APIs are Objective-C — would need `objc2` bindings for private frameworks |

### Useful Takeaways

- **IOSurface patterns** — ANE repo uses IOSurface for zero-copy data sharing between CPU and ANE; similar patterns could optimize Metal ↔ CPU transfers
- **MIL format** — Understanding MIL would give a head start if Apple opens up public ANE compute APIs
- **SRAM/dispatch benchmarks** — Useful context for understanding Apple Silicon memory hierarchy

### What Would Be Needed

If Apple provides public APIs in the future, an ANE backend would follow the same pattern as Metal:

1. Feature-gated `src/ane/` module (`--features ane`)
2. `AneContext` singleton with thread-local accessor (`with_ane()`)
3. `AneBuffer<T>` wrapper for IOSurface-backed tensors
4. Per-op dispatch: ANE path tried first, CPU fallback
5. Lazy sync via `ane_dirty` flag on `TensorInner`
6. MIL kernel generation for supported ops (matmul, add, gelu, softmax, layernorm)

### Decision

**Deferred.** The private API dependency is a dealbreaker for a production framework. Revisit when Apple provides public ANE compute APIs beyond CoreML.

---

## RL for Reasoning LLMs — March 2026

**Source:** [A. Weers — State of RL for Reasoning LLMs](https://aweers.de/blog/2026/rl-for-llms/#state-of-rl-for-reasoning-llms) (March 15, 2026)

Survey of reinforcement learning methods for reasoning-capable LLMs, covering developments from 2024–2026.

### RL Setup for LLMs

Standard RL simplified: sample responses from prompts, assign scalar rewards — no token-level feedback. Agent observes states, selects actions via policy, receives rewards, maximizes expected discounted returns.

### Methods

**REINFORCE** — "Weighted SFT" that reinforces sampled responses based on rewards. Uses baselines to reduce variance without biasing gradient estimates.

**PPO (Proximal Policy Optimization)** — Importance sampling ratios with clipping for off-policy updates across multiple optimization steps. Clipping acts as trust region, blocking gradient updates that diverge too far from generation policy.

**GRPO (Group Relative Policy Optimization)** — Removes PPO's value model critic, replaces with group-relative baselines (normalizing rewards within sample groups per prompt). Significantly reduces memory requirements.

**RLOO (REINFORCE Leave-One-Out)** — Leave-one-out baselines without standard deviation normalization or PPO-style clipping. Pure policy gradient updates with unbiased advantage estimates.

**Dr. GRPO** — Identifies and corrects normalizations that bias learning signals. Key fix: aggregating losses at token level with fixed normalization rather than sequence-level averaging.

**DAPO (Decoupled Advantage Policy Optimization)** — Token-level aggregation, asymmetric clipping bounds for rare tokens, dynamic sampling to ensure mixed outcomes per prompt.

**CISPO (Clipped Importance Sampling Policy Optimization)** — Decouples clipping from gradient masking via stop-gradient on clipped weights rather than blocking gradients entirely. "More informative gradients" for high-impact tokens.

**DPPO (Divergence PPO)** — Replaces probability ratio-based trust regions with divergence-based constraints. Token probability changes are "a poor proxy for actual policy divergence."

**MaxRL (Maximum Likelihood RL)** — Frames RL as approximate maximum-likelihood training. Optimizes truncated harmonic mixture of pass@k objectives (not just pass@1), improving diversity and test-time scaling.

**ScaleRL** — Large-scale empirical validation (400,000+ GPU-hours). Findings: asynchronous RL, FP32 logits for ratio stability, and prompt-level loss aggregation yield optimal asymptotic performance.

### Key Patterns

| Pattern | Detail |
|---------|--------|
| **Critic removal** | All post-PPO methods eliminate learned value functions, saving ~50% memory while maintaining/improving performance |
| **Std-dev concerns** | Dividing advantages by standard deviation overweights nearly-solved problems, reduces asymptotic performance |
| **Loss aggregation** | Method of reducing losses significantly impacts per-token learning signals — "not a minor detail" |
| **Trust region evolution** | PPO's symmetric clipping (ε=0.2) works well, but newer approaches explore asymmetric bounds, weight clipping, or divergence-based definitions |

### Implementation

All 8 algorithms (GRPO, RLOO, Dr. GRPO, DAPO, CISPO, DPPO, MaxRL, ScaleRL) implemented in `src/rl.rs` with shared `SequenceGroup`/`Completion` types, `group_relative_advantages()`, and `token_level_loss()` helpers. Each has `XxxConfig` (with `Default` + builders) and `XxxTrainer` (with `new`, `compute_loss`, `update`). 37 tests cover config defaults, builders, and smoke tests.

### Open Problems

1. **Credit assignment** — Outcome-based rewards treat all tokens equally despite token-level importance variations
2. **Sample efficiency** — Methods typically require 8–64 rollouts per prompt; better signal extraction from failed attempts could reduce costs
3. **Hard problems** — No gradient when models never produce correct rollouts; curriculum learning only partially addresses this
4. **Beyond math/code** — Extension to domains with noisy, delayed, or subjective rewards remains difficult
5. **Empirical reliability** — Most evidence is "narrow and expensive to reproduce," testing single model families and compute budgets

---

## Recursive Language Models (RLMs) — October 2025

**Source:** [Alex Zhang & Omar Khattab — Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) (MIT CSAIL, October 15, 2025)

**Paper:** [arXiv:2512.24601v1](https://arxiv.org/abs/2512.24601v1) | **Code:** [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm) | [rlm-minimal](https://github.com/alexzhang13/rlm-minimal)

Inference strategy enabling LMs to decompose and recursively interact with unbounded input context through REPL environments.

### Key Results

- RLM(GPT-5-mini) outperforms GPT-5 by over 2× on OOLONG benchmark's hardest questions
- Handles 10M+ token contexts without performance degradation
- Comparable cost to direct GPT-5 calls
- New axis for test-time compute scaling beyond CoT and ReAct

### Problem: Context Rot

Model performance degrades as context length increases. Needle-in-haystack benchmarks show 90%+ performance, but real-world tasks degrade in extended conversations and long sessions.

### Framework

RLMs wrap language models with ability to spawn recursive LM calls for intermediate computation. Context-centric (not problem-centric) decomposition — maintains functional view where the system answers queries over associated context.

**Implementation:** Python REPL notebook environment. Root LM (depth=0) interacts with environment and can:
- Call recursive LMs (depth=1) as functions within code
- Read/write to notebook cells
- Peek at, partition, grep through, and launch sub-queries over context

**Output:** `FINAL(answer)` for direct answers, `FINAL_VAR(variable_name)` for environment-constructed answers.

### Formal Definition

For model M, query q, context C = [c₁, c₂, …, cₘ]: RLM_M(q,C) is an expressive wrapper over environment ℰ with same I/O spaces. Provides tool to spawn isolated sub-RLM instances with new query q̂ and transformed context Ĉ. Simplest environment ℰ₀ just queries model directly — RLMs generalize this.

### Experimental Results

**OOLONG Benchmark (132k tokens):**

| Method | Relative Performance |
|--------|---------------------|
| GPT-5 (direct) | baseline |
| GPT-5-mini (direct) | below baseline |
| RLM(GPT-5-mini) | +34 points (~114% over GPT-5) |
| ReAct + GPT-5 + BM25 | below RLM |

At 263k tokens (~context limit): RLM(GPT-5-mini) outperforms GPT-5 by ~15 points (~49% increase), cheaper per query.

**BrowseComp-Plus (10M+ tokens):**
- Only RLM(GPT-5) achieved perfect performance at 1000-document scale
- Base GPT-5 approaches showed clear dropoff with increasing documents
- RLM cost per query scales reasonably

### Emergent Strategies

- **Peeking** — Examine initial context sections to understand structure before detailed analysis
- **Grepping** — Keyword/regex patterns to narrow search (not semantic retrieval)
- **Partition + Map** — Chunk context, launch recursive calls for extraction/semantic mapping
- **Summarization** — Summarize subsets for outer LM decision-making
- **Long-input, Long-output** — Excels at extensive output generation (e.g., BibTeX from paper lists)

### Comparison to Prior Work

| Approach | Distinction |
|----------|-------------|
| **MemGPT** | Defers context management to model but builds on single context |
| **MemWalker** | Imposes tree-like structure for summarization ordering |
| **LADDER** | Decomposes from problem perspective rather than context perspective |
| **THREAD** | Modifies output generation to spawn child threads |

**Key philosophical distinction:** Unlike agents based on human/expert intuition for problem decomposition, RLMs let the LM decide decomposition strategy.

### Implementation

RLM orchestrator implemented in `src/rlm.rs` with `GenerativeLM` trait, `RlmOrchestrator` recursive engine, `ReplContext` variable store, 7 `RlmAction` variants, dependency-free tag parser. `examples/rlm/` wraps the Llama model as a `GenerativeLM` backend. 41 tests cover parsing, orchestration, depth limits, token budgets, and mock model flows.

### Limitations

- No asynchronous execution or prefix caching
- Each recursive call blocks subsequent execution
- Runtime varies from seconds to minutes per query
- No strong cost/runtime guarantees

### Future Directions

- RLM capabilities correlate directly with base model improvements — if frontier models handle 10M tokens, RLMs could handle 100M (at ~half cost)
- RLM trajectory for recursing over context is learnable and can be RL-optimized
- Fixed formats (à la CoT, ReAct) improve performance; scaling opportunity through fixed-format training data

---

## LLM Architecture Gallery — March 2026

**Source:** [Sebastian Raschka — LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) (last updated March 17, 2026)

Visual reference of 48 LLM architectures from four articles: *The Big LLM Architecture Comparison*, *From GPT-2 to gpt-oss*, *From DeepSeek V3 to V3.2*, and *A Dream of Spring for Open-Weight LLMs*.

### Dense Architectures

| Model | Params | Key Features | Context |
|-------|--------|-------------|---------|
| GPT-2 XL | 1.5B | MHA, dropout, classic 2019 baseline | 1,024 |
| Llama 3 | 8B | GQA + RoPE, reference dense stack | 8,192 |
| Llama 3.2 | 1B | Small dense, fewer layers, wider | 128,000 |
| OLMo 2 | 7B | Inside-residual post-norm, QK-Norm, transparent | — |
| Gemma 3 | 27B | GQA + QK-Norm, 5:1 sliding-window/global attention | 128,000 |
| Mistral Small 3.1 | 24B | Latency-focused, no sliding-window | 128,000 |
| Qwen3 | 32B | GQA + QK-Norm, dense reference | 128,000 |
| Qwen3 | 8B | Dense baseline, 8 KV heads | 128,000 |
| Qwen3 | 4B | Compact dense, 151k vocab | 32,768 |
| SmolLM3 | 3B | Periodic NoPE layers (omits RoPE every 4th layer) | — |
| Gemma 3 | 270M | Tiny variant, local-global attention at toy scale | — |
| OLMo 3 | 7B | Post-norm, MHA, 3:1 sliding-window/global | 65,536 |
| OLMo 3 | 32B | GQA + QK-Norm, selective YaRN | — |
| Nanbeige 4.1 | 3B | On-device, no input embedding tie-down | — |
| Tiny Aya | 3.35B | Cohere multilingual, parallel transformer block | — |

### Sparse MoE Architectures

| Model | Total / Active | Key Features | Context |
|-------|---------------|-------------|---------|
| DeepSeek V3 | 671B / 37B | Dense prefix + shared expert, MLA attention | 128,000 |
| DeepSeek R1 | 671B / 37B | Reasoning-tuned V3 variant | 128,000 |
| Kimi K2 | 1T | DeepSeek V3 recipe scaled up, more experts | 128,000 |
| GLM-4.5 | 355B / 32B | 3 dense layers before MoE routing, shared expert | — |
| GPT-OSS | 120B / 3.6B | Alternating sliding-window and global attention | — |
| GPT-OSS | 20B / 3.6B | Wider/shallower, attention bias + sink mechanisms | — |
| Grok 2.5 | 270B | Always-on SwiGLU path (shared expert behavior) | 131,072 |
| Qwen3 | 235B / 22B | No shared expert, close to DeepSeek V3 | — |
| GLM-4.7 | 355B / 32B | Pre-MLA baseline, predecessor to GLM-4.5 | — |
| MiniMax M2 | 230B / 10B | Leaner/sparser, per-layer QK-Norm | — |
| MiniMax-M2.5 | 230B / 10B | No sliding-window or linear-attention hybrids | — |
| Qwen3 Coder Flash | 30B / 3.3B | 128 experts, 8 active per token | 256,000 |
| Step 3.5 Flash | 196B / 11B | MTP-3 during training and inference | — |
| Sarvam | 30B | Reasoning-focused, strong Indic language support | — |
| Sarvam | 105B | Switches GQA → MLA, large vocab, Indic support | — |

### Sparse Hybrid Architectures

| Model | Total / Active | Key Features | Context |
|-------|---------------|-------------|---------|
| Qwen3 Next | 80B / 3B | 3:1 Gated DeltaNet / Gated Attention | 262,144 |
| Kimi Linear | 48B / 3B | Linear-attention hybrid, transformer backbone | — |
| Qwen3.5 | 397B / 17B | Hybrid attention, former Qwen3-Next side branch → core | — |
| Nemotron 3 Nano | 30B / 3B | Most extreme transformer–state-space hybrid (Mamba-2 + MoE) | — |
| Nemotron 3 Super | 120B / 12B | Scales Nano, latent experts, native speculative decoding | — |
| Xiaomi MiMo-V2-Flash | 309B / 15B | 128-token local window, multi-token prediction | — |
| Arcee AI Trinity Large | 400B / 13B | QK-Norm, RoPE+NoPE, sandwich norm | — |
| GLM-5 | 744B / 40B | MLA + DeepSeek Sparse Attention | 202,752 |
| Ling 2.5 | 1T / 63B | Lightning Attention (replaces DeltaNet) | 256,000 |

### Source Articles

1. **The Big LLM Architecture Comparison** — Dense, MoE, MLA, and hybrid families
2. **From GPT-2 to gpt-oss** — RoPE, SwiGLU, MoE, GQA, sliding-window, RMSNorm shifts
3. **From DeepSeek V3 to V3.2** — Sparse attention changes, RL-related developments
4. **A Dream of Spring for Open-Weight LLMs** — 2026 open-weight releases (MiniMax, Qwen, Ling, Sarvam)
