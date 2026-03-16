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

**Reference:** `~/Desktop/ANE/` — reverse-engineered private ANE APIs (Obj-C), demonstrates Stories110M training on ANE.

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
