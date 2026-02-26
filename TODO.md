1. Optimize training speed
2. Optimize inference speed
3. Memory optimizations

What BarraCUDA does:

CUDA → AMD is a relatively "easier" translation because both NVIDIA and AMD GPUs use a SIMT (Single Instruction Multiple Threads) execution model. The programming concepts map fairly directly — warps ≈ wavefronts, shared memory ≈ LDS, etc.

Why CUDA → Metal is harder:

Metal uses a fundamentally different shading language (MSL, which is C++ based) and a different GPU architecture (Apple's unified memory, tile-based rendering, TBDR pipeline). The execution model doesn't map as cleanly.
Apple doesn't publish their GPU ISA publicly. BarraCUDA works because AMD's GFX11 ISA is documented (even if poorly). Apple's GPU machine code is completely undocumented — you'd be reverse engineering the instruction set.
There's no equivalent of AMD's .hsaco binary format that you can just load and run. Metal uses its own compiler pipeline (metallib format), which is opaque.

What you could realistically do instead:

CUDA → Metal Shading Language (source-to-source) — translate .cu files to .metal source, then let Apple's Metal compiler handle the rest. This is basically what HIP does for AMD (source translation). This is the most practical approach.
Write Metal compute shaders directly from Rust — skip the CUDA compatibility layer entirely and write Metal kernels optimized for M-series. This is probably smarter for your goal of beating MLX, since you'd be writing native code rather than going through a translation layer.
Target Apple's IR (AIR) — Metal compiles to an intermediate representation called AIR (Apple IR, based on LLVM IR). You could potentially emit AIR directly, but it's underdocumented and fragile across OS versions.