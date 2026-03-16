//! Apple AMX (Apple Matrix Extension) coprocessor access via reverse-engineered instructions.
//!
//! AMX is an undocumented matrix multiplication coprocessor on Apple Silicon.
//! Instructions are encoded as `.word` opcodes that operate on a hidden register file:
//!   - 8 X registers (512 bits / 64 bytes each = 16 f32)
//!   - 8 Y registers (512 bits / 64 bytes each = 16 f32)
//!   - 64 Z registers (64 bytes each = 4KB total Z accumulator, arranged as 64×64 bytes)
//!
//! A single `fma32` instruction computes a 16×16 outer product (256 FMAs).
//!
//! Based on corsix's reverse-engineering: <https://github.com/corsix/amx>
//!
//! WARNING: These are private, undocumented instructions. They may break on
//! future macOS versions. This module is gated behind `#[cfg(target_arch = "aarch64")]`.

#[cfg(target_arch = "aarch64")]
use std::arch::asm;

// AMX instruction encoding (from corsix/amx aarch64.h):
//
// Regular ops (LDX, LDY, STX, STY, LDZ, STZ, LDZI, STZI, EXTRX, EXTRY,
//              FMA64, FMS64, FMA32, FMS32, MAC16, FMA16, FMS16,
//              VECINT, VECFP, MATINT, MATFP, GENLUT):
//   .word (0x00201000 | (op << 5) | gpr)
//
// SET/CLR (op=17) uses a different encoding with NOP sled:
//   nop; nop; nop; .word (0x00201000 | (17 << 5) | imm5)
//   where imm5=0 for SET (enable), imm5=1 for CLR (disable)

// AMX opcode numbers (corsix canonical numbering):
//  0=LDX  1=LDY  2=STX  3=STY  4=LDZ  5=STZ  6=LDZI  7=STZI
//  8=EXTRX  9=EXTRY  10=FMA64  11=FMS64  12=FMA32  13=FMS32
// 14=MAC16  15=FMA16  16=FMS16  17=SET/CLR
// 18=VECINT  19=VECFP  20=MATINT  21=MATFP  22=GENLUT

/// Enable AMX coprocessor. Must be called before any AMX operations.
/// Uses the NOP-sled encoding: nop; nop; nop; .word (0x201000 | (17<<5) | 0)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_set() {
    asm!(
        "nop", "nop", "nop",
        ".word 0x00201220",  // 0x201000 | (17 << 5) | 0
        options(nostack, nomem)
    );
}

/// Disable AMX coprocessor. Call when done with AMX operations.
/// Uses the NOP-sled encoding: nop; nop; nop; .word (0x201000 | (17<<5) | 1)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_clr() {
    asm!(
        "nop", "nop", "nop",
        ".word 0x00201221",  // 0x201000 | (17 << 5) | 1
        options(nostack, nomem)
    );
}

/// Load 64 bytes (16 f32) from `ptr` into AMX X register `reg` (0-7).
/// Operand: ptr | (reg << 56)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_ldx(ptr: *const f32, reg: u64) {
    let operand = ptr as u64 | (reg << 56);
    // LDX = op 0: .word (0x00201000 | (0 << 5) | 0) = 0x00201000
    asm!(
        ".word 0x00201000",
        in("x0") operand,
        options(nostack)
    );
}

/// Load 64 bytes (16 f32) from `ptr` into AMX Y register `reg` (0-7).
/// Operand: ptr | (reg << 56)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_ldy(ptr: *const f32, reg: u64) {
    let operand = ptr as u64 | (reg << 56);
    // LDY = op 1: .word (0x00201000 | (1 << 5) | 0) = 0x00201020
    asm!(
        ".word 0x00201020",
        in("x0") operand,
        options(nostack)
    );
}

/// Store 64 bytes from Z accumulator row `row` to `ptr`.
/// Operand: ptr | (row << 56)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_stz(ptr: *mut f32, row: u64) {
    let operand = ptr as u64 | (row << 56);
    // STZ = op 5: .word (0x00201000 | (5 << 5) | 0) = 0x002010A0
    asm!(
        ".word 0x002010A0",
        in("x0") operand,
        options(nostack)
    );
}

/// Load 64 bytes into Z accumulator row `row` from `ptr`.
/// Operand: ptr | (row << 56)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_ldz(ptr: *const f32, row: u64) {
    let operand = ptr as u64 | (row << 56);
    // LDZ = op 4: .word (0x00201000 | (4 << 5) | 0) = 0x00201080
    asm!(
        ".word 0x00201080",
        in("x0") operand,
        options(nostack)
    );
}

/// FMA32: Z += X[x_off] ⊗ Y[y_off]
/// Computes 16×16 outer product of X and Y register data, accumulating into Z.
///
/// Operand encoding for fma32 (from corsix):
///   bits [9:0]   = Y offset (byte offset within Y register file, 64-byte aligned)
///   bits [19:10] = X offset (byte offset within X register file, 64-byte aligned)
///   bits [62:20] = Z row and mode flags
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_fma32(operand: u64) {
    // FMA32 = op 12: .word (0x00201000 | (12 << 5) | 0) = 0x00201180
    asm!(
        ".word 0x00201180",
        in("x0") operand,
        options(nostack)
    );
}

/// AMX matmul: C[M,N] += A[M,K] * B[K,N]
/// Uses AMX 16×16 outer product tiles.
///
/// For M=N=K=128: 8 tiles × 8 tiles × 128 k-steps = 8192 AMX FMA instructions.
/// Each FMA does 256 multiply-adds = 2M FMAs total.
#[cfg(target_arch = "aarch64")]
pub fn amx_sgemm(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    assert!(m % 16 == 0 && n % 16 == 0, "AMX matmul requires M,N divisible by 16");

    // Zero output
    for v in c.iter_mut() { *v = 0.0; }

    unsafe {
        amx_set();

        // Tile over M and N in 16-element blocks
        for i_tile in (0..m).step_by(16) {
            for j_tile in (0..n).step_by(16) {
                // Zero the 16 Z rows that FMA32 writes to (stride-4: 0, 4, 8, ..., 60)
                let zeros = [0.0f32; 16];
                for j in 0..16u64 {
                    amx_ldz(zeros.as_ptr(), j * 4);
                }

                // Accumulate K dimension
                // FMA32 outer product: Z[j*4][i] += X[i] * Y[j]
                // We want C[i_tile+j, j_tile+i] = sum_p A[i_tile+j, p] * B[p, j_tile+i]
                // So load A column into Y (Y[j] = A[i_tile+j, p]) → output row j
                //    load B row into X   (X[i] = B[p, j_tile+i]) → output col i
                for p in 0..k {
                    // Load column p of A tile into Y register
                    // A[i_tile+r, p] for r=0..15 — gather from non-contiguous rows
                    let mut a_col = [0.0f32; 16];
                    for r in 0..16 {
                        a_col[r] = a[(i_tile + r) * k + p];
                    }
                    amx_ldy(a_col.as_ptr(), 0);

                    // Load row p of B tile into X register (contiguous)
                    // B[p, j_tile..j_tile+16]
                    amx_ldx(b.as_ptr().add(p * n + j_tile), 0);

                    // Outer product: Z[j*4][i] += Y[j] * X[i]
                    amx_fma32(0);
                }

                // Store Z accumulator to C (Z rows are stride-4 for f32)
                // Z[j*4] -> output row j
                for j in 0..16u64 {
                    amx_stz(c.as_mut_ptr().add((i_tile + j as usize) * n + j_tile), j * 4);
                }
            }
        }

        amx_clr();
    }
}

/// Quick smoke test: try amx_set + amx_clr to see if AMX is accessible.
/// Returns Ok(()) if AMX is available, or a description of the failure.
#[cfg(target_arch = "aarch64")]
pub fn amx_probe() -> Result<(), &'static str> {
    unsafe {
        amx_set();
        amx_clr();
    }
    Ok(())
}

/// Benchmark AMX sgemm vs cblas_sgemm. Returns (amx_us, cblas_us) for the given size.
#[cfg(target_arch = "aarch64")]
pub fn amx_bench(n: usize, iters: usize) -> (f64, f64) {
    use std::time::Instant;

    extern "C" {
        fn cblas_sgemm(
            order: i32, transa: i32, transb: i32,
            m: i32, n: i32, k: i32,
            alpha: f32, a: *const f32, lda: i32,
            b: *const f32, ldb: i32,
            beta: f32, c: *mut f32, ldc: i32,
        );
    }

    let a: Vec<f32> = (0..n*n).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..n*n).map(|i| (i as f32) * 0.001).collect();
    let mut c = vec![0.0f32; n * n];

    // Warmup
    amx_sgemm(&a, &b, &mut c, n, n, n);
    unsafe { cblas_sgemm(101, 111, 111, n as i32, n as i32, n as i32, 1.0, a.as_ptr(), n as i32, b.as_ptr(), n as i32, 0.0, c.as_mut_ptr(), n as i32); }

    // AMX timing
    let start = Instant::now();
    for _ in 0..iters {
        amx_sgemm(&a, &b, &mut c, n, n, n);
    }
    let amx_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // cblas timing
    let start = Instant::now();
    for _ in 0..iters {
        unsafe { cblas_sgemm(101, 111, 111, n as i32, n as i32, n as i32, 1.0, a.as_ptr(), n as i32, b.as_ptr(), n as i32, 0.0, c.as_mut_ptr(), n as i32); }
    }
    let cblas_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;

    (amx_us, cblas_us)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore] // AMX requires Apple Silicon with AMX support — SIGILL on CI runners
    fn test_amx_probe() {
        // This is the first test — just enable/disable AMX.
        // If AMX is blocked, this will SIGILL.
        match super::amx_probe() {
            Ok(()) => eprintln!("AMX probe: SUCCESS — AMX is accessible!"),
            Err(e) => panic!("AMX probe failed: {}", e),
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore]
    fn test_amx_sgemm_identity() {
        use super::amx_sgemm;

        let n = 16;
        // A = identity, B = [1,2,...,16] repeated
        let mut a = vec![0.0f32; n * n];
        for i in 0..n { a[i * n + i] = 1.0; }
        let b: Vec<f32> = (0..n*n).map(|i| (i % n + 1) as f32).collect();
        let mut c = vec![0.0f32; n * n];

        amx_sgemm(&a, &b, &mut c, n, n, n);

        // C should equal B (identity * B = B)
        for i in 0..n*n {
            assert!((c[i] - b[i]).abs() < 1e-3, "mismatch at {}: {} vs {}", i, c[i], b[i]);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore]
    fn test_amx_sgemm_128() {
        use super::amx_sgemm;

        let n = 128;
        let a: Vec<f32> = (0..n*n).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..n*n).map(|i| (i as f32) * 0.001).collect();
        let mut c_amx = vec![0.0f32; n * n];
        let mut c_ref = vec![0.0f32; n * n];

        amx_sgemm(&a, &b, &mut c_amx, n, n, n);

        // Reference naive matmul
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..n { sum += a[i * n + p] * b[p * n + j]; }
                c_ref[i * n + j] = sum;
            }
        }

        // Check parity
        let max_err = c_amx.iter().zip(c_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.1, "AMX matmul max error: {} (expected < 0.1)", max_err);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore]
    fn test_amx_bench() {
        for &n in &[16, 32, 64, 128, 256] {
            let iters = if n <= 64 { 10000 } else if n <= 128 { 1000 } else { 100 };
            let (amx_us, cblas_us) = super::amx_bench(n, iters);
            let gflops_amx = 2.0 * (n as f64).powi(3) / amx_us / 1e3;
            let gflops_cblas = 2.0 * (n as f64).powi(3) / cblas_us / 1e3;
            eprintln!(
                "  {}x{}: AMX={:.1}µs ({:.1} GFLOPS) cblas={:.1}µs ({:.1} GFLOPS) ratio={:.2}x",
                n, n, amx_us, gflops_amx, cblas_us, gflops_cblas, amx_us / cblas_us
            );
        }
    }
}
