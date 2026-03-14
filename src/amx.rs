//! Apple AMX (Apple Matrix Extension) coprocessor access via reverse-engineered instructions.
//!
//! AMX is an undocumented matrix multiplication coprocessor on Apple Silicon.
//! Instructions are encoded as `.word` opcodes that write to a hidden register file:
//!   - 8 X registers (512 bits each = 16 floats)
//!   - 8 Y registers (512 bits each = 16 floats)
//!   - Z accumulator (64x64 bytes = 16x16 floats)
//!
//! A single `fma32` instruction computes a 16x16 outer product (256 FMAs).
//!
//! Based on Dougall Johnson's reverse-engineering: https://github.com/corsix/amx
//!
//! WARNING: These are private, undocumented instructions. They may break on
//! future macOS versions. This module is gated behind `#[cfg(target_arch = "aarch64")]`.

#[cfg(target_arch = "aarch64")]
use std::arch::asm;

/// AMX instruction encoding: .word (0x00201000 | (op << 5) | gpr)
/// We always use x0 (gpr=0) as the operand register.
///
/// Op codes:
///  0 = set (enable AMX)
///  1 = clr (disable AMX)
///  2 = ldx (load 64 bytes to X register)
///  3 = ldy (load 64 bytes to Y register)
///  6 = stz (store from Z accumulator)
///  7 = ldz (load to Z accumulator)
/// 12 = fma32 (f32 outer product: Z += X ⊗ Y)

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_op(op: u32, operand: u64) {
    // Encode: .word (0x00201000 | (op << 5) | 0)
    // The operand goes in x0
    match op {
        0 => asm!(".word (0x00201000 | (0 << 5) | 0)", in("x0") operand, options(nostack)),
        1 => asm!(".word (0x00201000 | (1 << 5) | 0)", in("x0") operand, options(nostack)),
        2 => asm!(".word (0x00201000 | (2 << 5) | 0)", in("x0") operand, options(nostack)),
        3 => asm!(".word (0x00201000 | (3 << 5) | 0)", in("x0") operand, options(nostack)),
        6 => asm!(".word (0x00201000 | (6 << 5) | 0)", in("x0") operand, options(nostack)),
        7 => asm!(".word (0x00201000 | (7 << 5) | 0)", in("x0") operand, options(nostack)),
        12 => asm!(".word (0x00201000 | (12 << 5) | 0)", in("x0") operand, options(nostack)),
        _ => panic!("unknown AMX op: {}", op),
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_set() { amx_op(0, 0); }

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_clr() { amx_op(1, 0); }

/// Load 64 bytes (16 f32) from `ptr` into AMX X register `reg` (0-7).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_ldx(ptr: *const f32, reg: u64) {
    amx_op(2, ptr as u64 | (reg << 56));
}

/// Load 64 bytes (16 f32) from `ptr` into AMX Y register `reg` (0-7).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_ldy(ptr: *const f32, reg: u64) {
    amx_op(3, ptr as u64 | (reg << 56));
}

/// Store 64 bytes from Z accumulator row to `ptr`.
/// `operand` encodes which row and destination.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_stz(ptr: *mut f32, row: u64) {
    amx_op(6, ptr as u64 | (row << 56));
}

/// Load 64 bytes into Z accumulator row from `ptr`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_ldz(ptr: *const f32, row: u64) {
    amx_op(7, ptr as u64 | (row << 56));
}

/// FMA32: Z[z_row] += X[x_offset] ⊗ Y[y_offset]
/// Computes 16x16 outer product of X and Y registers, accumulating into Z.
///
/// Operand encoding for fma32:
///   bits [19:0]  = Y offset (which Y register pair * 64)
///   bits [29:20] = X offset (which X register pair * 64)
///   bits [46:42] = Z row
///   bit [27]     = skip Y load (use cached)
///   bit [28]     = skip X load (use cached)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn amx_fma32(x_off: u64, y_off: u64, z_row: u64) {
    let operand = y_off | (x_off << 20) | (z_row << 42);
    amx_op(12, operand);
}

/// AMX matmul: C[M,N] += A[M,K] * B[K,N]
/// Uses AMX outer product: processes 16 rows of A × 16 cols of B at once.
///
/// For M=N=K=128: 8 tiles × 8 tiles × 128 k-steps = 8192 AMX FMA instructions.
/// Each FMA does 256 multiply-adds = 2M FMAs total. At ~1 FMA/cycle this is ~2.5µs.
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
                // Zero Z accumulator by loading zeros
                let zeros = [0.0f32; 16];
                for z_row in 0..16u64 {
                    amx_ldz(zeros.as_ptr(), z_row);
                }

                // Accumulate K dimension
                for p in 0..k {
                    // Load column p of A tile (16 elements from 16 consecutive rows)
                    // A is row-major: A[i_tile + r, p] = a[(i_tile + r) * k + p]
                    // We need 16 elements from different rows — AMX wants contiguous data
                    // Pack into a temporary buffer
                    let mut a_col = [0.0f32; 16];
                    for r in 0..16 {
                        a_col[r] = a[(i_tile + r) * k + p];
                    }
                    amx_ldx(a_col.as_ptr(), 0);

                    // Load row p of B tile (16 contiguous elements)
                    // B[p, j_tile..j_tile+16] = b[p * n + j_tile..]
                    amx_ldy(b.as_ptr().add(p * n + j_tile), 0);

                    // Outer product: Z += X ⊗ Y
                    amx_fma32(0, 0, 0);
                }

                // Store Z accumulator to C
                for z_row in 0..16u64 {
                    amx_stz(c.as_mut_ptr().add((i_tile + z_row as usize) * n + j_tile), z_row);
                }
            }
        }

        amx_clr();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_arch = "aarch64")]
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
}
