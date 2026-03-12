//! 2:4 structured sparsity: per group of 4 consecutive elements along K,
//! keep the 2 largest-magnitude values. Nibble-packed indices.
//!
//! Layout: values [K/2, N] (f32), indices [K/4, N] (u8, nibble-packed: low=idx0, high=idx1).
//! 1.78x bandwidth reduction vs dense (8+1 bytes per 4 elements vs 16).

#[cfg(feature = "metal")]
use crate::metal::{GpuBuffer, GpuContext};

/// A weight tensor pruned to 2:4 structured sparsity.
///
/// For a dense weight matrix [K, N] (row-major), every group of 4 consecutive
/// rows in each column keeps only the 2 largest-abs values.
pub struct SparseTensor24 {
    /// Packed non-zero values: [K/2, N] row-major f32
    pub values: Vec<f32>,
    /// Nibble-packed indices: [K/4, N] — low nibble = idx of 1st kept value,
    /// high nibble = idx of 2nd kept value (indices 0..3 within the group of 4)
    pub indices: Vec<u8>,
    /// Original rows (K) — must be divisible by 4
    pub rows: usize,
    /// Original cols (N)
    pub cols: usize,
    #[cfg(feature = "metal")]
    pub gpu_values: Option<GpuBuffer<f32>>,
    #[cfg(feature = "metal")]
    pub gpu_indices: Option<GpuBuffer<u8>>,
}

/// Prune a dense [rows, cols] f32 matrix to 2:4 structured sparsity.
/// `rows` must be divisible by 4.
pub fn prune_to_24(data: &[f32], rows: usize, cols: usize) -> SparseTensor24 {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(rows % 4, 0, "rows must be divisible by 4 for 2:4 sparsity");

    let groups = rows / 4;
    let mut values = Vec::with_capacity(groups * 2 * cols);
    let mut indices = Vec::with_capacity(groups * cols);

    for g in 0..groups {
        let base = g * 4;
        for n in 0..cols {
            // Gather the 4 elements in this group for column n
            let mut vals = [
                (data[(base + 0) * cols + n].abs(), 0u8, data[(base + 0) * cols + n]),
                (data[(base + 1) * cols + n].abs(), 1u8, data[(base + 1) * cols + n]),
                (data[(base + 2) * cols + n].abs(), 2u8, data[(base + 2) * cols + n]),
                (data[(base + 3) * cols + n].abs(), 3u8, data[(base + 3) * cols + n]),
            ];

            // Sort descending by abs value — keep top 2
            vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let (idx0, idx1) = if vals[0].1 <= vals[1].1 {
                (vals[0].1, vals[1].1)
            } else {
                (vals[1].1, vals[0].1)
            };

            // Values stored in order of ascending index
            let v0 = data[(base + idx0 as usize) * cols + n];
            let v1 = data[(base + idx1 as usize) * cols + n];

            values.push(v0);
            values.push(v1);
            indices.push(idx0 | (idx1 << 4));
        }
    }

    SparseTensor24 {
        values,
        indices,
        rows,
        cols,
        #[cfg(feature = "metal")]
        gpu_values: None,
        #[cfg(feature = "metal")]
        gpu_indices: None,
    }
}

/// Expand a 2:4 sparse tensor back to dense [rows, cols] for verification.
pub fn densify_24(st: &SparseTensor24) -> Vec<f32> {
    let rows = st.rows;
    let cols = st.cols;
    let groups = rows / 4;
    let mut out = vec![0.0f32; rows * cols];

    for g in 0..groups {
        let base = g * 4;
        for n in 0..cols {
            let idx_byte = st.indices[g * cols + n];
            let idx0 = (idx_byte & 0x0F) as usize;
            let idx1 = ((idx_byte >> 4) & 0x0F) as usize;

            let v_base = g * cols * 2 + n * 2;
            let v0 = st.values[v_base];
            let v1 = st.values[v_base + 1];

            out[(base + idx0) * cols + n] = v0;
            out[(base + idx1) * cols + n] = v1;
        }
    }

    out
}

/// CPU scalar matmul: C[M,N] = A[M,K] @ W_sparse[K,N].
/// A is dense [M,K], W is 2:4 sparse with original shape [K,N].
pub fn matmul_sparse_24(
    a_data: &[f32],
    m: usize,
    k: usize,
    w: &SparseTensor24,
) -> Vec<f32> {
    assert_eq!(w.rows, k);
    assert_eq!(k % 4, 0);
    let n = w.cols;
    assert_eq!(a_data.len(), m * k);

    let mut out = vec![0.0f32; m * n];

    #[cfg(target_arch = "aarch64")]
    {
        crate::simd_kernels::gemm_sparse_24(
            a_data, &w.values, &w.indices, &mut out, m, n, k,
        );
        return out;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let groups = k / 4;
        for mi in 0..m {
            for g in 0..groups {
                let a_base = mi * k + g * 4;
                for ni in 0..n {
                    let idx_byte = w.indices[g * n + ni];
                    let idx0 = (idx_byte & 0x0F) as usize;
                    let idx1 = ((idx_byte >> 4) & 0x0F) as usize;

                    let v_base = g * n * 2 + ni * 2;
                    let v0 = w.values[v_base];
                    let v1 = w.values[v_base + 1];

                    out[mi * n + ni] += a_data[a_base + idx0] * v0
                                      + a_data[a_base + idx1] * v1;
                }
            }
        }
        out
    }
}

/// Upload sparse tensor data to Metal GPU.
#[cfg(feature = "metal")]
impl SparseTensor24 {
    pub fn to_gpu(&mut self, gpu: &GpuContext) {
        if self.gpu_values.is_none() {
            self.gpu_values = Some(gpu.upload(&self.values));
        }
        if self.gpu_indices.is_none() {
            self.gpu_indices = Some(gpu.upload(&self.indices));
        }
    }
}

/// GPU-accelerated 2:4 sparse matmul: C[M,N] = A_f32[M,K] @ W_sparse_24[K,N].
/// Uploads A, dispatches the sparse matmul kernel.
#[cfg(feature = "metal")]
pub fn matmul_sparse_24_gpu(
    gpu: &GpuContext,
    a_data: &[f32],
    m: usize,
    k: usize,
    w: &SparseTensor24,
) -> GpuBuffer<f32> {
    let n = w.cols;
    assert_eq!(w.rows, k);
    assert_eq!(a_data.len(), m * k);

    let a_buf = gpu.upload(a_data);
    let w_vals_buf = w.gpu_values.as_ref().expect("sparse values not on GPU");
    let w_idx_buf = w.gpu_indices.as_ref().expect("sparse indices not on GPU");
    let out_buf: GpuBuffer<f32> = gpu.alloc(m * n);

    gpu.dispatch_matmul_sparse_24(
        &a_buf, w_vals_buf, w_idx_buf, &out_buf,
        m as u32, n as u32, k as u32,
    );

    out_buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_data(n: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n);
        let mut state: u32 = 42;
        for _ in 0..n {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let f = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            data.push(f);
        }
        data
    }

    #[test]
    fn test_prune_densify_roundtrip() {
        let k = 8;
        let n = 4;
        let data = random_data(k * n);
        let st = prune_to_24(&data, k, n);
        let recovered = densify_24(&st);

        // Each group of 4 rows per column should have exactly 2 non-zeros
        for g in 0..(k / 4) {
            for col in 0..n {
                let mut nz = 0;
                for r in 0..4 {
                    if recovered[(g * 4 + r) * n + col] != 0.0 {
                        nz += 1;
                    }
                }
                assert_eq!(nz, 2, "group {g} col {col} should have exactly 2 non-zeros");
            }
        }
    }

    #[test]
    fn test_keeps_largest() {
        // Construct data where we know which 2 are largest
        let k = 4;
        let n = 1;
        let data = vec![1.0, -3.0, 2.0, -0.5]; // largest abs: -3.0 (idx 1), 2.0 (idx 2)
        let st = prune_to_24(&data, k, n);
        let recovered = densify_24(&st);

        assert_eq!(recovered[0], 0.0); // idx 0 pruned
        assert_eq!(recovered[1], -3.0); // kept
        assert_eq!(recovered[2], 2.0); // kept
        assert_eq!(recovered[3], 0.0); // idx 3 pruned
    }

    #[test]
    fn test_sparse_matmul_vs_pruned_dense() {
        let m = 8;
        let k = 16;
        let n = 12;

        let a = random_data(m * k);
        let w = random_data(k * n);

        let st = prune_to_24(&w, k, n);
        let w_pruned = densify_24(&st);

        // Dense matmul with pruned weights
        let mut expected = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += a[mi * k + ki] * w_pruned[ki * n + ni];
                }
                expected[mi * n + ni] = sum;
            }
        }

        let result = matmul_sparse_24(&a, m, k, &st);

        for i in 0..m * n {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-4,
                "i={}: got {}, expected {}, diff={}",
                i, result[i], expected[i], diff
            );
        }
    }
}
