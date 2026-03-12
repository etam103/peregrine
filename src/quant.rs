//! Int8 quantized inference: per-column symmetric weight quantization,
//! per-row dynamic activation quantization, NEON sdot matmul, Metal dequant matmul.

#[cfg(feature = "metal")]
use crate::metal::{GpuBuffer, GpuContext};

/// A weight tensor quantized to int8 with per-column symmetric scales.
///
/// `data_i8[row * cols + col]` stores the quantized value.
/// `scales[col] = max(|w[:,col]|) / 127` — no zero-point.
pub struct QuantizedTensor {
    pub data_i8: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    #[cfg(feature = "metal")]
    pub gpu_data_i8: Option<GpuBuffer<i8>>,
    #[cfg(feature = "metal")]
    pub gpu_scales: Option<GpuBuffer<f32>>,
}

/// Quantize f32 weights [rows, cols] to int8 per-column symmetric.
/// `scale[n] = max(|w[:,n]|) / 127`, `q[m,n] = round(w[m,n] / scale[n])` clamped to [-127,127].
pub fn quantize_weights(data: &[f32], rows: usize, cols: usize) -> QuantizedTensor {
    assert_eq!(data.len(), rows * cols);

    // Compute per-column absmax
    let mut col_absmax = vec![0.0f32; cols];
    for m in 0..rows {
        for n in 0..cols {
            let v = data[m * cols + n].abs();
            if v > col_absmax[n] {
                col_absmax[n] = v;
            }
        }
    }

    // Compute scales (avoid division by zero)
    let scales: Vec<f32> = col_absmax
        .iter()
        .map(|&amax| if amax == 0.0 { 1.0 } else { amax / 127.0 })
        .collect();

    // Quantize
    let mut data_i8 = vec![0i8; rows * cols];
    for m in 0..rows {
        for n in 0..cols {
            let v = data[m * cols + n] / scales[n];
            data_i8[m * cols + n] = v.round().clamp(-127.0, 127.0) as i8;
        }
    }

    QuantizedTensor {
        data_i8,
        scales,
        rows,
        cols,
        #[cfg(feature = "metal")]
        gpu_data_i8: None,
        #[cfg(feature = "metal")]
        gpu_scales: None,
    }
}

/// Dynamic per-row activation quantization: returns (quantized_i8, row_scales).
/// `scale[m] = max(|a[m,:]|) / 127`
pub fn quantize_activations(data: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(data.len(), rows * cols);

    let mut out_i8 = vec![0i8; rows * cols];
    let mut row_scales = vec![0.0f32; rows];

    #[cfg(target_arch = "aarch64")]
    {
        use crate::simd_kernels::{absmax_f32, quantize_row_i8};
        for m in 0..rows {
            let row = &data[m * cols..(m + 1) * cols];
            let amax = absmax_f32(row);
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            row_scales[m] = scale;
            quantize_row_i8(row, scale, &mut out_i8[m * cols..(m + 1) * cols]);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for m in 0..rows {
            let row = &data[m * cols..(m + 1) * cols];
            let amax = row.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            row_scales[m] = scale;
            let inv_scale = 1.0 / scale;
            for k in 0..cols {
                out_i8[m * cols + k] = (row[k] * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }

    (out_i8, row_scales)
}

/// Dequantize an int8 tensor back to f32 using per-column scales.
pub fn dequantize(qt: &QuantizedTensor) -> Vec<f32> {
    let mut out = vec![0.0f32; qt.rows * qt.cols];
    for m in 0..qt.rows {
        for n in 0..qt.cols {
            out[m * qt.cols + n] = qt.data_i8[m * qt.cols + n] as f32 * qt.scales[n];
        }
    }
    out
}

/// Int8 quantized matmul: C[M,N] = A_f32[M,K] @ W_i8[K,N], with on-the-fly activation
/// quantization and post-loop dequantization via `a_scale[m] * w_scale[n]`.
///
/// On aarch64 dispatches to the NEON sdot kernel; otherwise falls back to scalar.
pub fn matmul_quantized(
    a_data: &[f32],
    m: usize,
    k: usize,
    w: &QuantizedTensor,
) -> Vec<f32> {
    assert_eq!(w.rows, k);
    let n = w.cols;
    assert_eq!(a_data.len(), m * k);

    // Quantize activations per-row
    let (a_i8, a_scales) = quantize_activations(a_data, m, k);

    // Pre-transpose weights to column-major (transposed) for the NEON kernel:
    // b_t[n, k] = w[k, n]
    let mut b_t = vec![0i8; k * n];
    for ki in 0..k {
        for ni in 0..n {
            b_t[ni * k + ki] = w.data_i8[ki * n + ni];
        }
    }

    let mut out = vec![0.0f32; m * n];

    #[cfg(target_arch = "aarch64")]
    {
        crate::simd_kernels::gemm_i8_sdot(
            &a_i8, &b_t, &mut out, &a_scales, &w.scales, m, n, k,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Scalar fallback: accumulate i32 then dequant
        for mi in 0..m {
            for ni in 0..n {
                let mut acc: i32 = 0;
                for ki in 0..k {
                    acc += a_i8[mi * k + ki] as i32 * b_t[ni * k + ki] as i32;
                }
                out[mi * n + ni] = acc as f32 * a_scales[mi] * w.scales[ni];
            }
        }
    }

    out
}

/// Upload quantized tensor data to Metal GPU.
#[cfg(feature = "metal")]
impl QuantizedTensor {
    pub fn to_gpu(&mut self, gpu: &GpuContext) {
        if self.gpu_data_i8.is_none() {
            self.gpu_data_i8 = Some(gpu.upload(&self.data_i8));
        }
        if self.gpu_scales.is_none() {
            self.gpu_scales = Some(gpu.upload(&self.scales));
        }
    }

    pub fn from_gpu(&mut self, _gpu: &GpuContext) {
        if let Some(ref buf) = self.gpu_data_i8 {
            self.data_i8 = buf.read();
        }
        if let Some(ref buf) = self.gpu_scales {
            self.scales = buf.read();
        }
    }
}

/// GPU-accelerated int8 dequant matmul: C[M,N] = A_f32[M,K] @ W_i8[K,N].
/// Uploads A as f32, dispatches the dequant matmul kernel.
#[cfg(feature = "metal")]
pub fn matmul_quantized_gpu(
    gpu: &GpuContext,
    a_data: &[f32],
    m: usize,
    k: usize,
    w: &QuantizedTensor,
) -> GpuBuffer<f32> {
    let n = w.cols;
    assert_eq!(w.rows, k);
    assert_eq!(a_data.len(), m * k);

    let a_buf = gpu.upload(a_data);
    let w_i8_buf = w.gpu_data_i8.as_ref().expect("weights not on GPU");
    let w_scales_buf = w.gpu_scales.as_ref().expect("scales not on GPU");
    let out_buf: GpuBuffer<f32> = gpu.alloc(m * n);

    gpu.dispatch_matmul_dequant_i8(
        &a_buf, w_i8_buf, w_scales_buf, &out_buf,
        m as u32, n as u32, k as u32,
    );

    out_buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let rows = 32;
        let cols = 64;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) - (rows * cols / 2) as f32) * 0.01)
            .collect();

        let qt = quantize_weights(&data, rows, cols);
        let recovered = dequantize(&qt);

        // Per-column: max error must be <= scale = absmax / 127
        for n in 0..cols {
            let col_absmax = (0..rows)
                .map(|m| data[m * cols + n].abs())
                .fold(0.0f32, f32::max);
            let scale = if col_absmax == 0.0 { 1.0 } else { col_absmax / 127.0 };
            for m in 0..rows {
                let err = (data[m * cols + n] - recovered[m * cols + n]).abs();
                assert!(
                    err <= scale + 1e-6,
                    "col {n} row {m}: err {err} > scale {scale}"
                );
            }
        }
    }

    #[test]
    fn test_matmul_quantized_vs_f32() {
        let m = 16;
        let k = 64;
        let n = 32;

        // Random-ish data
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 7) % 100) as f32 / 100.0 - 0.5)
            .collect();

        // f32 reference matmul
        let mut expected = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += a[mi * k + ki] * w[ki * n + ni];
                }
                expected[mi * n + ni] = sum;
            }
        }

        let qt = quantize_weights(&w, k, n);
        let result = matmul_quantized(&a, m, k, &qt);

        // Check relative tolerance ~ 1%
        for i in 0..m * n {
            let diff = (result[i] - expected[i]).abs();
            let denom = expected[i].abs().max(1e-6);
            let rtol = diff / denom;
            assert!(
                rtol < 0.05 || diff < 0.01,
                "i={}: got {}, expected {}, rtol={}, diff={}",
                i, result[i], expected[i], rtol, diff
            );
        }
    }
}
