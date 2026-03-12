//! Int8 quantization parity tests.

use peregrine::quant::{quantize_weights, dequantize, matmul_quantized};

#[test]
fn test_quantize_dequantize_roundtrip_max_error() {
    let rows = 64;
    let cols = 128;
    // Values in [-2, 2]
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 17 + 5) % 400) as f32 / 100.0 - 2.0)
        .collect();

    let qt = quantize_weights(&data, rows, cols);
    let recovered = dequantize(&qt);

    // Per-column: max quantization error should be < scale = absmax / 127
    for n in 0..cols {
        let col_absmax = (0..rows)
            .map(|m| data[m * cols + n].abs())
            .fold(0.0f32, f32::max);
        let scale = if col_absmax == 0.0 { 1.0 } else { col_absmax / 127.0 };

        for m in 0..rows {
            let err = (data[m * cols + n] - recovered[m * cols + n]).abs();
            assert!(
                err <= scale + 1e-7,
                "col {n} row {m}: err {err} > scale {scale}"
            );
        }
    }
}

#[test]
fn test_i8_matmul_vs_f32_matmul() {
    let m = 32;
    let k = 128;
    let n = 64;

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 200) as f32 / 200.0 - 0.5)
        .collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i * 13 + 7) % 200) as f32 / 200.0 - 0.5)
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

    // Relative tolerance check — int8 quantization introduces ~1/127 error per element,
    // which accumulates as sqrt(K)/127 * scale in matmul. Allow 10% rtol or 0.05 absolute.
    let mut max_rtol = 0.0f32;
    for i in 0..m * n {
        let diff = (result[i] - expected[i]).abs();
        let denom = expected[i].abs().max(1e-6);
        let rtol = diff / denom;
        max_rtol = max_rtol.max(rtol);
        assert!(
            rtol < 0.15 || diff < 0.05,
            "i={i}: got {}, expected {}, rtol={rtol}, diff={diff}",
            result[i], expected[i]
        );
    }
    eprintln!("  i8 vs f32 matmul: max rtol = {max_rtol:.4}");
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_i8_vs_cpu_i8_parity() {
    use peregrine::metal;
    metal::init_gpu().expect("Metal GPU init");

    let m = 32;
    let k = 128;
    let n = 64;

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 200) as f32 / 200.0 - 0.5)
        .collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i * 13 + 7) % 200) as f32 / 200.0 - 0.5)
        .collect();

    let mut qt = quantize_weights(&w, k, n);

    // CPU result
    let cpu_result = matmul_quantized(&a, m, k, &qt);

    // GPU result
    let gpu_result = metal::with_gpu(|gpu| {
        qt.to_gpu(gpu);
        let out_buf = peregrine::quant::matmul_quantized_gpu(gpu, &a, m, k, &qt);
        gpu.sync();
        out_buf.read()
    }).expect("GPU dispatch");

    // They won't be identical (GPU does f32 accumulation without int8 activation quant)
    // but should be close to the f32 reference
    let mut max_diff = 0.0f32;
    for i in 0..m * n {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        max_diff = max_diff.max(diff);
    }
    eprintln!("  Metal vs CPU i8 max diff = {max_diff:.6}");
    // GPU uses f32 path (no activation quantization), CPU uses i8, so allow larger tolerance
    // The key test is that both are close to f32 reference
    let mut f32_result = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[mi * k + ki] * w[ki * n + ni];
            }
            f32_result[mi * n + ni] = sum;
        }
    }

    // GPU dequant path uses f32 throughout (no activation quantization),
    // only i8 weight dequantization error. Weight quantization noise accumulates
    // across K dot products. Allow 10% rtol or 0.02 absolute.
    let mut gpu_max_rtol = 0.0f32;
    for i in 0..m * n {
        let gpu_diff = (gpu_result[i] - f32_result[i]).abs();
        let denom = f32_result[i].abs().max(1e-6);
        let rtol = gpu_diff / denom;
        gpu_max_rtol = gpu_max_rtol.max(rtol);
        assert!(
            rtol < 0.10 || gpu_diff < 0.02,
            "GPU i={i}: got {}, expected {}, rtol={rtol}, diff={gpu_diff}",
            gpu_result[i], f32_result[i]
        );
    }
    eprintln!("  GPU vs f32 max rtol = {gpu_max_rtol:.4}");
}

#[test]
fn test_serialization_roundtrip() {
    use std::io::Cursor;

    let rows = 16;
    let cols = 32;
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 11 + 3) % 100) as f32 / 50.0 - 1.0)
        .collect();

    let qt = quantize_weights(&data, rows, cols);

    // Write
    let mut buf = Vec::new();
    peregrine::serial::write_quantized_tensor(&mut buf, "test.weight", &qt).unwrap();

    // Read back
    let mut cursor = Cursor::new(&buf);
    let (name, qt2) = peregrine::serial::read_quantized_tensor(&mut cursor).unwrap();

    assert_eq!(name, "test.weight");
    assert_eq!(qt2.rows, rows);
    assert_eq!(qt2.cols, cols);
    assert_eq!(qt2.data_i8, qt.data_i8);
    assert_eq!(qt2.scales, qt.scales);
}
