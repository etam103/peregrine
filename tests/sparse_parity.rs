//! Integration tests for 2:4 structured sparsity.
//! Run with: cargo test --test sparse_parity --features metal

use peregrine::sparse::{prune_to_24, densify_24, matmul_sparse_24};

fn random_data(n: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    let mut state: u32 = 54321;
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let f = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        data.push(f);
    }
    data
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn sparse_vs_dense_matmul_parity() {
    for &(m, k, n) in &[(8, 16, 12), (1, 4, 1), (32, 64, 48), (16, 256, 32)] {
        let a = random_data(m * k);
        let w = random_data(k * n);

        let st = prune_to_24(&w, k, n);
        let w_pruned = densify_24(&st);

        // Dense matmul with pruned weights (reference)
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
        let err = max_abs_error(&result, &expected);
        assert!(
            err < 1e-3,
            "sparse vs dense parity failed for M={m} K={k} N={n}: max_err={err}"
        );
    }
}

#[test]
#[should_panic(expected = "rows must be divisible by 4")]
fn k_not_multiple_of_4_panics() {
    let data = vec![1.0; 6 * 4]; // K=6 is not divisible by 4
    prune_to_24(&data, 6, 4);
}

#[cfg(feature = "metal")]
#[test]
fn gpu_sparse_vs_cpu_sparse() {
    use peregrine::metal::GpuContext;
    use peregrine::sparse::matmul_sparse_24_gpu;

    let gpu = GpuContext::new().unwrap();

    for &(m, k, n) in &[(8, 16, 12), (32, 64, 48), (16, 256, 32)] {
        let a = random_data(m * k);
        let w = random_data(k * n);

        let mut st = prune_to_24(&w, k, n);

        // CPU result
        let cpu_result = matmul_sparse_24(&a, m, k, &st);

        // GPU result
        st.to_gpu(&gpu);
        let gpu_buf = matmul_sparse_24_gpu(&gpu, &a, m, k, &st);
        gpu.sync();
        let gpu_result = gpu_buf.read();

        let err = max_abs_error(&cpu_result, &gpu_result);
        assert!(
            err < 1e-3,
            "GPU vs CPU sparse parity failed for M={m} K={k} N={n}: max_err={err}"
        );
    }
}

#[test]
fn serialization_roundtrip() {
    use peregrine::serial::{write_sparse_tensor_24, read_sparse_tensor_24};

    let k = 16;
    let n = 8;
    let data = random_data(k * n);
    let st = prune_to_24(&data, k, n);

    // Serialize
    let mut buf: Vec<u8> = Vec::new();
    write_sparse_tensor_24(&mut buf, "test_sparse", &st).unwrap();

    // Deserialize
    let mut cursor = std::io::Cursor::new(&buf);
    let (name, loaded) = read_sparse_tensor_24(&mut cursor).unwrap();

    assert_eq!(name, "test_sparse");
    assert_eq!(loaded.rows, st.rows);
    assert_eq!(loaded.cols, st.cols);
    assert_eq!(loaded.values, st.values);
    assert_eq!(loaded.indices, st.indices);
}
