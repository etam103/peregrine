//! Basic Metal GPU pipeline tests.
//! Run with: cargo test --test metal_basics --features metal

#![cfg(feature = "metal")]

use peregrine::metal::{GpuContext, GpuBuffer};

#[test]
fn metal_device_info() {
    let gpu = GpuContext::new().expect("Metal device");
    let name = gpu.device_name();
    println!("GPU: {}", name);
    println!("Unified memory: {}", gpu.has_unified_memory());
    println!("Max working set: {} MB", gpu.max_working_set_size() / 1024 / 1024);
    assert!(!name.is_empty());
    assert!(gpu.has_unified_memory()); // Always true on Apple Silicon
}

#[test]
fn metal_buffer_roundtrip() {
    let gpu = GpuContext::new().unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let buf = gpu.upload(&data);
    let back = buf.read();
    assert_eq!(data, back);
}

#[test]
fn metal_add() {
    let gpu = GpuContext::new().unwrap();
    let a = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0]);
    let b = gpu.upload(&[10.0f32, 20.0, 30.0, 40.0]);
    let out: GpuBuffer<f32> = gpu.alloc(4);
    gpu.dispatch_binary("add_f32", &a, &b, &out);
    gpu.sync();
    assert_eq!(out.read(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn metal_sub() {
    let gpu = GpuContext::new().unwrap();
    let a = gpu.upload(&[10.0f32, 20.0, 30.0, 40.0]);
    let b = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0]);
    let out: GpuBuffer<f32> = gpu.alloc(4);
    gpu.dispatch_binary("sub_f32", &a, &b, &out);
    gpu.sync();
    assert_eq!(out.read(), vec![9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn metal_mul() {
    let gpu = GpuContext::new().unwrap();
    let a = gpu.upload(&[2.0f32, 3.0, 4.0, 5.0]);
    let b = gpu.upload(&[10.0f32, 10.0, 10.0, 10.0]);
    let out: GpuBuffer<f32> = gpu.alloc(4);
    gpu.dispatch_binary("mul_f32", &a, &b, &out);
    gpu.sync();
    assert_eq!(out.read(), vec![20.0, 30.0, 40.0, 50.0]);
}

#[test]
fn metal_relu() {
    let gpu = GpuContext::new().unwrap();
    let a = gpu.upload(&[-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let out: GpuBuffer<f32> = gpu.alloc(6);
    gpu.dispatch_unary("relu_f32", &a, &out);
    gpu.sync();
    assert_eq!(out.read(), vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn metal_exp() {
    let gpu = GpuContext::new().unwrap();
    let a = gpu.upload(&[0.0f32, 1.0, 2.0]);
    let out: GpuBuffer<f32> = gpu.alloc(3);
    gpu.dispatch_unary("exp_f32", &a, &out);
    gpu.sync();
    let result = out.read();
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::E).abs() < 1e-5);
    assert!((result[2] - std::f32::consts::E.powi(2)).abs() < 1e-4);
}

#[test]
fn metal_matmul() {
    let gpu = GpuContext::new().unwrap();
    // A = [[1, 2], [3, 4]]  (2x2)
    // B = [[5, 6], [7, 8]]  (2x2)
    // C = [[19, 22], [43, 50]]
    let a = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0]);
    let b = gpu.upload(&[5.0f32, 6.0, 7.0, 8.0]);
    let c: GpuBuffer<f32> = gpu.alloc(4);
    gpu.dispatch_matmul(&a, &b, &c, None, 2, 2, 2, false, false, false);
    gpu.sync();
    assert_eq!(c.read(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn metal_matmul_rect() {
    let gpu = GpuContext::new().unwrap();
    // A = [[1, 2, 3]]  (1x3)
    // B = [[1, 0], [0, 1], [1, 1]]  (3x2)
    // C = [[4, 5]]
    let a = gpu.upload(&[1.0f32, 2.0, 3.0]);
    let b = gpu.upload(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let c: GpuBuffer<f32> = gpu.alloc(2);
    gpu.dispatch_matmul(&a, &b, &c, None, 1, 2, 3, false, false, false);
    gpu.sync();
    assert_eq!(c.read(), vec![4.0, 5.0]);
}

#[test]
fn metal_matmul_fused_bias_relu() {
    let gpu = GpuContext::new().unwrap();
    // A = [[-1, 2], [3, -4]]
    // B = [[1, 0], [0, 1]]  (identity)
    // bias = [10, 10]
    // relu(A + bias) = relu([[-1+10, 2+10], [3+10, -4+10]]) = [[9, 12], [13, 6]]
    let a = gpu.upload(&[-1.0f32, 2.0, 3.0, -4.0]);
    let b = gpu.upload(&[1.0f32, 0.0, 0.0, 1.0]);
    let bias = gpu.upload(&[10.0f32, 10.0]);
    let c: GpuBuffer<f32> = gpu.alloc(4);
    gpu.dispatch_matmul(&a, &b, &c, Some(&bias), 2, 2, 2, true, false, false);
    gpu.sync();
    assert_eq!(c.read(), vec![9.0, 12.0, 13.0, 6.0]);
}

#[test]
fn metal_softmax() {
    let gpu = GpuContext::new().unwrap();
    // softmax([1, 2, 3]) = [0.0900, 0.2447, 0.6652]
    let input = gpu.upload(&[1.0f32, 2.0, 3.0]);
    let output: GpuBuffer<f32> = gpu.alloc(3);
    gpu.dispatch_softmax(&input, &output, 1, 3);
    gpu.sync();
    let result = output.read();
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1, got {sum}");
    assert!((result[0] - 0.0900).abs() < 0.001);
    assert!((result[1] - 0.2447).abs() < 0.001);
    assert!((result[2] - 0.6652).abs() < 0.001);
}

#[test]
fn metal_large_add() {
    let gpu = GpuContext::new().unwrap();
    let n = 100_000;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let a = gpu.upload(&a_data);
    let b = gpu.upload(&b_data);
    let out: GpuBuffer<f32> = gpu.alloc(n);
    gpu.dispatch_binary("add_f32", &a, &b, &out);
    gpu.sync();
    let result = out.read();
    for i in 0..n {
        assert_eq!(result[i], n as f32, "mismatch at index {i}");
    }
}
