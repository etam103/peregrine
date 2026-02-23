//! Performance benchmarks for Peregrine tensor operations.
//!
//! CPU:  cargo bench
//! GPU:  cargo bench --features metal
//!
//! Results are saved to target/criterion/ with HTML reports.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use peregrine::tensor::Tensor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_tensor(shape: &[usize]) -> Tensor {
    Tensor::randn(shape, false)
}

fn random_tensor_grad(shape: &[usize]) -> Tensor {
    Tensor::randn(shape, true)
}

// ---------------------------------------------------------------------------
// MatMul benchmarks
// ---------------------------------------------------------------------------

fn bench_matmul_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_cpu");
    for size in [128, 256, 512, 1024] {
        let flops = 2 * size * size * size; // 2*M*N*K for matmul
        group.throughput(Throughput::Elements(flops as u64));
        group.bench_with_input(
            BenchmarkId::new("square", size),
            &size,
            |b, &size| {
                let a = random_tensor(&[size, size]);
                let w = random_tensor(&[size, size]);
                b.iter(|| black_box(a.matmul(&w)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Element-wise benchmarks
// ---------------------------------------------------------------------------

fn bench_elementwise_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_cpu");
    let n = 100_000;
    group.throughput(Throughput::Elements(n as u64));

    let a = random_tensor(&[n]);
    let b = random_tensor(&[n]);

    group.bench_function("add", |bench| bench.iter(|| black_box(a.add(&b))));
    group.bench_function("mul", |bench| bench.iter(|| black_box(a.mul(&b))));
    group.bench_function("exp", |bench| bench.iter(|| black_box(a.exp())));
    group.bench_function("relu", |bench| bench.iter(|| black_box(a.relu())));
    group.bench_function("tanh", |bench| bench.iter(|| black_box(a.tanh())));
    group.bench_function("sigmoid", |bench| bench.iter(|| black_box(a.sigmoid())));

    group.finish();
}

fn bench_elementwise_large_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_large_cpu");
    let n = 1_000_000;
    group.throughput(Throughput::Elements(n as u64));

    let a = random_tensor(&[n as usize]);
    let b = random_tensor(&[n as usize]);

    group.bench_function("add", |bench| bench.iter(|| black_box(a.add(&b))));
    group.bench_function("mul", |bench| bench.iter(|| black_box(a.mul(&b))));
    group.bench_function("exp", |bench| bench.iter(|| black_box(a.exp())));

    group.finish();
}

// ---------------------------------------------------------------------------
// Softmax benchmarks
// ---------------------------------------------------------------------------

fn bench_softmax_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_cpu");
    for seq_len in [128, 512, 2048] {
        let batch = 8;
        group.throughput(Throughput::Elements((batch * seq_len) as u64));
        group.bench_with_input(
            BenchmarkId::new("batch8", seq_len),
            &seq_len,
            |b, &seq_len| {
                let x = random_tensor(&[batch, seq_len]);
                b.iter(|| black_box(x.softmax(-1)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Forward pass benchmarks
// ---------------------------------------------------------------------------

fn bench_mlp_forward_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_forward_cpu");

    // 784 -> 128 -> 64 -> 10 (MNIST-like MLP)
    let w1 = random_tensor(&[784, 128]);
    let b1 = random_tensor(&[1, 128]);
    let w2 = random_tensor(&[128, 64]);
    let b2 = random_tensor(&[1, 64]);
    let w3 = random_tensor(&[64, 10]);
    let b3 = random_tensor(&[1, 10]);

    for batch in [1, 32, 64, 256] {
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(
            BenchmarkId::new("mnist_mlp", batch),
            &batch,
            |b, &batch| {
                let x = random_tensor(&[batch, 784]);
                b.iter(|| {
                    let h1 = x.matmul(&w1).add_bias(&b1).relu();
                    let h2 = h1.matmul(&w2).add_bias(&b2).relu();
                    let out = h2.matmul(&w3).add_bias(&b3);
                    black_box(out)
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Training step (forward + backward + optimizer)
// ---------------------------------------------------------------------------

fn bench_training_step_cpu(c: &mut Criterion) {
    use peregrine::nn;
    use peregrine::optim::Adam;

    let mut group = c.benchmark_group("training_step_cpu");
    group.sample_size(20); // fewer samples since each iter is expensive

    let w1 = random_tensor_grad(&[784, 128]);
    let b1 = Tensor::zeros(&[1, 128], true);
    let w2 = random_tensor_grad(&[128, 64]);
    let b2 = Tensor::zeros(&[1, 64], true);
    let w3 = random_tensor_grad(&[64, 10]);
    let b3 = Tensor::zeros(&[1, 10], true);

    let batch = 64;
    let targets: Vec<usize> = (0..batch).map(|i| i % 10).collect();

    group.bench_function("mnist_batch64", |b| {
        let mut opt = Adam::new(
            vec![w1.clone(), b1.clone(), w2.clone(), b2.clone(), w3.clone(), b3.clone()],
            1e-3,
        );
        b.iter(|| {
            let x = random_tensor(&[batch, 784]);
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
            let logits = h2.matmul(&w3).add_bias(&b3);
            let loss = nn::cross_entropy_loss(&logits, &targets);
            loss.backward();
            opt.step();
            opt.zero_grad();
            black_box(loss)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Metal GPU benchmarks (only compiled with --features metal)
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
fn bench_matmul_gpu(c: &mut Criterion) {
    use peregrine::metal::GpuContext;

    let gpu = GpuContext::new().expect("Metal device");
    let mut group = c.benchmark_group("matmul_gpu");

    for size in [128, 256, 512, 1024] {
        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.001) % 1.0).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.002) % 1.0).collect();
        let a = gpu.upload(&a_data);
        let b = gpu.upload(&b_data);
        let c_buf = gpu.alloc(size * size);

        group.bench_with_input(
            BenchmarkId::new("square", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    gpu.dispatch_matmul(&a, &b, &c_buf, None, size as u32, size as u32, size as u32, false);
                    black_box(&c_buf);
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "metal")]
fn bench_elementwise_gpu(c: &mut Criterion) {
    use peregrine::metal::GpuContext;

    let gpu = GpuContext::new().expect("Metal device");
    let mut group = c.benchmark_group("elementwise_gpu");
    let n = 1_000_000;
    group.throughput(Throughput::Elements(n as u64));

    let a_data: Vec<f32> = (0..n as usize).map(|i| (i as f32 * 0.001) % 2.0 - 1.0).collect();
    let b_data: Vec<f32> = (0..n as usize).map(|i| (i as f32 * 0.002) % 2.0 - 1.0).collect();
    let a = gpu.upload(&a_data);
    let b = gpu.upload(&b_data);
    let out = gpu.alloc(n as usize);

    group.bench_function("add", |bench| {
        bench.iter(|| { gpu.dispatch_binary("add_f32", &a, &b, &out); black_box(&out); })
    });
    group.bench_function("mul", |bench| {
        bench.iter(|| { gpu.dispatch_binary("mul_f32", &a, &b, &out); black_box(&out); })
    });
    group.bench_function("exp", |bench| {
        bench.iter(|| { gpu.dispatch_unary("exp_f32", &a, &out); black_box(&out); })
    });

    group.finish();
}

#[cfg(feature = "metal")]
fn bench_softmax_gpu(c: &mut Criterion) {
    use peregrine::metal::GpuContext;

    let gpu = GpuContext::new().expect("Metal device");
    let mut group = c.benchmark_group("softmax_gpu");

    for seq_len in [128, 512, 2048] {
        let batch = 8usize;
        group.throughput(Throughput::Elements((batch * seq_len) as u64));

        let data: Vec<f32> = (0..batch * seq_len).map(|i| (i as f32 * 0.01) % 5.0 - 2.5).collect();
        let input = gpu.upload(&data);
        let output = gpu.alloc(batch * seq_len);

        group.bench_with_input(
            BenchmarkId::new("batch8", seq_len),
            &seq_len,
            |b, &seq_len| {
                b.iter(|| {
                    gpu.dispatch_softmax(&input, &output, batch as u32, seq_len as u32);
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

#[cfg(not(feature = "metal"))]
criterion_group!(
    benches,
    bench_matmul_cpu,
    bench_elementwise_cpu,
    bench_elementwise_large_cpu,
    bench_softmax_cpu,
    bench_mlp_forward_cpu,
    bench_training_step_cpu,
);

#[cfg(feature = "metal")]
criterion_group!(
    benches,
    bench_matmul_cpu,
    bench_elementwise_cpu,
    bench_elementwise_large_cpu,
    bench_softmax_cpu,
    bench_mlp_forward_cpu,
    bench_training_step_cpu,
    bench_matmul_gpu,
    bench_elementwise_gpu,
    bench_softmax_gpu,
);

criterion_main!(benches);
