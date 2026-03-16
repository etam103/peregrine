//! MNIST CPU vs Metal GPU benchmark — tests model size scaling.
//!
//! Tests two hypotheses:
//!   1. GPU wins with larger matmuls (wider hidden layers)
//!   2. GPU wins with a fully GPU-resident training loop (no CPU fallback)
//!
//! Uses MSE loss instead of cross_entropy to keep the entire graph on GPU
//! (cross_entropy's `select` op breaks the GPU graph).
//!
//! Usage:
//!   cargo run --example mnist_gpu_bench --release --features metal

use peregrine::nn::Linear;
use peregrine::optim::Adam;
use peregrine::tensor::Tensor;

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

// --- IDX file parser ---

fn read_idx_images(path: &str) -> Vec<Vec<f32>> {
    let mut file = File::open(path).unwrap_or_else(|_| {
        eprintln!("Could not open {}. Run the mnist example first to download data.", path);
        std::process::exit(1);
    });
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    let num_images = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let pixels = rows * cols;
    let mut images = Vec::with_capacity(num_images);
    for i in 0..num_images {
        let offset = 16 + i * pixels;
        let img: Vec<f32> = buf[offset..offset + pixels].iter().map(|&b| b as f32 / 255.0).collect();
        images.push(img);
    }
    images
}

fn read_idx_labels(path: &str) -> Vec<usize> {
    let mut file = File::open(path).unwrap_or_else(|_| {
        eprintln!("Could not open {}.", path);
        std::process::exit(1);
    });
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    let num_labels = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    buf[8..8 + num_labels].iter().map(|&b| b as usize).collect()
}

// --- Configurable MLP ---

struct Mlp {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl Mlp {
    fn new(input_dim: usize, hidden: usize, output_dim: usize) -> Self {
        Mlp {
            fc1: Linear::new(input_dim, hidden),
            fc2: Linear::new(hidden, hidden),
            fc3: Linear::new(hidden, output_dim),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.fc1.forward(x).relu();
        let h2 = self.fc2.forward(&h1).relu();
        self.fc3.forward(&h2)
    }

    fn params(&self) -> Vec<Tensor> {
        self.fc1.params().into_iter()
            .chain(self.fc2.params())
            .chain(self.fc3.params())
            .cloned()
            .collect()
    }

    fn param_count(&self) -> usize {
        self.params().iter().map(|p| p.size()).sum()
    }

    #[cfg(feature = "metal")]
    fn to_gpu(&self) {
        for p in &self.fc1.params() { p.to_gpu(); }
        for p in &self.fc2.params() { p.to_gpu(); }
        for p in &self.fc3.params() { p.to_gpu(); }
    }
}

// One-hot encode labels into a [batch, num_classes] tensor
fn one_hot(labels: &[usize], num_classes: usize) -> Tensor {
    let batch = labels.len();
    let mut data = vec![0.0f32; batch * num_classes];
    for (i, &l) in labels.iter().enumerate() {
        data[i * num_classes + l] = 1.0;
    }
    Tensor::new(data, vec![batch, num_classes], false)
}

/// MSE loss: mean((pred - target)^2) — all ops stay on GPU
fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.sub(target);
    diff.mul(&diff).mean()
}

fn bench_cpu(
    hidden: usize,
    images: &[Vec<f32>],
    labels: &[usize],
    batch_size: usize,
    num_steps: usize,
) -> f64 {
    let model = Mlp::new(784, hidden, 10);
    let mut opt = Adam::new(model.params(), 1e-3);

    // Warmup: 10 steps
    for step in 0..10 {
        let start = (step * batch_size) % (images.len() - batch_size);
        let end = start + batch_size;
        let mut batch_data = Vec::with_capacity(batch_size * 784);
        for i in start..end { batch_data.extend_from_slice(&images[i]); }
        let x = Tensor::new(batch_data, vec![batch_size, 784], false);
        let target = one_hot(&labels[start..end], 10);
        let logits = model.forward(&x);
        let loss = mse_loss(&logits, &target);
        loss.backward();
        opt.step();
        opt.zero_grad();
    }

    // Timed run
    let t = Instant::now();
    for step in 0..num_steps {
        let start = (step * batch_size) % (images.len() - batch_size);
        let end = start + batch_size;
        let mut batch_data = Vec::with_capacity(batch_size * 784);
        for i in start..end { batch_data.extend_from_slice(&images[i]); }
        let x = Tensor::new(batch_data, vec![batch_size, 784], false);
        let target = one_hot(&labels[start..end], 10);
        let logits = model.forward(&x);
        let loss = mse_loss(&logits, &target);
        loss.backward();
        opt.step();
        opt.zero_grad();
    }
    t.elapsed().as_secs_f64()
}

#[cfg(feature = "metal")]
fn bench_gpu(
    hidden: usize,
    images: &[Vec<f32>],
    labels: &[usize],
    batch_size: usize,
    num_steps: usize,
) -> f64 {
    let model = Mlp::new(784, hidden, 10);
    model.to_gpu();
    let mut opt = Adam::new(model.params(), 1e-3);

    // Warmup: 10 steps
    for step in 0..10 {
        let start = (step * batch_size) % (images.len() - batch_size);
        let end = start + batch_size;
        let mut batch_data = Vec::with_capacity(batch_size * 784);
        for i in start..end { batch_data.extend_from_slice(&images[i]); }
        let x = Tensor::new(batch_data, vec![batch_size, 784], false);
        x.to_gpu();
        let target = one_hot(&labels[start..end], 10);
        target.to_gpu();
        let logits = model.forward(&x);
        let loss = mse_loss(&logits, &target);
        loss.backward();
        opt.step();
        opt.zero_grad();
    }

    // Timed run
    let t = Instant::now();
    for step in 0..num_steps {
        let start = (step * batch_size) % (images.len() - batch_size);
        let end = start + batch_size;
        let mut batch_data = Vec::with_capacity(batch_size * 784);
        for i in start..end { batch_data.extend_from_slice(&images[i]); }
        let x = Tensor::new(batch_data, vec![batch_size, 784], false);
        x.to_gpu();
        let target = one_hot(&labels[start..end], 10);
        target.to_gpu();
        let logits = model.forward(&x);
        let loss = mse_loss(&logits, &target);
        loss.backward();
        opt.step();
        opt.zero_grad();
    }
    t.elapsed().as_secs_f64()
}

fn main() {
    let data_dir = "data/mnist";
    if !Path::new(&format!("{}/train-images-idx3-ubyte", data_dir)).exists() {
        eprintln!("MNIST data not found. Run `cargo run --example mnist` first.");
        return;
    }

    println!("Loading MNIST...");
    let train_images = read_idx_images(&format!("{}/train-images-idx3-ubyte", data_dir));
    let train_labels = read_idx_labels(&format!("{}/train-labels-idx1-ubyte", data_dir));
    println!("Train: {} images\n", train_images.len());

    let batch_size = 64;
    let num_steps = 200;
    let hidden_sizes = [128, 256, 512, 1024, 2048];

    #[cfg(not(feature = "metal"))]
    {
        println!("Rebuild with --features metal for GPU benchmark");
        return;
    }

    #[cfg(feature = "metal")]
    {
        peregrine::metal::init_gpu().expect("Failed to initialize Metal GPU");

        println!("Benchmark: {} steps per config, batch_size={}", num_steps, batch_size);
        println!("Model: MLP (784 -> H -> H -> 10) with ReLU, MSE loss, Adam\n");
        println!("{:<8} {:>10} {:>8} {:>8} {:>10} {:>10}",
                 "Hidden", "Params", "CPU", "GPU", "Speedup", "Winner");
        println!("{}", "-".repeat(62));

        for &h in &hidden_sizes {
            let param_count = Mlp::new(784, h, 10).param_count();

            let cpu_time = bench_cpu(h, &train_images, &train_labels, batch_size, num_steps);
            let gpu_time = bench_gpu(h, &train_images, &train_labels, batch_size, num_steps);

            let ratio = cpu_time / gpu_time;
            let (speedup_str, winner) = if ratio > 1.0 {
                (format!("{:.2}x", ratio), "GPU")
            } else {
                (format!("{:.2}x", 1.0 / ratio), "CPU")
            };

            println!("{:<8} {:>10} {:>7.3}s {:>7.3}s {:>10} {:>10}",
                     h, param_count, cpu_time, gpu_time, speedup_str, winner);
        }
    }
}
