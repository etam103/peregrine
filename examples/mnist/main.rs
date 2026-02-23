//! MNIST digit classifier — end-to-end validation of Peregrine.
//!
//! Model: MLP (784 → 128 → 64 → 10) with ReLU activations
//! Training: CrossEntropyLoss + Adam optimizer
//! Goal: >95% accuracy, validating the full stack works.
//!
//! Usage: cargo run --example mnist

use peregrine::nn::{self, Linear};
use peregrine::optim::Adam;
use peregrine::tensor::Tensor;

use std::fs::File;
use std::io::Read;
use std::path::Path;

// --- IDX file parser for MNIST ---

fn read_idx_images(path: &str) -> Vec<Vec<f32>> {
    let mut file = File::open(path).unwrap_or_else(|_| {
        eprintln!("Could not open {}. Download MNIST first:", path);
        eprintln!("  cd data && ./download_mnist.sh");
        std::process::exit(1);
    });
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2051, "Not an MNIST image file");
    let num_images = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let pixels = rows * cols;

    let mut images = Vec::with_capacity(num_images);
    for i in 0..num_images {
        let offset = 16 + i * pixels;
        let img: Vec<f32> = buf[offset..offset + pixels]
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect();
        images.push(img);
    }
    images
}

fn read_idx_labels(path: &str) -> Vec<usize> {
    let mut file = File::open(path).unwrap_or_else(|_| {
        eprintln!("Could not open {}. Download MNIST first.", path);
        std::process::exit(1);
    });
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2049, "Not an MNIST label file");
    let num_labels = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;

    buf[8..8 + num_labels].iter().map(|&b| b as usize).collect()
}

// --- MLP model ---

struct MnistMlp {
    fc1: Linear, // 784 -> 128
    fc2: Linear, // 128 -> 64
    fc3: Linear, // 64 -> 10
}

impl MnistMlp {
    fn new() -> Self {
        MnistMlp {
            fc1: Linear::new(784, 128),
            fc2: Linear::new(128, 64),
            fc3: Linear::new(64, 10),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.fc1.forward(x).relu();
        let h2 = self.fc2.forward(&h1).relu();
        self.fc3.forward(&h2)
    }

    fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for t in self.fc1.params().into_iter()
            .chain(self.fc2.params())
            .chain(self.fc3.params())
        {
            p.push(t.clone());
        }
        p
    }

    fn param_count(&self) -> usize {
        self.params().iter().map(|p| p.size()).sum()
    }
}

// --- Training loop ---

fn main() {
    let data_dir = "data/mnist";

    // Check if data exists, provide download instructions
    if !Path::new(&format!("{}/train-images-idx3-ubyte", data_dir)).exists() {
        eprintln!("MNIST data not found in {}/", data_dir);
        eprintln!("Creating download script...");
        create_download_script(data_dir);
        eprintln!("Run: cd {} && sh download_mnist.sh", data_dir);
        return;
    }

    println!("Loading MNIST...");
    let train_images = read_idx_images(&format!("{}/train-images-idx3-ubyte", data_dir));
    let train_labels = read_idx_labels(&format!("{}/train-labels-idx1-ubyte", data_dir));
    let test_images = read_idx_images(&format!("{}/t10k-images-idx3-ubyte", data_dir));
    let test_labels = read_idx_labels(&format!("{}/t10k-labels-idx1-ubyte", data_dir));

    println!("Train: {} images, Test: {} images", train_images.len(), test_images.len());

    let model = MnistMlp::new();
    println!("Model: {} parameters", model.param_count());

    let mut optimizer = Adam::new(model.params(), 1e-3);

    let batch_size = 64;
    let num_epochs = 10;

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        let mut total = 0usize;
        let num_batches = train_images.len() / batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            // Build batch tensor [batch_size, 784]
            let mut batch_data = Vec::with_capacity(batch_size * 784);
            let batch_targets: Vec<usize> = train_labels[start..end].to_vec();
            for i in start..end {
                batch_data.extend_from_slice(&train_images[i]);
            }
            let x = Tensor::new(batch_data, vec![batch_size, 784], false);

            // Forward
            let logits = model.forward(&x);

            // Accuracy (on this batch)
            let logit_data = logits.data();
            for i in 0..batch_size {
                let offset = i * 10;
                let pred = logit_data[offset..offset + 10]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap();
                if pred == batch_targets[i] {
                    correct += 1;
                }
                total += 1;
            }

            // Loss
            let loss = nn::cross_entropy_loss(&logits, &batch_targets);
            total_loss += loss.data()[0];

            // Backward
            loss.backward();

            // Update
            optimizer.step();
            optimizer.zero_grad();
        }

        let train_acc = 100.0 * correct as f32 / total as f32;
        let avg_loss = total_loss / num_batches as f32;

        // Test accuracy
        let test_acc = evaluate(&model, &test_images, &test_labels);

        println!(
            "Epoch {}/{}: loss={:.4}, train_acc={:.1}%, test_acc={:.1}%",
            epoch + 1, num_epochs, avg_loss, train_acc, test_acc
        );
    }
}

fn evaluate(model: &MnistMlp, images: &[Vec<f32>], labels: &[usize]) -> f32 {
    let batch_size = 256;
    let mut correct = 0usize;
    let mut total = 0usize;

    let num_batches = images.len() / batch_size;
    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = start + batch_size;

        let mut batch_data = Vec::with_capacity(batch_size * 784);
        for i in start..end {
            batch_data.extend_from_slice(&images[i]);
        }
        let x = Tensor::new(batch_data, vec![batch_size, 784], false);
        let logits = model.forward(&x);
        let logit_data = logits.data();

        for i in 0..batch_size {
            let offset = i * 10;
            let pred = logit_data[offset..offset + 10]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            if pred == labels[start + i] {
                correct += 1;
            }
            total += 1;
        }
    }
    100.0 * correct as f32 / total as f32
}

fn create_download_script(data_dir: &str) {
    std::fs::create_dir_all(data_dir).ok();
    let script = format!(
        r#"#!/bin/sh
cd "$(dirname "$0")"
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    echo "Downloading $f..."
    curl -O "$BASE_URL/$f"
    gunzip -f "$f"
done
echo "Done!"
"#
    );
    std::fs::write(format!("{}/download_mnist.sh", data_dir), script).ok();
}
