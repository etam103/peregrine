mod tensor;
mod detection;
pub mod dataset;

use std::collections::HashMap;
use tensor::Tensor;
use detection::{YoloNet, GroundTruth, GRID_H, GRID_W, INPUT_SIZE, NUM_ANCHORS};
use tensorboard_rs::summary_writer::SummaryWriter;

/// Simple PRNG for data generation.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Rng(seed) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 { self.next_u32() as f32 / u32::MAX as f32 }
}

const NUM_CLASSES: usize = 3;

fn class_name(cls: usize) -> &'static str {
    match cls { 0 => "car", 1 => "person", _ => "dog" }
}

/// Generate a synthetic 416x416 RGB "image" with one object.
fn make_synthetic_sample(rng: &mut Rng) -> (Vec<f32>, GroundTruth) {
    let cls = (rng.next_u32() % NUM_CLASSES as u32) as usize;
    let cx_norm = 0.15 + rng.next_f32() * 0.7;
    let cy_norm = 0.15 + rng.next_f32() * 0.7;
    let w_norm = 0.10 + rng.next_f32() * 0.25;
    let h_norm = 0.10 + rng.next_f32() * 0.25;

    let sz = INPUT_SIZE;
    let cx = cx_norm * sz as f32;
    let cy = cy_norm * sz as f32;
    let w = w_norm * sz as f32;
    let h = h_norm * sz as f32;

    let npixels = 3 * sz * sz;
    let mut img = vec![0.0f32; npixels];
    let x0 = ((cx - w / 2.0).max(0.0)) as usize;
    let y0 = ((cy - h / 2.0).max(0.0)) as usize;
    let x1 = ((cx + w / 2.0).min(sz as f32)) as usize;
    let y1 = ((cy + h / 2.0).min(sz as f32)) as usize;

    let (r, g, b) = match cls {
        0 => (0.9, 0.2, 0.2),
        1 => (0.2, 0.9, 0.2),
        _ => (0.2, 0.2, 0.9),
    };

    for row in y0..y1 {
        for col in x0..x1 {
            let offset = row * sz + col;
            img[0 * sz * sz + offset] = r;
            img[1 * sz * sz + offset] = g;
            img[2 * sz * sz + offset] = b;
        }
    }

    let gt = GroundTruth { cx, cy, w, h, class_id: cls };
    (img, gt)
}

/// Convert CHW f32 [0,1] image to RGB u8 bytes for TensorBoard.
fn chw_to_rgb_bytes(chw: &[f32], sz: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; 3 * sz * sz];
    for y in 0..sz {
        for x in 0..sz {
            let offset = y * sz + x;
            let out_idx = (y * sz + x) * 3;
            rgb[out_idx + 0] = (chw[0 * sz * sz + offset] * 255.0).clamp(0.0, 255.0) as u8;
            rgb[out_idx + 1] = (chw[1 * sz * sz + offset] * 255.0).clamp(0.0, 255.0) as u8;
            rgb[out_idx + 2] = (chw[2 * sz * sz + offset] * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}

/// Draw a bounding box (outlined) onto an RGB u8 buffer.
fn draw_box(rgb: &mut [u8], sz: usize, cx: f32, cy: f32, w: f32, h: f32, color: [u8; 3]) {
    let x0 = ((cx - w / 2.0).max(0.0)) as usize;
    let y0 = ((cy - h / 2.0).max(0.0)) as usize;
    let x1 = ((cx + w / 2.0).min(sz as f32 - 1.0)) as usize;
    let y1 = ((cy + h / 2.0).min(sz as f32 - 1.0)) as usize;
    // Top and bottom edges
    for x in x0..=x1 {
        for &y in &[y0, y1] {
            let idx = (y * sz + x) * 3;
            if idx + 2 < rgb.len() {
                rgb[idx..idx + 3].copy_from_slice(&color);
            }
        }
    }
    // Left and right edges
    for y in y0..=y1 {
        for &x in &[x0, x1] {
            let idx = (y * sz + x) * 3;
            if idx + 2 < rgb.len() {
                rgb[idx..idx + 3].copy_from_slice(&color);
            }
        }
    }
}

fn main() {
    let logdir = "runs/yolo_training";
    println!("=== rustorch: YOLO Object Detection with TensorBoard ===");
    println!("Input: {}x{}x3, Grid: {}x{}, Anchors: {}, Classes: {}",
        INPUT_SIZE, INPUT_SIZE, GRID_H, GRID_W, NUM_ANCHORS, NUM_CLASSES);
    println!("TensorBoard logdir: {}/", logdir);
    println!("  -> Run: tensorboard --logdir runs/\n");

    let mut writer = SummaryWriter::new(logdir);

    let net = YoloNet::new(NUM_CLASSES);
    let params = net.params();
    println!("Parameters: {} tensors\n", params.len());

    let batch_size = 2;
    let num_epochs = 30;
    let mut rng = Rng::new(42);
    let mut min_loss = f32::MAX;

    for epoch in 0..num_epochs {
        let lr = if epoch < 10 { 0.001 } else { 0.0005 };

        let mut img_data = Vec::new();
        let mut targets: Vec<Vec<GroundTruth>> = Vec::new();
        let mut sample_pixels: Option<Vec<f32>> = None;
        let mut sample_gt: Option<GroundTruth> = None;

        for i in 0..batch_size {
            let (pixels, gt) = make_synthetic_sample(&mut rng);
            if i == 0 {
                sample_pixels = Some(pixels.clone());
                sample_gt = Some(gt.clone());
            }
            img_data.extend_from_slice(&pixels);
            targets.push(vec![gt]);
        }

        let images = Tensor::new(img_data, vec![batch_size, 3, INPUT_SIZE, INPUT_SIZE], false);

        let loss = net.loss(&images, &targets);
        let loss_val = loss.data()[0];
        if loss_val < min_loss { min_loss = loss_val; }

        // --- Log scalars to TensorBoard ---
        writer.add_scalar("loss/train", loss_val, epoch);
        writer.add_scalar("loss/min", min_loss, epoch);
        writer.add_scalar("hyperparams/learning_rate", lr as f32, epoch);

        // Log per-component loss breakdown
        let mut loss_map = HashMap::new();
        loss_map.insert("train".to_string(), loss_val);
        loss_map.insert("min".to_string(), min_loss);
        writer.add_scalars("loss/overview", &loss_map, epoch);

        // --- Log sample image with detections every 5 epochs ---
        if epoch % 5 == 0 {
            if let (Some(ref pixels), Some(ref gt)) = (&sample_pixels, &sample_gt) {
                let input = Tensor::new(pixels.clone(), vec![1, 3, INPUT_SIZE, INPUT_SIZE], false);
                let detections = net.detect_nms(&input, 0.2, 0.5);

                let mut rgb = chw_to_rgb_bytes(pixels, INPUT_SIZE);

                // Draw ground truth box in green
                draw_box(&mut rgb, INPUT_SIZE, gt.cx, gt.cy, gt.w, gt.h, [0, 255, 0]);

                // Draw predicted boxes in red
                for d in &detections[0] {
                    draw_box(&mut rgb, INPUT_SIZE, d.cx, d.cy, d.w, d.h, [255, 0, 0]);
                }

                writer.add_image("detections/sample", &rgb, &[3, INPUT_SIZE, INPUT_SIZE], epoch);
            }
        }

        println!("epoch {:>2}  loss = {:.4}  min = {:.4}  lr = {}", epoch, loss_val, min_loss, lr);

        loss.backward();
        for p in &params {
            p.sgd_step(lr);
        }
    }

    // --- Final inference: log test images ---
    println!("\n--- Inference on 3 test samples (logged to TensorBoard) ---\n");
    let mut rng = Rng::new(999);
    for i in 0..3 {
        let (pixels, gt) = make_synthetic_sample(&mut rng);
        let input = Tensor::new(pixels.clone(), vec![1, 3, INPUT_SIZE, INPUT_SIZE], false);
        let detections = net.detect_nms(&input, 0.3, 0.5);
        let dets = &detections[0];

        let mut rgb = chw_to_rgb_bytes(&pixels, INPUT_SIZE);
        draw_box(&mut rgb, INPUT_SIZE, gt.cx, gt.cy, gt.w, gt.h, [0, 255, 0]);
        for d in dets {
            draw_box(&mut rgb, INPUT_SIZE, d.cx, d.cy, d.w, d.h, [255, 0, 0]);
        }
        let tag = format!("inference/sample_{}", i + 1);
        writer.add_image(&tag, &rgb, &[3, INPUT_SIZE, INPUT_SIZE], 0);

        println!("Sample {}:", i + 1);
        println!("  Ground truth: class={:<6} box=({:.0}, {:.0}, {:.0}, {:.0})",
            class_name(gt.class_id), gt.cx, gt.cy, gt.w, gt.h);
        if dets.is_empty() {
            println!("  No detections above threshold");
        } else {
            for (j, d) in dets.iter().take(3).enumerate() {
                println!("  Detection {}: class={:<6} box=({:.0}, {:.0}, {:.0}, {:.0})  conf={:.3}",
                    j + 1, class_name(d.class_id), d.cx, d.cy, d.w, d.h, d.confidence);
            }
        }
        println!();
    }

    writer.flush();
    println!("TensorBoard logs written to {}/", logdir);
    println!("Run: tensorboard --logdir runs/");
}
