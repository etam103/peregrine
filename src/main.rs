mod tensor;
mod detection;
pub mod dataset;

use std::collections::HashMap;
use tensor::Tensor;
use detection::{YoloNet, GroundTruth, GRID_H, GRID_W, INPUT_SIZE, NUM_ANCHORS};
use dataset::{VocDataset, Dataset};
use tensorboard_rs::summary_writer::SummaryWriter;

const NUM_CLASSES: usize = 3;
const CLASS_NAMES: [&str; NUM_CLASSES] = ["car", "person", "dog"];

fn class_name(cls: usize) -> &'static str { CLASS_NAMES[cls] }

/// Convert CHW f32 [0,1] image to RGB u8 bytes for TensorBoard.
fn chw_to_rgb_bytes(chw: &[f32], sz: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; 3 * sz * sz];
    for y in 0..sz {
        for x in 0..sz {
            let offset = y * sz + x;
            let out_idx = (y * sz + x) * 3;
            rgb[out_idx] = (chw[offset] * 255.0).clamp(0.0, 255.0) as u8;
            rgb[out_idx + 1] = (chw[sz * sz + offset] * 255.0).clamp(0.0, 255.0) as u8;
            rgb[out_idx + 2] = (chw[2 * sz * sz + offset] * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}

/// Color per class: car=red, person=green, dog=blue.
fn gt_color(class_id: usize) -> [u8; 3] {
    match class_id {
        0 => [255, 60, 60],   // car = red
        1 => [60, 255, 60],   // person = green
        _ => [80, 120, 255],  // dog = blue
    }
}

fn pred_color(class_id: usize) -> [u8; 3] {
    match class_id {
        0 => [255, 150, 150], // car = light red
        1 => [150, 255, 150], // person = light green
        _ => [150, 180, 255], // dog = light blue
    }
}

// 5x3 bitmap font for A-Z and 0-9. Each char is 5 rows of 3 columns.
fn char_bitmap(c: char) -> [u8; 5] {
    match c.to_ascii_uppercase() {
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'N' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'O' => [0b010, 0b101, 0b101, 0b101, 0b010],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        _ =>   [0b000, 0b000, 0b000, 0b000, 0b000],
    }
}

/// Draw a small text label at (x, y) on the RGB buffer.
fn draw_label(rgb: &mut [u8], sz: usize, x: usize, y: usize, text: &str, color: [u8; 3], scale: usize) {
    let mut cx = x;
    for ch in text.chars() {
        let bitmap = char_bitmap(ch);
        for row in 0..5 {
            for col in 0..3 {
                if bitmap[row] & (1 << (2 - col)) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cx + col * scale + sx;
                            let py = y + row * scale + sy;
                            if px < sz && py < sz {
                                let idx = (py * sz + px) * 3;
                                if idx + 2 < rgb.len() {
                                    rgb[idx..idx + 3].copy_from_slice(&color);
                                }
                            }
                        }
                    }
                }
            }
        }
        cx += 4 * scale; // char width + 1px gap
    }
}

/// Draw a labeled bounding box on an RGB buffer.
fn draw_labeled_box(rgb: &mut [u8], sz: usize, cx: f32, cy: f32, w: f32, h: f32,
                    color: [u8; 3], thickness: usize, label: &str) {
    let x0 = ((cx - w / 2.0).max(0.0)) as usize;
    let y0 = ((cy - h / 2.0).max(0.0)) as usize;
    let x1 = ((cx + w / 2.0).min(sz as f32 - 1.0)) as usize;
    let y1 = ((cy + h / 2.0).min(sz as f32 - 1.0)) as usize;
    // Draw box outline
    for t in 0..thickness {
        let y0t = if y0 >= t { y0 - t } else { 0 };
        let y1t = (y1 + t).min(sz - 1);
        let x0t = if x0 >= t { x0 - t } else { 0 };
        let x1t = (x1 + t).min(sz - 1);
        for x in x0t..=x1t {
            for &y in &[y0t, y1t] {
                let idx = (y * sz + x) * 3;
                if idx + 2 < rgb.len() { rgb[idx..idx + 3].copy_from_slice(&color); }
            }
        }
        for y in y0t..=y1t {
            for &x in &[x0t, x1t] {
                let idx = (y * sz + x) * 3;
                if idx + 2 < rgb.len() { rgb[idx..idx + 3].copy_from_slice(&color); }
            }
        }
    }
    // Draw filled background for label
    let label_h = 5 * 2 + 2; // 5 rows * scale=2 + padding
    let label_w = label.len() * 4 * 2 + 2;
    let ly = if y0 >= label_h + 2 { y0 - label_h - 2 } else { y1 + 2 };
    for dy in 0..label_h {
        for dx in 0..label_w {
            let px = x0 + dx;
            let py = ly + dy;
            if px < sz && py < sz {
                let idx = (py * sz + px) * 3;
                if idx + 2 < rgb.len() {
                    rgb[idx] = color[0] / 2;
                    rgb[idx + 1] = color[1] / 2;
                    rgb[idx + 2] = color[2] / 2;
                }
            }
        }
    }
    // Draw text
    draw_label(rgb, sz, x0 + 1, ly + 1, label, [255, 255, 255], 2);
}

/// Convert dataset Target to GroundTruth in pixel coordinates.
fn targets_to_gt(targets: &[dataset::Target], img_size: usize) -> Vec<GroundTruth> {
    targets.iter().map(|t| GroundTruth {
        cx: t.cx * img_size as f32,
        cy: t.cy * img_size as f32,
        w: t.w * img_size as f32,
        h: t.h * img_size as f32,
        class_id: t.class_id,
    }).collect()
}

fn main() {
    let data_path = "data/coco_voc";
    let logdir = "runs/coco_training";

    println!("=== rustorch: Training YOLO on Real COCO Images ===");
    println!("Input: {}x{}x3, Grid: {}x{}, Anchors: {}", INPUT_SIZE, INPUT_SIZE, GRID_H, GRID_W, NUM_ANCHORS);
    println!("Classes: {:?}", CLASS_NAMES);
    println!("Dataset: {}", data_path);
    println!("TensorBoard: tensorboard --logdir runs/\n");

    // Load real COCO images in VOC format
    let ds = VocDataset::load(data_path, INPUT_SIZE, &CLASS_NAMES);
    println!("Loaded {} images with {} classes\n", ds.len(), ds.num_classes());

    if ds.is_empty() {
        eprintln!("ERROR: No images found. Run the download script first.");
        return;
    }

    let mut writer = SummaryWriter::new(logdir);
    let net = YoloNet::new(NUM_CLASSES);
    let params = net.params();
    let total_params: usize = params.iter().map(|p| p.size()).sum();
    println!("Network: {} parameter tensors, {} total weights\n", params.len(), total_params);

    let batch_size = 1; // one image at a time for memory/speed
    let num_epochs = 20;
    let mut min_loss = f32::MAX;
    let mut global_step = 0;

    for epoch in 0..num_epochs {
        let lr = if epoch < 8 { 0.001 } else { 0.0005 };
        let mut epoch_loss = 0.0;
        let num_batches = ds.num_batches(batch_size);

        for batch_idx in 0..num_batches {
            let (img_data, batch_targets) = ds.get_batch(batch_idx, batch_size);
            let actual_batch = (ds.len() - batch_idx * batch_size).min(batch_size);
            let pixels_per_img = 3 * INPUT_SIZE * INPUT_SIZE;

            // Convert targets to per-image GroundTruth vectors
            let mut gt_per_image: Vec<Vec<GroundTruth>> = vec![Vec::new(); actual_batch];
            for t in &batch_targets {
                gt_per_image[t.batch_idx].push(GroundTruth {
                    cx: t.cx * INPUT_SIZE as f32,
                    cy: t.cy * INPUT_SIZE as f32,
                    w: t.w * INPUT_SIZE as f32,
                    h: t.h * INPUT_SIZE as f32,
                    class_id: t.class_id,
                });
            }

            let images = Tensor::new(img_data.clone(), vec![actual_batch, 3, INPUT_SIZE, INPUT_SIZE], false);
            let loss = net.loss(&images, &gt_per_image);
            let loss_val = loss.data()[0];
            epoch_loss += loss_val;

            if loss_val < min_loss { min_loss = loss_val; }
            writer.add_scalar("loss/step", loss_val, global_step);

            // Log first image of first batch each epoch with detection overlay
            if batch_idx == 0 {
                let single_img = &img_data[..pixels_per_img];
                let input = Tensor::new(single_img.to_vec(), vec![1, 3, INPUT_SIZE, INPUT_SIZE], false);
                let detections = net.detect_nms(&input, 0.5, 0.4);

                let mut rgb = chw_to_rgb_bytes(single_img, INPUT_SIZE);
                // Draw ground truth with class-colored boxes
                for gt in &gt_per_image[0] {
                    let label = format!("GT:{}", class_name(gt.class_id).to_uppercase());
                    draw_labeled_box(&mut rgb, INPUT_SIZE, gt.cx, gt.cy, gt.w, gt.h, gt_color(gt.class_id), 2, &label);
                }
                // Draw top 5 predictions with class-colored boxes
                for d in detections[0].iter().take(5) {
                    let label = format!("{}", class_name(d.class_id).to_uppercase());
                    draw_labeled_box(&mut rgb, INPUT_SIZE, d.cx, d.cy, d.w, d.h, pred_color(d.class_id), 2, &label);
                }
                writer.add_image("coco/detections", &rgb, &[3, INPUT_SIZE, INPUT_SIZE], epoch);
            }

            loss.backward();
            for p in &params {
                p.sgd_step(lr);
            }
            global_step += 1;

            if batch_idx % 5 == 0 {
                print!("  epoch {:>2} [{:>3}/{}]  loss = {:.4}\r", epoch, batch_idx + 1, num_batches, loss_val);
            }
        }

        let avg_loss = epoch_loss / num_batches as f32;
        writer.add_scalar("loss/epoch_avg", avg_loss, epoch);
        writer.add_scalar("loss/min", min_loss, epoch);
        writer.add_scalar("hyperparams/lr", lr as f32, epoch);

        let mut loss_map = HashMap::new();
        loss_map.insert("avg".to_string(), avg_loss);
        loss_map.insert("min".to_string(), min_loss);
        writer.add_scalars("loss/overview", &loss_map, epoch);

        println!("epoch {:>2}  avg_loss = {:.4}  min = {:.4}  lr = {}  ({} batches)",
            epoch, avg_loss, min_loss, lr, num_batches);
    }

    // --- Inference on a few real images ---
    println!("\n--- Inference on real COCO images ---\n");
    let num_test = ds.len().min(5);
    for i in 0..num_test {
        let img_data = ds.load_image(i);
        let gt_targets = ds.get_targets(i);
        let gt = targets_to_gt(&gt_targets, INPUT_SIZE);

        let input = Tensor::new(img_data.clone(), vec![1, 3, INPUT_SIZE, INPUT_SIZE], false);
        let detections = net.detect_nms(&input, 0.5, 0.4);
        let dets = &detections[0];

        // Log to TensorBoard
        let mut rgb = chw_to_rgb_bytes(&img_data, INPUT_SIZE);
        for g in &gt {
            let label = format!("GT:{}", class_name(g.class_id).to_uppercase());
            draw_labeled_box(&mut rgb, INPUT_SIZE, g.cx, g.cy, g.w, g.h, gt_color(g.class_id), 2, &label);
        }
        for d in dets.iter().take(5) {
            let label = format!("{}", class_name(d.class_id).to_uppercase());
            draw_labeled_box(&mut rgb, INPUT_SIZE, d.cx, d.cy, d.w, d.h, pred_color(d.class_id), 2, &label);
        }
        let tag = format!("coco/inference_{}", i);
        writer.add_image(&tag, &rgb, &[3, INPUT_SIZE, INPUT_SIZE], 0);

        println!("Image {}:", i + 1);
        println!("  Ground truth: {} objects", gt.len());
        for g in &gt {
            println!("    {:<6} ({:.0}, {:.0}, {:.0}, {:.0})", class_name(g.class_id), g.cx, g.cy, g.w, g.h);
        }
        if dets.is_empty() {
            println!("  Predictions: none above threshold");
        } else {
            println!("  Predictions: {} detections", dets.len());
            for d in dets.iter().take(5) {
                println!("    {:<6} ({:.0}, {:.0}, {:.0}, {:.0})  conf={:.3}",
                    class_name(d.class_id), d.cx, d.cy, d.w, d.h, d.confidence);
            }
        }
        println!();
    }

    writer.flush();
    println!("TensorBoard logs written to {}/", logdir);
    println!("Run: tensorboard --logdir runs/");
}
