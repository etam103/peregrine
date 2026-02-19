mod tensor;

use tensor::Tensor;

/// Simple PRNG for data generation (separate from tensor weights PRNG).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Rng(seed) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 { self.next_u32() as f32 / u32::MAX as f32 }
}

const GRID: usize = 8;
const PIXELS: usize = GRID * GRID; // 64-pixel "image"
const NUM_CLASSES: usize = 3;
// Per-cell output: x, y, w, h, confidence, class0, class1, class2 = 8
const CELL_OUT: usize = 4 + 1 + NUM_CLASSES;
const OUT_DIM: usize = CELL_OUT; // single detection per image for simplicity

fn make_sample(rng: &mut Rng) -> (Vec<f32>, f32, f32, f32, f32, usize) {
    let cls = (rng.next_u32() % NUM_CLASSES as u32) as usize;
    // Object center and size (normalized 0..1)
    let cx = 0.15 + rng.next_f32() * 0.7;
    let cy = 0.15 + rng.next_f32() * 0.7;
    let w = 0.15 + rng.next_f32() * 0.25;
    let h = 0.15 + rng.next_f32() * 0.25;

    // Render object as bright pixels on a dark background
    let mut img = vec![0.0f32; PIXELS];
    let x0 = ((cx - w / 2.0) * GRID as f32).max(0.0) as usize;
    let y0 = ((cy - h / 2.0) * GRID as f32).max(0.0) as usize;
    let x1 = ((cx + w / 2.0) * GRID as f32).min(GRID as f32) as usize;
    let y1 = ((cy + h / 2.0) * GRID as f32).min(GRID as f32) as usize;

    // Each class gets a different intensity pattern
    let intensity = match cls {
        0 => 1.0,  // "car" — bright
        1 => 0.7,  // "person" — medium
        _ => 0.4,  // "dog" — dim
    };
    for row in y0..y1 {
        for col in x0..x1 {
            img[row * GRID + col] = intensity;
        }
    }
    (img, cx, cy, w, h, cls)
}

fn class_name(cls: usize) -> &'static str {
    match cls { 0 => "car", 1 => "person", _ => "dog" }
}

fn main() {
    println!("=== rustorch: Object Detection Demo ===");
    println!("Grid: {}x{}, Classes: car/person/dog\n", GRID, GRID);

    // Network: image(batch, 64) -> FC(64,128) -> ReLU -> FC(128,64) -> ReLU -> FC(64, 8)
    // Output 8 values: (x, y, w, h, conf, cls0, cls1, cls2)
    let w1 = Tensor::randn(&[PIXELS, 128], true);
    let b1 = Tensor::zeros(&[1, 128], true);
    let w2 = Tensor::randn(&[128, 64], true);
    let b2 = Tensor::zeros(&[1, 64], true);
    let w3 = Tensor::randn(&[64, OUT_DIM], true);
    let b3 = Tensor::zeros(&[1, OUT_DIM], true);

    let params: Vec<&Tensor> = vec![&w1, &b1, &w2, &b2, &w3, &b3];
    let batch_size = 16;
    let num_epochs = 1500;
    let mut rng = Rng::new(123);

    for epoch in 0..num_epochs {
        let lr = if epoch < 500 { 0.01 } else if epoch < 1000 { 0.005 } else { 0.002 };
        // Generate a batch of synthetic images + labels
        let mut img_data = Vec::with_capacity(batch_size * PIXELS);
        let mut target_data = Vec::with_capacity(batch_size * OUT_DIM);

        for _ in 0..batch_size {
            let (img, cx, cy, w, h, cls) = make_sample(&mut rng);
            img_data.extend_from_slice(&img);
            // Target: [cx, cy, w, h, 1.0 (object present), one-hot class]
            target_data.push(cx);
            target_data.push(cy);
            target_data.push(w);
            target_data.push(h);
            target_data.push(1.0);
            for c in 0..NUM_CLASSES {
                target_data.push(if c == cls { 1.0 } else { 0.0 });
            }
        }

        let images = Tensor::new(img_data, vec![batch_size, PIXELS], false);
        let targets = Tensor::new(target_data, vec![batch_size, OUT_DIM], false);

        // Forward pass
        let h1 = images.matmul(&w1).add_bias(&b1).relu();
        let h2 = h1.matmul(&w2).add_bias(&b2).relu();
        let raw_out = h2.matmul(&w3).add_bias(&b3);
        let pred = raw_out.sigmoid(); // squash everything to (0,1)

        // MSE loss
        let diff = pred.add(&targets.scale(-1.0));
        let sq = diff.mul(&diff);
        let loss = sq.sum().scale(1.0 / (batch_size * OUT_DIM) as f32);

        if epoch % 150 == 0 {
            println!("epoch {:>3}  loss = {:.6}", epoch, loss.data()[0]);
        }

        loss.backward();
        for p in &params {
            p.sgd_step(lr);
        }
    }

    // --- Inference on test samples ---
    println!("\n--- Detection Results on 5 Test Samples ---\n");
    let mut rng = Rng::new(999);
    for i in 0..5 {
        let (img, gt_cx, gt_cy, gt_w, gt_h, gt_cls) = make_sample(&mut rng);
        let input = Tensor::new(img.clone(), vec![1, PIXELS], false);
        let h1 = input.matmul(&w1).add_bias(&b1).relu();
        let h2 = h1.matmul(&w2).add_bias(&b2).relu();
        let pred = h2.matmul(&w3).add_bias(&b3).sigmoid();
        let p = pred.data();

        let pred_cls = if p[5] >= p[6] && p[5] >= p[7] { 0 }
            else if p[6] >= p[5] && p[6] >= p[7] { 1 }
            else { 2 };
        let conf = p[4];

        println!("Sample {}:", i + 1);
        println!("  Ground truth: class={:<6} box=({:.2}, {:.2}, {:.2}, {:.2})",
            class_name(gt_cls), gt_cx, gt_cy, gt_w, gt_h);
        println!("  Prediction:   class={:<6} box=({:.2}, {:.2}, {:.2}, {:.2})  conf={:.2}",
            class_name(pred_cls), p[0], p[1], p[2], p[3], conf);
        let cls_ok = pred_cls == gt_cls;
        let iou = compute_iou(gt_cx, gt_cy, gt_w, gt_h, p[0], p[1], p[2], p[3]);
        println!("  Class correct: {}  IoU: {:.2}", if cls_ok { "YES" } else { "NO" }, iou);

        // ASCII visualization
        print_detection(&img, gt_cx, gt_cy, gt_w, gt_h, p[0], p[1], p[2], p[3]);
        println!();
    }
}

fn compute_iou(cx1: f32, cy1: f32, w1: f32, h1: f32,
               cx2: f32, cy2: f32, w2: f32, h2: f32) -> f32 {
    let (ax0, ay0, ax1, ay1) = (cx1 - w1/2.0, cy1 - h1/2.0, cx1 + w1/2.0, cy1 + h1/2.0);
    let (bx0, by0, bx1, by1) = (cx2 - w2/2.0, cy2 - h2/2.0, cx2 + w2/2.0, cy2 + h2/2.0);
    let ix0 = ax0.max(bx0); let iy0 = ay0.max(by0);
    let ix1 = ax1.min(bx1); let iy1 = ay1.min(by1);
    let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
    let area_a = w1 * h1;
    let area_b = w2 * h2;
    let union = area_a + area_b - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

fn print_detection(img: &[f32], gt_cx: f32, gt_cy: f32, gt_w: f32, gt_h: f32,
                   pd_cx: f32, pd_cy: f32, pd_w: f32, pd_h: f32) {
    let gt_x0 = ((gt_cx - gt_w/2.0) * GRID as f32).max(0.0) as usize;
    let gt_y0 = ((gt_cy - gt_h/2.0) * GRID as f32).max(0.0) as usize;
    let gt_x1 = ((gt_cx + gt_w/2.0) * GRID as f32).min(GRID as f32) as usize;
    let gt_y1 = ((gt_cy + gt_h/2.0) * GRID as f32).min(GRID as f32) as usize;
    let pd_x0 = ((pd_cx - pd_w/2.0) * GRID as f32).max(0.0) as usize;
    let pd_y0 = ((pd_cy - pd_h/2.0) * GRID as f32).max(0.0) as usize;
    let pd_x1 = ((pd_cx + pd_w/2.0) * GRID as f32).min(GRID as f32) as usize;
    let pd_y1 = ((pd_cy + pd_h/2.0) * GRID as f32).min(GRID as f32) as usize;

    println!("  Image + Boxes (G=ground truth, P=predicted, B=both):");
    print!("  ");
    for _ in 0..GRID { print!("--"); }
    println!();
    for r in 0..GRID {
        print!("  ");
        for c in 0..GRID {
            let in_gt = r >= gt_y0 && r < gt_y1 && c >= gt_x0 && c < gt_x1;
            let in_pd = r >= pd_y0 && r < pd_y1 && c >= pd_x0 && c < pd_x1;
            match (in_gt, in_pd) {
                (true, true)   => print!("B "),
                (true, false)  => print!("G "),
                (false, true)  => print!("P "),
                (false, false) => {
                    if img[r * GRID + c] > 0.0 { print!("# ") } else { print!(". ") }
                }
            }
        }
        println!();
    }
}
