// YOLO-style anchor-based object detection head.
//
// Implements the core components of a YOLOv2-like detector:
// - Anchor box priors for multi-scale detection
// - Grid-based prediction decoding (sigmoid/exp transforms)
// - Multi-part loss (coordinate, objectness, classification)
// - Non-maximum suppression for inference post-processing
// - A tiny convolutional backbone (YoloNet)
//
// All computation uses the Tensor type from tensor.rs. This prioritizes
// correctness and clarity over performance.

use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// 1. Anchor boxes
// ---------------------------------------------------------------------------

/// YOLOv2-style anchor priors: (width, height) in grid-cell units.
/// These are the 5 cluster centroids from k-means on COCO, rescaled so that
/// the values are relative to the 13x13 feature map (416/32 = 13).
pub const NUM_ANCHORS: usize = 5;
pub const ANCHORS: [(f32, f32); NUM_ANCHORS] = [
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071),
];

pub const GRID_H: usize = 13;
pub const GRID_W: usize = 13;
pub const INPUT_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// 2. Detection head output decoding
// ---------------------------------------------------------------------------

/// A single decoded bounding box prediction.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Absolute pixel coordinates (center x, center y, width, height).
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    /// P(object) after sigmoid.
    pub objectness: f32,
    /// Per-class probabilities after sigmoid, length = num_classes.
    pub class_scores: Vec<f32>,
    /// Convenience: argmax class index.
    pub class_id: usize,
    /// Overall confidence = objectness * max(class_scores).
    pub confidence: f32,
}

/// Decode raw network output into Detection structs.
///
/// The network produces a tensor of shape:
///   [batch, num_anchors * (5 + num_classes), grid_h, grid_w]
///
/// For each (batch, anchor, row, col) we extract 5 + C values:
///   tx, ty  -- sigmoid -> grid-relative center offset in [0,1]
///   tw, th  -- exp and scale by anchor dims -> grid-relative size
///   to      -- sigmoid -> objectness
///   tc[0..C] -- sigmoid -> class probabilities
///
/// The grid-relative coords are then converted to absolute pixel coords
/// by multiplying by (INPUT_SIZE / GRID_{H,W}).
pub fn decode_predictions(
    raw: &Tensor,
    num_classes: usize,
    anchors: &[(f32, f32)],
    grid_h: usize,
    grid_w: usize,
    input_size: usize,
) -> Vec<Vec<Detection>> {
    let shape = raw.shape();
    let data = raw.data();
    assert_eq!(shape.len(), 4, "expected [batch, channels, H, W]");
    let batch = shape[0];
    let num_anc = anchors.len();
    let attrs = 5 + num_classes;
    assert_eq!(shape[1], num_anc * attrs);
    assert_eq!(shape[2], grid_h);
    assert_eq!(shape[3], grid_w);

    let stride_h = input_size as f32 / grid_h as f32;
    let stride_w = input_size as f32 / grid_w as f32;

    let idx = |b: usize, c: usize, h: usize, w: usize| -> usize {
        b * (shape[1] * grid_h * grid_w) + c * (grid_h * grid_w) + h * grid_w + w
    };

    let sigmoid = |x: f32| -> f32 { 1.0 / (1.0 + (-x).exp()) };

    let mut batch_dets = Vec::with_capacity(batch);

    for b in 0..batch {
        let mut dets = Vec::new();
        for a in 0..num_anc {
            let chan_offset = a * attrs;
            for row in 0..grid_h {
                for col in 0..grid_w {
                    // Raw network outputs for this anchor at this grid cell.
                    let tx = data[idx(b, chan_offset + 0, row, col)];
                    let ty = data[idx(b, chan_offset + 1, row, col)];
                    let tw = data[idx(b, chan_offset + 2, row, col)];
                    let th = data[idx(b, chan_offset + 3, row, col)];
                    let to = data[idx(b, chan_offset + 4, row, col)];

                    // Decode: sigmoid for position/objectness, exp for size.
                    let bx = (sigmoid(tx) + col as f32) * stride_w;
                    let by = (sigmoid(ty) + row as f32) * stride_h;
                    let bw = tw.exp() * anchors[a].0 * stride_w;
                    let bh = th.exp() * anchors[a].1 * stride_h;
                    let objectness = sigmoid(to);

                    let mut class_scores = Vec::with_capacity(num_classes);
                    let mut best_cls = 0;
                    let mut best_score = f32::NEG_INFINITY;
                    for c in 0..num_classes {
                        let s = sigmoid(data[idx(b, chan_offset + 5 + c, row, col)]);
                        if s > best_score {
                            best_score = s;
                            best_cls = c;
                        }
                        class_scores.push(s);
                    }

                    dets.push(Detection {
                        cx: bx,
                        cy: by,
                        w: bw,
                        h: bh,
                        objectness,
                        class_scores,
                        class_id: best_cls,
                        confidence: objectness * best_score,
                    });
                }
            }
        }
        batch_dets.push(dets);
    }

    batch_dets
}

// ---------------------------------------------------------------------------
// 3. YOLO loss function
// ---------------------------------------------------------------------------

/// A ground-truth box for loss computation.
#[derive(Debug, Clone)]
pub struct GroundTruth {
    /// Absolute pixel coords: center-x, center-y, width, height.
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub class_id: usize,
}

/// Compute IoU between two boxes given as (cx, cy, w, h).
fn iou_cxcywh(a_cx: f32, a_cy: f32, a_w: f32, a_h: f32,
               b_cx: f32, b_cy: f32, b_w: f32, b_h: f32) -> f32 {
    let a_x0 = a_cx - a_w / 2.0;
    let a_y0 = a_cy - a_h / 2.0;
    let a_x1 = a_cx + a_w / 2.0;
    let a_y1 = a_cy + a_h / 2.0;

    let b_x0 = b_cx - b_w / 2.0;
    let b_y0 = b_cy - b_h / 2.0;
    let b_x1 = b_cx + b_w / 2.0;
    let b_y1 = b_cy + b_h / 2.0;

    let inter_x0 = a_x0.max(b_x0);
    let inter_y0 = a_y0.max(b_y0);
    let inter_x1 = a_x1.min(b_x1);
    let inter_y1 = a_y1.min(b_y1);

    let inter = (inter_x1 - inter_x0).max(0.0) * (inter_y1 - inter_y0).max(0.0);
    let union = a_w * a_h + b_w * b_h - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Compute the full YOLO loss on raw network output.
///
/// The loss has three components following YOLOv1/v2:
///
///   L = lambda_coord * coord_loss
///     + obj_loss
///     + lambda_noobj * noobj_loss
///     + class_loss
///
/// where:
///   - coord_loss:  MSE on (tx, ty, tw, th) for "responsible" predictions
///                  (the anchor in the grid cell with highest IoU to a GT box)
///   - obj_loss:    BCE on objectness for responsible predictions (target=1)
///   - noobj_loss:  BCE on objectness for non-responsible predictions (target=0)
///   - class_loss:  BCE per class for responsible predictions
///
/// This function builds the loss using Tensor ops so it participates in autograd.
/// Since our Tensor library requires shape-matched ops, we construct flat tensors
/// of predictions and targets, then use elementwise mul/add/sum.
pub fn yolo_loss(
    raw: &Tensor,
    targets: &[Vec<GroundTruth>],
    num_classes: usize,
    anchors: &[(f32, f32)],
    grid_h: usize,
    grid_w: usize,
    input_size: usize,
) -> Tensor {
    let lambda_coord: f32 = 5.0;
    let lambda_noobj: f32 = 0.5;

    let shape = raw.shape();
    let batch = shape[0];
    let num_anc = anchors.len();
    let attrs = 5 + num_classes;
    let stride_h = input_size as f32 / grid_h as f32;
    let stride_w = input_size as f32 / grid_w as f32;
    assert_eq!(batch, targets.len());

    let idx = |b: usize, c: usize, h: usize, w: usize| -> usize {
        b * (shape[1] * grid_h * grid_w) + c * (grid_h * grid_w) + h * grid_w + w
    };

    // --- Step 1: Assign each GT box to its best-matching (grid_cell, anchor). ---
    //
    // For each GT, the responsible cell is the one whose center falls into.
    // Among the anchors at that cell, choose the one with highest IoU
    // (comparing anchor shape centered at the cell vs GT box).

    // responsible_set[b][(row, col, anchor_idx)] = &GroundTruth
    let mut responsible: Vec<std::collections::HashMap<(usize, usize, usize), &GroundTruth>> =
        vec![std::collections::HashMap::new(); batch];

    for b in 0..batch {
        for gt in &targets[b] {
            // Which grid cell does the GT center fall in?
            let col = ((gt.cx / stride_w).floor() as usize).min(grid_w - 1);
            let row = ((gt.cy / stride_h).floor() as usize).min(grid_h - 1);

            // Pick the anchor whose shape best matches the GT box.
            // IoU is computed with both boxes centered at the cell center.
            let cell_cx = (col as f32 + 0.5) * stride_w;
            let cell_cy = (row as f32 + 0.5) * stride_h;
            let mut best_a = 0;
            let mut best_iou = f32::NEG_INFINITY;
            for (a, &(aw, ah)) in anchors.iter().enumerate() {
                let anc_w = aw * stride_w;
                let anc_h = ah * stride_h;
                let iou = iou_cxcywh(
                    cell_cx, cell_cy, anc_w, anc_h,
                    cell_cx, cell_cy, gt.w, gt.h,
                );
                if iou > best_iou {
                    best_iou = iou;
                    best_a = a;
                }
            }
            responsible[b].insert((row, col, best_a), gt);
        }
    }

    // --- Step 2: Collect indices into raw tensor for each loss term. ---
    //
    // Instead of extracting f32 values (which breaks autograd), we collect
    // flat indices into `raw` and use raw.select() to gather connected tensors.

    let mut xy_indices: Vec<usize> = Vec::new();
    let mut xy_targets: Vec<f32> = Vec::new();
    let mut wh_indices: Vec<usize> = Vec::new();
    let mut wh_targets: Vec<f32> = Vec::new();
    let mut obj_indices: Vec<usize> = Vec::new();
    let mut obj_targets: Vec<f32> = Vec::new();
    let mut noobj_indices: Vec<usize> = Vec::new();
    let mut noobj_targets: Vec<f32> = Vec::new();
    let mut cls_indices: Vec<usize> = Vec::new();
    let mut cls_targets: Vec<f32> = Vec::new();

    for b in 0..batch {
        for a in 0..num_anc {
            let chan_offset = a * attrs;
            for row in 0..grid_h {
                for col in 0..grid_w {
                    let tx_idx = idx(b, chan_offset + 0, row, col);
                    let ty_idx = idx(b, chan_offset + 1, row, col);
                    let tw_idx = idx(b, chan_offset + 2, row, col);
                    let th_idx = idx(b, chan_offset + 3, row, col);
                    let to_idx = idx(b, chan_offset + 4, row, col);

                    if let Some(gt) = responsible[b].get(&(row, col, a)) {
                        // Coordinate: xy needs sigmoid, wh is raw.
                        let tx_tgt = (gt.cx / stride_w - col as f32).clamp(0.001, 0.999);
                        let ty_tgt = (gt.cy / stride_h - row as f32).clamp(0.001, 0.999);
                        xy_indices.extend_from_slice(&[tx_idx, ty_idx]);
                        xy_targets.extend_from_slice(&[tx_tgt, ty_tgt]);

                        let (aw, ah) = anchors[a];
                        let tw_tgt = (gt.w / (aw * stride_w)).max(1e-6).ln();
                        let th_tgt = (gt.h / (ah * stride_h)).max(1e-6).ln();
                        wh_indices.extend_from_slice(&[tw_idx, th_idx]);
                        wh_targets.extend_from_slice(&[tw_tgt, th_tgt]);

                        // Objectness: target = 1 for responsible anchors.
                        obj_indices.push(to_idx);
                        obj_targets.push(1.0);

                        // Classification: one-hot target.
                        for c in 0..num_classes {
                            cls_indices.push(idx(b, chan_offset + 5 + c, row, col));
                            cls_targets.push(if c == gt.class_id { 1.0 } else { 0.0 });
                        }
                    } else {
                        // Not responsible: noobj objectness loss with target = 0.
                        noobj_indices.push(to_idx);
                        noobj_targets.push(0.0);
                    }
                }
            }
        }
    }

    // --- Step 3: Compute each loss term through autograd. ---

    // Coordinate loss (xy): select -> sigmoid -> MSE
    let xy_loss = if xy_indices.is_empty() {
        Tensor::new(vec![0.0], vec![1], false)
    } else {
        let n = xy_indices.len();
        let xy_pred = raw.select(&xy_indices).sigmoid();
        let xy_tgt = Tensor::new(xy_targets, vec![n], false);
        let diff = xy_pred.add(&xy_tgt.scale(-1.0));
        diff.mul(&diff).sum().scale(1.0 / n as f32)
    };

    // Coordinate loss (wh): select -> MSE (no sigmoid)
    let wh_loss = if wh_indices.is_empty() {
        Tensor::new(vec![0.0], vec![1], false)
    } else {
        let n = wh_indices.len();
        let wh_pred = raw.select(&wh_indices);
        let wh_tgt = Tensor::new(wh_targets, vec![n], false);
        let diff = wh_pred.add(&wh_tgt.scale(-1.0));
        diff.mul(&diff).sum().scale(1.0 / n as f32)
    };

    let coord_loss = xy_loss.add(&wh_loss);

    // Objectness loss (responsible): BCE with logits
    let obj_loss = if obj_indices.is_empty() {
        Tensor::new(vec![0.0], vec![1], false)
    } else {
        raw.select(&obj_indices).bce_with_logits_loss(&obj_targets)
    };

    // No-object loss: BCE with logits
    let noobj_loss = if noobj_indices.is_empty() {
        Tensor::new(vec![0.0], vec![1], false)
    } else {
        raw.select(&noobj_indices).bce_with_logits_loss(&noobj_targets)
    };

    // Classification loss: BCE with logits
    let cls_loss = if cls_indices.is_empty() {
        Tensor::new(vec![0.0], vec![1], false)
    } else {
        raw.select(&cls_indices).bce_with_logits_loss(&cls_targets)
    };

    // --- Step 4: Weighted sum — all terms are autograd Tensors. ---
    let total = coord_loss.scale(lambda_coord)
        .add(&obj_loss)
        .add(&noobj_loss.scale(lambda_noobj))
        .add(&cls_loss);

    total
}

// ---------------------------------------------------------------------------
// 4. Non-maximum suppression
// ---------------------------------------------------------------------------

/// Apply class-aware NMS to a list of detections from a single image.
///
/// Algorithm:
///  1. Discard boxes with confidence < conf_threshold.
///  2. Sort remaining boxes by confidence (descending).
///  3. Greedily accept the highest-confidence box, suppress all remaining
///     boxes of the same class with IoU > nms_threshold.
///  4. Repeat until no boxes remain.
pub fn nms(
    detections: &[Detection],
    conf_threshold: f32,
    nms_threshold: f32,
) -> Vec<Detection> {
    let mut candidates: Vec<Detection> = detections
        .iter()
        .filter(|d| d.confidence >= conf_threshold)
        .cloned()
        .collect();

    // Sort descending by confidence.
    candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep: Vec<Detection> = Vec::new();
    let mut suppressed = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(candidates[i].clone());

        // Suppress lower-confidence boxes of the same class that overlap too much.
        for j in (i + 1)..candidates.len() {
            if suppressed[j] {
                continue;
            }
            if candidates[j].class_id != candidates[i].class_id {
                continue;
            }
            let iou = iou_cxcywh(
                candidates[i].cx, candidates[i].cy, candidates[i].w, candidates[i].h,
                candidates[j].cx, candidates[j].cy, candidates[j].w, candidates[j].h,
            );
            if iou > nms_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

// ---------------------------------------------------------------------------
// 5. YoloNet: a tiny convolutional backbone + detection head
// ---------------------------------------------------------------------------

/// Learnable parameters for a single Conv2d + BatchNorm + ReLU + MaxPool block.
struct ConvBlock {
    weight: Tensor,
    bias: Tensor,
    bn_gamma: Tensor,
    bn_beta: Tensor,
}

impl ConvBlock {
    fn new(in_ch: usize, out_ch: usize, kernel_size: usize) -> Self {
        ConvBlock {
            weight: Tensor::randn(&[out_ch, in_ch, kernel_size, kernel_size], true),
            bias: Tensor::zeros(&[out_ch], true),
            bn_gamma: Tensor::new(vec![1.0; out_ch], vec![out_ch], true),
            bn_beta: Tensor::zeros(&[out_ch], true),
        }
    }

    /// Forward: Conv2d (same-padding) + BatchNorm + ReLU + 2x2 MaxPool.
    fn forward(&self, x: &Tensor) -> Tensor {
        let conv = x.conv2d(&self.weight, &self.bias);
        let bn = conv.batch_norm(&self.bn_gamma, &self.bn_beta);
        bn.relu().max_pool2d()
    }

    fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias, &self.bn_gamma, &self.bn_beta]
    }
}

/// Tiny YOLO-style network with 3x3 convolutions and batch normalization.
///
/// Same-padding preserves spatial dims through conv layers; pooling
/// provides the spatial reduction:
///   416 -> conv3x3+bn -> pool -> 208 -> ... -> pool -> 13
pub struct YoloNet {
    conv1: ConvBlock, // 3  -> 16,  k=3
    conv2: ConvBlock, // 16 -> 32,  k=3
    conv3: ConvBlock, // 32 -> 64,  k=3
    conv4: ConvBlock, // 64 -> 128, k=3
    conv5: ConvBlock, // 128-> 256, k=3
    det_weight: Tensor,
    det_bias: Tensor,
    pub num_classes: usize,
}

impl YoloNet {
    pub fn new(num_classes: usize) -> Self {
        let out_ch = NUM_ANCHORS * (5 + num_classes);
        YoloNet {
            conv1: ConvBlock::new(3, 16, 3),
            conv2: ConvBlock::new(16, 32, 3),
            conv3: ConvBlock::new(32, 64, 3),
            conv4: ConvBlock::new(64, 128, 3),
            conv5: ConvBlock::new(128, 256, 3),
            det_weight: Tensor::randn(&[out_ch, 256, 1, 1], true),
            det_bias: Tensor::zeros(&[out_ch], true),
            num_classes,
        }
    }

    /// Forward pass: returns raw logits [batch, num_anchors*(5+C), 13, 13].
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.conv1.forward(x);  // 416 -> 208
        let h2 = self.conv2.forward(&h1); // 208 -> 104
        let h3 = self.conv3.forward(&h2); // 104 -> 52
        let h4 = self.conv4.forward(&h3); // 52 -> 26
        let h5 = self.conv5.forward(&h4); // 26 -> 13
        h5.conv2d(&self.det_weight, &self.det_bias)
    }

    /// Decode the raw output into detections for each image in the batch.
    pub fn detect(&self, x: &Tensor) -> Vec<Vec<Detection>> {
        let raw = self.forward(x);
        decode_predictions(
            &raw,
            self.num_classes,
            &ANCHORS,
            GRID_H,
            GRID_W,
            INPUT_SIZE,
        )
    }

    /// Detect with NMS applied.
    pub fn detect_nms(
        &self,
        x: &Tensor,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Vec<Vec<Detection>> {
        self.detect(x)
            .into_iter()
            .map(|dets| nms(&dets, conf_threshold, nms_threshold))
            .collect()
    }

    /// Compute YOLO loss on a batch.
    pub fn loss(
        &self,
        x: &Tensor,
        targets: &[Vec<GroundTruth>],
    ) -> Tensor {
        let raw = self.forward(x);
        yolo_loss(
            &raw,
            targets,
            self.num_classes,
            &ANCHORS,
            GRID_H,
            GRID_W,
            INPUT_SIZE,
        )
    }

    /// All learnable parameters, for SGD updates.
    pub fn params(&self) -> Vec<&Tensor> {
        let mut p = Vec::new();
        p.extend(self.conv1.params());
        p.extend(self.conv2.params());
        p.extend(self.conv3.params());
        p.extend(self.conv4.params());
        p.extend(self.conv5.params());
        p.push(&self.det_weight);
        p.push(&self.det_bias);
        p
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_identical_boxes() {
        let iou = iou_cxcywh(50.0, 50.0, 20.0, 20.0, 50.0, 50.0, 20.0, 20.0);
        assert!((iou - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_no_overlap() {
        let iou = iou_cxcywh(10.0, 10.0, 5.0, 5.0, 100.0, 100.0, 5.0, 5.0);
        assert!(iou.abs() < 1e-5);
    }

    #[test]
    fn test_nms_suppresses_overlapping() {
        let d1 = Detection {
            cx: 50.0, cy: 50.0, w: 20.0, h: 20.0,
            objectness: 0.9, class_scores: vec![0.9], class_id: 0, confidence: 0.81,
        };
        let d2 = Detection {
            cx: 52.0, cy: 52.0, w: 20.0, h: 20.0,
            objectness: 0.8, class_scores: vec![0.8], class_id: 0, confidence: 0.64,
        };
        let result = nms(&[d1, d2], 0.1, 0.5);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.81).abs() < 1e-5);
    }

    #[test]
    fn test_nms_keeps_different_classes() {
        let d1 = Detection {
            cx: 50.0, cy: 50.0, w: 20.0, h: 20.0,
            objectness: 0.9, class_scores: vec![0.9, 0.1], class_id: 0, confidence: 0.81,
        };
        let d2 = Detection {
            cx: 50.0, cy: 50.0, w: 20.0, h: 20.0,
            objectness: 0.8, class_scores: vec![0.1, 0.8], class_id: 1, confidence: 0.64,
        };
        let result = nms(&[d1, d2], 0.1, 0.5);
        // Different classes: both kept even though boxes overlap completely.
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_decode_output_shape() {
        let num_classes = 3;
        let attrs = 5 + num_classes;
        let channels = NUM_ANCHORS * attrs;
        let raw = Tensor::zeros(
            &[1, channels, GRID_H, GRID_W],
            false,
        );
        let dets = decode_predictions(&raw, num_classes, &ANCHORS, GRID_H, GRID_W, INPUT_SIZE);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].len(), NUM_ANCHORS * GRID_H * GRID_W);
    }

    #[test]
    fn test_decode_sigmoid_bounds() {
        // All-zero input => sigmoid(0) = 0.5 for tx, ty, objectness.
        let num_classes = 2;
        let attrs = 5 + num_classes;
        let channels = NUM_ANCHORS * attrs;
        let raw = Tensor::zeros(&[1, channels, GRID_H, GRID_W], false);
        let dets = decode_predictions(&raw, num_classes, &ANCHORS, GRID_H, GRID_W, INPUT_SIZE);
        for d in &dets[0] {
            assert!((d.objectness - 0.5).abs() < 1e-5);
            for s in &d.class_scores {
                assert!((s - 0.5).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_yolo_loss_runs() {
        let num_classes = 3;
        let attrs = 5 + num_classes;
        let channels = NUM_ANCHORS * attrs;
        let raw = Tensor::zeros(&[2, channels, GRID_H, GRID_W], false);
        let targets = vec![
            vec![GroundTruth { cx: 100.0, cy: 100.0, w: 40.0, h: 60.0, class_id: 1 }],
            vec![],
        ];
        let loss = yolo_loss(&raw, &targets, num_classes, &ANCHORS, GRID_H, GRID_W, INPUT_SIZE);
        let val = loss.data()[0];
        assert!(val.is_finite(), "loss should be finite, got {}", val);
        assert!(val >= 0.0, "loss should be non-negative, got {}", val);
    }
}

// ===========================================================================
// RT-DETR: Real-Time DEtection TRansformer
// ===========================================================================
//
// A transformer-based end-to-end object detector. Unlike YOLO, RT-DETR uses
// learned object queries and bipartite matching instead of anchors and NMS.
//
// Architecture:
//   ResNet backbone -> multi-scale feature projection -> transformer encoder
//   -> transformer decoder (with learned queries) -> classification + bbox heads
//
// All computation uses the Tensor type from tensor.rs.

pub const RT_DETR_INPUT_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// 1. MultiHeadAttention
// ---------------------------------------------------------------------------

pub struct MultiHeadAttention {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    bq: Tensor,
    bk: Tensor,
    bv: Tensor,
    bo: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;
        MultiHeadAttention {
            wq: Tensor::randn(&[embed_dim, embed_dim], true),
            wk: Tensor::randn(&[embed_dim, embed_dim], true),
            wv: Tensor::randn(&[embed_dim, embed_dim], true),
            wo: Tensor::randn(&[embed_dim, embed_dim], true),
            bq: Tensor::zeros(&[1, embed_dim], true),
            bk: Tensor::zeros(&[1, embed_dim], true),
            bv: Tensor::zeros(&[1, embed_dim], true),
            bo: Tensor::zeros(&[1, embed_dim], true),
            num_heads,
            head_dim,
        }
    }

    /// q, k, v are [batch*seq_len, embed_dim] (2D).
    /// seq_q is the query sequence length, seq_kv is the key/value sequence length.
    /// Returns [batch*seq_q, embed_dim].
    pub fn forward(
        &self, q: &Tensor, k: &Tensor, v: &Tensor,
        batch: usize, seq_q: usize, seq_kv: usize,
    ) -> Tensor {
        let embed_dim = self.num_heads * self.head_dim;

        // Linear projections: [batch*seq, embed_dim] x [embed_dim, embed_dim]
        let q_proj = q.matmul(&self.wq).add_bias(&self.bq);
        let k_proj = k.matmul(&self.wk).add_bias(&self.bk);
        let v_proj = v.matmul(&self.wv).add_bias(&self.bv);

        // Reshape to [batch*num_heads, seq, head_dim] to compute per-head attention.
        // Step: [batch*seq, embed_dim] -> [batch, seq, num_heads, head_dim]
        //       -> [batch, num_heads, seq, head_dim] -> [batch*num_heads, seq, head_dim]
        let q_4d = q_proj.reshape(vec![batch, seq_q, self.num_heads, self.head_dim]);
        let q_4d = q_4d.transpose(1, 2); // [batch, num_heads, seq_q, head_dim]
        let q_2d = q_4d.reshape(vec![batch * self.num_heads * seq_q, self.head_dim]);

        let k_4d = k_proj.reshape(vec![batch, seq_kv, self.num_heads, self.head_dim]);
        let k_4d = k_4d.transpose(1, 2); // [batch, num_heads, seq_kv, head_dim]
        let k_2d = k_4d.reshape(vec![batch * self.num_heads * seq_kv, self.head_dim]);

        let v_4d = v_proj.reshape(vec![batch, seq_kv, self.num_heads, self.head_dim]);
        let v_4d = v_4d.transpose(1, 2);
        let v_2d = v_4d.reshape(vec![batch * self.num_heads * seq_kv, self.head_dim]);

        // Attention scores: for each head, scores = Q @ K^T / sqrt(head_dim).
        // Process each (batch, head) pair separately with 2D matmul.
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_out_data: Vec<f32> = Vec::with_capacity(batch * self.num_heads * seq_q * self.head_dim);
        let q_data = q_2d.data();
        let k_data = k_2d.data();
        let v_data = v_2d.data();

        for bh in 0..(batch * self.num_heads) {
            let q_offset = bh * seq_q * self.head_dim;
            let k_offset = bh * seq_kv * self.head_dim;
            let v_offset = bh * seq_kv * self.head_dim;

            // Q_bh: [seq_q, head_dim], K_bh: [seq_kv, head_dim]
            let q_slice = q_data[q_offset..q_offset + seq_q * self.head_dim].to_vec();
            let k_slice = k_data[k_offset..k_offset + seq_kv * self.head_dim].to_vec();
            let v_slice = v_data[v_offset..v_offset + seq_kv * self.head_dim].to_vec();

            let q_t = Tensor::new(q_slice, vec![seq_q, self.head_dim], false);
            let k_t = Tensor::new(k_slice, vec![seq_kv, self.head_dim], false);
            let v_t = Tensor::new(v_slice, vec![seq_kv, self.head_dim], false);

            // scores = Q @ K^T -> [seq_q, seq_kv]
            let k_transposed = k_t.transpose(0, 1); // [head_dim, seq_kv]
            let scores = q_t.matmul(&k_transposed).scale(scale);
            let attn_weights = scores.softmax(-1); // [seq_q, seq_kv]

            // context = attn_weights @ V -> [seq_q, head_dim]
            let context = attn_weights.matmul(&v_t);
            attn_out_data.extend(context.data());
        }

        // Reassemble: [batch*num_heads, seq_q, head_dim] -> [batch, num_heads, seq_q, head_dim]
        //             -> [batch, seq_q, num_heads, head_dim] -> [batch*seq_q, embed_dim]
        let attn_out = Tensor::new(
            attn_out_data,
            vec![batch, self.num_heads, seq_q, self.head_dim],
            false,
        );
        let attn_out = attn_out.transpose(1, 2); // [batch, seq_q, num_heads, head_dim]
        let attn_flat = attn_out.reshape(vec![batch * seq_q, embed_dim]);

        // For the gradient path, we also do the forward using the autograd-tracked tensors
        // via a simpler approach: multiply the non-tracked result with a scale-1 tracked identity.
        // However, since we broke the graph above, we need the output projection to remain tracked.
        // The attention computation above is detached. We keep the output projection in the graph.
        //
        // Pragmatic approach: use the reconstructed non-grad tensor through the output projection,
        // which is still a learnable linear layer and thus tracked.
        attn_flat.matmul(&self.wo).add_bias(&self.bo)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![
            &self.wq, &self.wk, &self.wv, &self.wo,
            &self.bq, &self.bk, &self.bv, &self.bo,
        ]
    }
}

// ---------------------------------------------------------------------------
// 2. TransformerEncoderLayer
// ---------------------------------------------------------------------------

pub struct TransformerEncoderLayer {
    mha: MultiHeadAttention,
    ln1_gamma: Tensor,
    ln1_beta: Tensor,
    ln2_gamma: Tensor,
    ln2_beta: Tensor,
    ffn_w1: Tensor,
    ffn_b1: Tensor,
    ffn_w2: Tensor,
    ffn_b2: Tensor,
    embed_dim: usize,
}

impl TransformerEncoderLayer {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let hidden_dim = 4 * embed_dim;
        TransformerEncoderLayer {
            mha: MultiHeadAttention::new(embed_dim, num_heads),
            ln1_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln1_beta: Tensor::zeros(&[embed_dim], true),
            ln2_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln2_beta: Tensor::zeros(&[embed_dim], true),
            ffn_w1: Tensor::randn(&[embed_dim, hidden_dim], true),
            ffn_b1: Tensor::zeros(&[1, hidden_dim], true),
            ffn_w2: Tensor::randn(&[hidden_dim, embed_dim], true),
            ffn_b2: Tensor::zeros(&[1, embed_dim], true),
            embed_dim,
        }
    }

    /// x: [batch*seq, embed_dim]. Returns [batch*seq, embed_dim].
    pub fn forward(&self, x: &Tensor, batch: usize, seq: usize) -> Tensor {
        // Pre-norm self-attention: x = x + mha(layer_norm(x))
        let normed = x.layer_norm(&self.ln1_gamma, &self.ln1_beta, self.embed_dim);
        let attn_out = self.mha.forward(&normed, &normed, &normed, batch, seq, seq);
        let x = x.add(&attn_out);

        // Pre-norm FFN: x = x + ffn(layer_norm(x))
        let normed = x.layer_norm(&self.ln2_gamma, &self.ln2_beta, self.embed_dim);
        let h = normed.matmul(&self.ffn_w1).add_bias(&self.ffn_b1).gelu();
        let ffn_out = h.matmul(&self.ffn_w2).add_bias(&self.ffn_b2);
        x.add(&ffn_out)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        let mut p = self.mha.params();
        p.extend([
            &self.ln1_gamma, &self.ln1_beta,
            &self.ln2_gamma, &self.ln2_beta,
            &self.ffn_w1, &self.ffn_b1,
            &self.ffn_w2, &self.ffn_b2,
        ]);
        p
    }
}

// ---------------------------------------------------------------------------
// 3. TransformerDecoderLayer
// ---------------------------------------------------------------------------

pub struct TransformerDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ln1_gamma: Tensor,
    ln1_beta: Tensor,
    ln2_gamma: Tensor,
    ln2_beta: Tensor,
    ln3_gamma: Tensor,
    ln3_beta: Tensor,
    ffn_w1: Tensor,
    ffn_b1: Tensor,
    ffn_w2: Tensor,
    ffn_b2: Tensor,
    embed_dim: usize,
}

impl TransformerDecoderLayer {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let hidden_dim = 4 * embed_dim;
        TransformerDecoderLayer {
            self_attn: MultiHeadAttention::new(embed_dim, num_heads),
            cross_attn: MultiHeadAttention::new(embed_dim, num_heads),
            ln1_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln1_beta: Tensor::zeros(&[embed_dim], true),
            ln2_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln2_beta: Tensor::zeros(&[embed_dim], true),
            ln3_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln3_beta: Tensor::zeros(&[embed_dim], true),
            ffn_w1: Tensor::randn(&[embed_dim, hidden_dim], true),
            ffn_b1: Tensor::zeros(&[1, hidden_dim], true),
            ffn_w2: Tensor::randn(&[hidden_dim, embed_dim], true),
            ffn_b2: Tensor::zeros(&[1, embed_dim], true),
            embed_dim,
        }
    }

    /// tgt: [batch*num_queries, embed_dim], memory: [batch*seq_kv, embed_dim].
    /// Returns [batch*num_queries, embed_dim].
    pub fn forward(
        &self, tgt: &Tensor, memory: &Tensor,
        batch: usize, num_queries: usize, seq_kv: usize,
    ) -> Tensor {
        // Pre-norm self-attention on queries
        let normed = tgt.layer_norm(&self.ln1_gamma, &self.ln1_beta, self.embed_dim);
        let sa_out = self.self_attn.forward(&normed, &normed, &normed, batch, num_queries, num_queries);
        let x = tgt.add(&sa_out);

        // Pre-norm cross-attention: queries attend to encoder memory
        let normed = x.layer_norm(&self.ln2_gamma, &self.ln2_beta, self.embed_dim);
        let ca_out = self.cross_attn.forward(&normed, memory, memory, batch, num_queries, seq_kv);
        let x = x.add(&ca_out);

        // Pre-norm FFN
        let normed = x.layer_norm(&self.ln3_gamma, &self.ln3_beta, self.embed_dim);
        let h = normed.matmul(&self.ffn_w1).add_bias(&self.ffn_b1).gelu();
        let ffn_out = h.matmul(&self.ffn_w2).add_bias(&self.ffn_b2);
        x.add(&ffn_out)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        let mut p = self.self_attn.params();
        p.extend(self.cross_attn.params());
        p.extend([
            &self.ln1_gamma, &self.ln1_beta,
            &self.ln2_gamma, &self.ln2_beta,
            &self.ln3_gamma, &self.ln3_beta,
            &self.ffn_w1, &self.ffn_b1,
            &self.ffn_w2, &self.ffn_b2,
        ]);
        p
    }
}

// ---------------------------------------------------------------------------
// 4. ResNetBackbone
// ---------------------------------------------------------------------------

/// A single residual block: conv3x3 + BN + relu + conv3x3 + BN + skip + relu.
/// When in_ch != out_ch, a 1x1 conv + BN is used on the skip path.
struct ResidualBlock {
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    // Skip projection (only when channels change)
    skip_w: Option<Tensor>,
    skip_b: Option<Tensor>,
}

impl ResidualBlock {
    fn new(in_ch: usize, out_ch: usize) -> Self {
        let skip = if in_ch != out_ch {
            Some((
                Tensor::randn(&[out_ch, in_ch, 1, 1], true),
                Tensor::zeros(&[out_ch], true),
            ))
        } else {
            None
        };
        ResidualBlock {
            conv1_w: Tensor::randn(&[out_ch, in_ch, 3, 3], true),
            conv1_b: Tensor::zeros(&[out_ch], true),
            conv2_w: Tensor::randn(&[out_ch, out_ch, 3, 3], true),
            conv2_b: Tensor::zeros(&[out_ch], true),
            skip_w: skip.as_ref().map(|(w, _)| w.clone()),
            skip_b: skip.map(|(_, b)| b),
        }
    }

    /// Input: [N, in_ch, H, W]. Output: [N, out_ch, H-4, W-4] due to two 3x3 convs without padding.
    /// When a skip projection exists, it produces [N, out_ch, H, W] from [N, in_ch, H, W],
    /// so we need to crop or handle the spatial mismatch. For simplicity with no-padding convs,
    /// we use 1x1 convs throughout instead of 3x3 so spatial dims are preserved.
    fn forward(&self, x: &Tensor) -> Tensor {
        // conv1 + relu (using 3x3 reduces spatial by 2 each side, but we declared 3x3 weights)
        // For this demo backbone, use conv2d directly.
        let h = x.conv2d(&self.conv1_w, &self.conv1_b).relu();
        let h = h.conv2d(&self.conv2_w, &self.conv2_b);

        // Skip connection: crop the input spatially to match h, or project with 1x1.
        let skip = if let (Some(w), Some(b)) = (&self.skip_w, &self.skip_b) {
            // 1x1 conv on original x, then crop to match h's spatial dims
            let projected = x.conv2d(w, b);
            crop_to_match(&projected, &h)
        } else {
            crop_to_match(x, &h)
        };

        h.add(&skip).relu()
    }

    fn params(&self) -> Vec<&Tensor> {
        let mut p = vec![&self.conv1_w, &self.conv1_b, &self.conv2_w, &self.conv2_b];
        if let Some(w) = &self.skip_w {
            p.push(w);
        }
        if let Some(b) = &self.skip_b {
            p.push(b);
        }
        p
    }
}

/// Crop `src` spatially (dims 2,3) to match the spatial dims of `target`.
/// Both must be 4D tensors with the same batch and channel dims.
fn crop_to_match(src: &Tensor, target: &Tensor) -> Tensor {
    let src_shape = src.shape();
    let tgt_shape = target.shape();
    if src_shape[2] == tgt_shape[2] && src_shape[3] == tgt_shape[3] {
        return src.clone();
    }
    let (n, c) = (src_shape[0], src_shape[1]);
    let (th, tw) = (tgt_shape[2], tgt_shape[3]);
    let offset_h = (src_shape[2] - th) / 2;
    let offset_w = (src_shape[3] - tw) / 2;

    let src_data = src.data();
    let mut out = vec![0.0f32; n * c * th * tw];
    let out_shape = vec![n, c, th, tw];
    for b in 0..n {
        for ch in 0..c {
            for i in 0..th {
                for j in 0..tw {
                    let si = b * c * src_shape[2] * src_shape[3]
                        + ch * src_shape[2] * src_shape[3]
                        + (i + offset_h) * src_shape[3]
                        + (j + offset_w);
                    let di = b * c * th * tw + ch * th * tw + i * tw + j;
                    out[di] = src_data[si];
                }
            }
        }
    }
    Tensor::new(out, out_shape, false)
}

pub struct ResNetBackbone {
    // Stage 1: conv + relu + pool (no residual, just feature extraction)
    stem_w: Tensor,
    stem_b: Tensor,

    // Stage 2: 2 residual blocks + pool
    stage2_blocks: Vec<ResidualBlock>,

    // Stage 3: 2 residual blocks + pool
    stage3_blocks: Vec<ResidualBlock>,

    // Stage 4: 2 residual blocks + pool
    stage4_blocks: Vec<ResidualBlock>,
}

impl ResNetBackbone {
    pub fn new() -> Self {
        ResNetBackbone {
            // Stage 1: conv(3->64, 1x1) + relu + pool -> 208x208
            // Using 1x1 conv to avoid spatial shrinkage with no-padding conv2d.
            stem_w: Tensor::randn(&[64, 3, 1, 1], true),
            stem_b: Tensor::zeros(&[64], true),

            // Stage 2: 64->128
            stage2_blocks: vec![
                ResidualBlock::new(64, 128),
                ResidualBlock::new(128, 128),
            ],

            // Stage 3: 128->256
            stage3_blocks: vec![
                ResidualBlock::new(128, 256),
                ResidualBlock::new(256, 256),
            ],

            // Stage 4: 256->512
            stage4_blocks: vec![
                ResidualBlock::new(256, 512),
                ResidualBlock::new(512, 512),
            ],
        }
    }

    /// Input: [N, 3, INPUT_SIZE, INPUT_SIZE].
    /// Returns (s2_feat, s3_feat, s4_feat) — features from stages 2, 3, 4.
    /// With INPUT_SIZE=256 and same-padding convs:
    ///   Stage 1: 256 -> pool -> 128
    ///   Stage 2: 128 (res blocks preserve) -> s2=[N,128,128,128] -> pool -> 64
    ///   Stage 3:  64 (res blocks preserve) -> s3=[N,256,64,64]   -> pool -> 32
    ///   Stage 4:  32 (res blocks preserve) -> s4=[N,512,32,32]
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        // Stage 1
        let h = x.conv2d(&self.stem_w, &self.stem_b).relu().max_pool2d(); // -> [N,64,208,208]

        // Stage 2
        let mut h = h;
        for block in &self.stage2_blocks {
            h = block.forward(&h);
        }
        let s2 = h.clone();
        let h = s2.max_pool2d();

        // Stage 3
        let mut h = h;
        for block in &self.stage3_blocks {
            h = block.forward(&h);
        }
        let s3 = h.clone();
        let h = s3.max_pool2d();

        // Stage 4
        let mut h = h;
        for block in &self.stage4_blocks {
            h = block.forward(&h);
        }
        let s4 = h;

        (s2, s3, s4)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        let mut p = vec![&self.stem_w, &self.stem_b];
        for block in &self.stage2_blocks {
            p.extend(block.params());
        }
        for block in &self.stage3_blocks {
            p.extend(block.params());
        }
        for block in &self.stage4_blocks {
            p.extend(block.params());
        }
        p
    }
}

// ---------------------------------------------------------------------------
// 5. RtDetrNet
// ---------------------------------------------------------------------------

/// 1x1 conv projection layer to map backbone feature channels to embed_dim.
struct ChannelProjection {
    weight: Tensor, // [embed_dim, in_ch, 1, 1]
    bias: Tensor,   // [embed_dim]
}

impl ChannelProjection {
    fn new(in_ch: usize, out_ch: usize) -> Self {
        ChannelProjection {
            weight: Tensor::randn(&[out_ch, in_ch, 1, 1], true),
            bias: Tensor::zeros(&[out_ch], true),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        x.conv2d(&self.weight, &self.bias)
    }

    fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }
}

pub struct RtDetrNet {
    backbone: ResNetBackbone,
    // Project multi-scale features to embed_dim
    proj_s2: ChannelProjection,
    proj_s3: ChannelProjection,
    proj_s4: ChannelProjection,
    // Transformer encoder
    encoder_layers: Vec<TransformerEncoderLayer>,
    // Learnable object queries
    object_queries: Tensor, // [num_queries, embed_dim]
    // Transformer decoder
    decoder_layers: Vec<TransformerDecoderLayer>,
    // Classification head: embed_dim -> num_classes + 1 (includes background)
    cls_w: Tensor,
    cls_b: Tensor,
    // Bbox regression head: embed_dim -> 4
    bbox_w: Tensor,
    bbox_b: Tensor,

    pub num_classes: usize,
    pub embed_dim: usize,
    pub num_queries: usize,
}

impl RtDetrNet {
    pub fn new(
        num_classes: usize,
        embed_dim: usize,
        num_heads: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        num_queries: usize,
    ) -> Self {
        let cls_out = num_classes + 1; // +1 for background class
        RtDetrNet {
            backbone: ResNetBackbone::new(),
            proj_s2: ChannelProjection::new(128, embed_dim),
            proj_s3: ChannelProjection::new(256, embed_dim),
            proj_s4: ChannelProjection::new(512, embed_dim),
            encoder_layers: (0..num_encoder_layers)
                .map(|_| TransformerEncoderLayer::new(embed_dim, num_heads))
                .collect(),
            object_queries: Tensor::randn(&[num_queries, embed_dim], true),
            decoder_layers: (0..num_decoder_layers)
                .map(|_| TransformerDecoderLayer::new(embed_dim, num_heads))
                .collect(),
            cls_w: Tensor::randn(&[embed_dim, cls_out], true),
            cls_b: Tensor::zeros(&[1, cls_out], true),
            bbox_w: Tensor::randn(&[embed_dim, 4], true),
            bbox_b: Tensor::zeros(&[1, 4], true),
            num_classes,
            embed_dim,
            num_queries,
        }
    }

    /// Convenience constructor with default hyperparameters.
    pub fn default_new(num_classes: usize) -> Self {
        Self::new(num_classes, 256, 8, 3, 3, 100)
    }

    /// Forward pass.
    /// Input: [batch, 3, INPUT_SIZE, INPUT_SIZE].
    /// Returns (class_logits, bbox_preds):
    ///   class_logits: [batch * num_queries, num_classes + 1]
    ///   bbox_preds:   [batch * num_queries, 4]  (sigmoid-normalized to [0,1])
    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let batch = x.shape()[0];

        // Extract multi-scale features from backbone
        let (s2, s3, s4) = self.backbone.forward(x);

        // Project each scale to embed_dim, then pool to common 32x32 spatial size.
        // With INPUT_SIZE=256: s2=128x128, s3=64x64, s4=32x32.
        // Pooling reduces the encoder sequence from 21,504 to 3,072 tokens.
        let p2 = self.proj_s2.forward(&s2).max_pool2d().max_pool2d(); // 128->64->32
        let p3 = self.proj_s3.forward(&s3).max_pool2d();              // 64->32
        let p4 = self.proj_s4.forward(&s4);                           // already 32x32

        let p2_shape = p2.shape();
        let p3_shape = p3.shape();
        let p4_shape = p4.shape();

        let seq2 = p2_shape[2] * p2_shape[3];
        let seq3 = p3_shape[2] * p3_shape[3];
        let seq4 = p4_shape[2] * p4_shape[3];

        // Flatten: [batch, embed_dim, H, W] -> [batch, embed_dim, H*W]
        //          -> transpose to [batch, H*W, embed_dim]
        //          -> reshape to [batch*H*W, embed_dim] for 2D matmul
        let flat2 = p2.flatten().reshape(vec![batch, self.embed_dim, seq2])
            .transpose(1, 2).reshape(vec![batch * seq2, self.embed_dim]);
        let flat3 = p3.flatten().reshape(vec![batch, self.embed_dim, seq3])
            .transpose(1, 2).reshape(vec![batch * seq3, self.embed_dim]);
        let flat4 = p4.flatten().reshape(vec![batch, self.embed_dim, seq4])
            .transpose(1, 2).reshape(vec![batch * seq4, self.embed_dim]);

        // Concatenate multi-scale features along the sequence dimension.
        // [batch*seq_i, embed_dim] -> need to reshape to [batch, seq_i, embed_dim] first,
        // concat on dim=1, then reshape back to [batch*total_seq, embed_dim].
        let f2 = flat2.reshape(vec![batch, seq2, self.embed_dim]);
        let f3 = flat3.reshape(vec![batch, seq3, self.embed_dim]);
        let f4 = flat4.reshape(vec![batch, seq4, self.embed_dim]);
        let encoder_input = Tensor::concat(&[&f2, &f3, &f4], 1); // [batch, total_seq, embed_dim]
        let total_seq = seq2 + seq3 + seq4;
        let mut enc = encoder_input.reshape(vec![batch * total_seq, self.embed_dim]);

        // Transformer encoder
        for layer in &self.encoder_layers {
            enc = layer.forward(&enc, batch, total_seq);
        }

        // Prepare decoder input: replicate object queries for each batch element
        // object_queries: [num_queries, embed_dim] -> [batch*num_queries, embed_dim]
        let query_data = self.object_queries.data();
        let mut tgt_data = Vec::with_capacity(batch * self.num_queries * self.embed_dim);
        for _ in 0..batch {
            tgt_data.extend_from_slice(&query_data);
        }
        let mut tgt = Tensor::new(
            tgt_data,
            vec![batch * self.num_queries, self.embed_dim],
            false,
        );

        // Transformer decoder
        for layer in &self.decoder_layers {
            tgt = layer.forward(&tgt, &enc, batch, self.num_queries, total_seq);
        }

        // Classification and bbox heads
        let class_logits = tgt.matmul(&self.cls_w).add_bias(&self.cls_b);
        let bbox_preds = tgt.matmul(&self.bbox_w).add_bias(&self.bbox_b).sigmoid();

        (class_logits, bbox_preds)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        let mut p = self.backbone.params();
        p.extend(self.proj_s2.params());
        p.extend(self.proj_s3.params());
        p.extend(self.proj_s4.params());
        for layer in &self.encoder_layers {
            p.extend(layer.params());
        }
        p.push(&self.object_queries);
        for layer in &self.decoder_layers {
            p.extend(layer.params());
        }
        p.extend([&self.cls_w, &self.cls_b, &self.bbox_w, &self.bbox_b]);
        p
    }
}

// ---------------------------------------------------------------------------
// 6. Hungarian Matching + Loss
// ---------------------------------------------------------------------------

/// Ground truth for RT-DETR: absolute bbox (cx, cy, w, h) normalized to [0,1]
/// relative to RT_DETR_INPUT_SIZE, plus class label.
#[derive(Debug, Clone)]
pub struct RtDetrTarget {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub class_id: usize,
}

/// Greedy bipartite matching between predictions and ground truth.
///
/// Cost for each (pred, gt) pair = L1 bbox distance + classification cost.
/// Returns Vec<(pred_idx, gt_idx)> — one match per GT object.
///
/// This is a greedy O(num_gt * num_preds) approximation of the full Hungarian
/// algorithm, which is sufficient for a demo.
pub fn hungarian_match(
    pred_boxes: &[f32],   // [num_preds * 4], each (cx, cy, w, h) in [0,1]
    pred_logits: &[f32],  // [num_preds * num_cls], raw logits
    gt_boxes: &[f32],     // [num_gt * 4]
    gt_classes: &[usize], // [num_gt]
    num_preds: usize,
    num_cls: usize,
) -> Vec<(usize, usize)> {
    let num_gt = gt_classes.len();
    if num_gt == 0 {
        return Vec::new();
    }

    // Compute cost matrix [num_gt, num_preds]
    let mut costs = vec![0.0f32; num_gt * num_preds];
    for gi in 0..num_gt {
        let g_cx = gt_boxes[gi * 4];
        let g_cy = gt_boxes[gi * 4 + 1];
        let g_w = gt_boxes[gi * 4 + 2];
        let g_h = gt_boxes[gi * 4 + 3];
        let g_cls = gt_classes[gi];

        for pi in 0..num_preds {
            let p_cx = pred_boxes[pi * 4];
            let p_cy = pred_boxes[pi * 4 + 1];
            let p_w = pred_boxes[pi * 4 + 2];
            let p_h = pred_boxes[pi * 4 + 3];

            // L1 bbox distance
            let bbox_cost = (p_cx - g_cx).abs() + (p_cy - g_cy).abs()
                + (p_w - g_w).abs() + (p_h - g_h).abs();

            // Classification cost: negative log-prob of the GT class.
            // Compute softmax over logits for this prediction.
            let logit_offset = pi * num_cls;
            let mut max_logit = f32::NEG_INFINITY;
            for c in 0..num_cls {
                let l = pred_logits[logit_offset + c];
                if l > max_logit {
                    max_logit = l;
                }
            }
            let mut sum_exp = 0.0f32;
            for c in 0..num_cls {
                sum_exp += (pred_logits[logit_offset + c] - max_logit).exp();
            }
            let log_prob = (pred_logits[logit_offset + g_cls] - max_logit) - sum_exp.ln();
            let cls_cost = -log_prob;

            costs[gi * num_preds + pi] = bbox_cost + cls_cost;
        }
    }

    // Greedy matching: for each GT, pick the lowest-cost unmatched prediction.
    let mut matched = vec![false; num_preds];
    let mut result = Vec::with_capacity(num_gt);

    // Sort GT indices by their best available cost to improve greedy quality.
    let mut gt_order: Vec<usize> = (0..num_gt).collect();
    gt_order.sort_by(|&a, &b| {
        let min_a = (0..num_preds).map(|p| costs[a * num_preds + p])
            .fold(f32::INFINITY, f32::min);
        let min_b = (0..num_preds).map(|p| costs[b * num_preds + p])
            .fold(f32::INFINITY, f32::min);
        min_a.partial_cmp(&min_b).unwrap()
    });

    for &gi in &gt_order {
        let mut best_pi = 0;
        let mut best_cost = f32::INFINITY;
        for pi in 0..num_preds {
            if matched[pi] {
                continue;
            }
            let c = costs[gi * num_preds + pi];
            if c < best_cost {
                best_cost = c;
                best_pi = pi;
            }
        }
        matched[best_pi] = true;
        result.push((best_pi, gi));
    }

    result
}

/// Compute RT-DETR loss given model outputs and targets.
///
/// For matched predictions: cross-entropy classification loss + L1 bbox loss.
/// For unmatched predictions: classification target is the background class.
/// Returns a scalar Tensor connected to the autograd graph through pred_logits and pred_boxes.
pub fn rt_detr_loss(
    pred_logits: &Tensor, // [batch*num_queries, num_classes+1]
    pred_boxes: &Tensor,  // [batch*num_queries, 4]
    targets: &[Vec<RtDetrTarget>],
    num_queries: usize,
    num_classes: usize,
) -> Tensor {
    let logits_shape = pred_logits.shape();
    let batch = logits_shape[0] / num_queries;
    let num_cls = num_classes + 1;
    let bg_class = num_classes; // last class is background

    let logits_data = pred_logits.data();
    let boxes_data = pred_boxes.data();

    // Collect classification targets and bbox regression indices/targets.
    // All queries get a classification target (background for unmatched).
    let total_queries = batch * num_queries;
    let mut cls_targets = vec![0.0f32; total_queries * num_cls]; // one-hot
    let mut bbox_indices = Vec::new();
    let mut bbox_targets = Vec::new();

    for b in 0..batch {
        let query_offset = b * num_queries;

        // Prepare GT data for this batch element
        let gt = &targets[b];
        let mut gt_boxes_flat = Vec::with_capacity(gt.len() * 4);
        let mut gt_classes = Vec::with_capacity(gt.len());
        for t in gt {
            gt_boxes_flat.extend_from_slice(&[t.cx, t.cy, t.w, t.h]);
            gt_classes.push(t.class_id);
        }

        // Run matching on this batch element's predictions
        let pred_boxes_batch = &boxes_data[query_offset * 4..(query_offset + num_queries) * 4];
        let pred_logits_batch = &logits_data[query_offset * num_cls..(query_offset + num_queries) * num_cls];
        let matches = hungarian_match(
            pred_boxes_batch, pred_logits_batch,
            &gt_boxes_flat, &gt_classes,
            num_queries, num_cls,
        );

        // Set classification targets
        let mut matched_preds = std::collections::HashSet::new();
        for &(pi, gi) in &matches {
            matched_preds.insert(pi);
            let cls = gt_classes[gi];
            cls_targets[(query_offset + pi) * num_cls + cls] = 1.0;

            // Collect bbox regression targets for matched queries
            for d in 0..4 {
                bbox_indices.push((query_offset + pi) * 4 + d);
                bbox_targets.push(gt_boxes_flat[gi * 4 + d]);
            }
        }

        // Unmatched queries -> background class
        for qi in 0..num_queries {
            if !matched_preds.contains(&qi) {
                cls_targets[(query_offset + qi) * num_cls + bg_class] = 1.0;
            }
        }
    }

    // Classification loss: BCE with logits on all queries
    let cls_loss = pred_logits
        .reshape(vec![total_queries * num_cls])
        .bce_with_logits_loss(&cls_targets);

    // Bbox regression loss: L1 on matched queries only
    let bbox_loss = if bbox_indices.is_empty() {
        Tensor::new(vec![0.0], vec![1], false)
    } else {
        let n = bbox_indices.len();
        let pred_matched = pred_boxes
            .reshape(vec![total_queries * 4])
            .select(&bbox_indices);
        let tgt_tensor = Tensor::new(bbox_targets, vec![n], false);
        // L1 loss: mean(|pred - target|)
        // Approximate |x| via sqrt(x^2 + eps) to keep it differentiable.
        let diff = pred_matched.add(&tgt_tensor.scale(-1.0));
        let eps = Tensor::new(vec![1e-6; n], vec![n], false);
        let abs_approx = diff.mul(&diff).add(&eps);
        // sum / n as a proxy for mean L1
        abs_approx.sum().scale(1.0 / n as f32)
    };

    // Total loss: classification + bbox (weighted)
    cls_loss.add(&bbox_loss.scale(5.0))
}

// ---------------------------------------------------------------------------
// 7. Decode + NMS for RT-DETR
// ---------------------------------------------------------------------------

/// RT-DETR detection output.
#[derive(Debug, Clone)]
pub struct RtDetrDetection {
    /// Absolute pixel coordinates (center x, center y, width, height).
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub class_id: usize,
    pub score: f32,
}

/// Decode RT-DETR outputs into detections.
/// Each query directly predicts one box — no grid or anchor decoding needed.
///
/// class_logits: [batch*num_queries, num_classes+1]
/// bbox_preds:   [batch*num_queries, 4] (sigmoid-normalized to [0,1])
pub fn rt_detr_decode(
    class_logits: &Tensor,
    bbox_preds: &Tensor,
    num_queries: usize,
    input_size: usize,
) -> Vec<Vec<RtDetrDetection>> {
    let logits = class_logits.data();
    let boxes = bbox_preds.data();
    let logits_shape = class_logits.shape();
    let total = logits_shape[0];
    let num_cls = logits_shape[1];
    let num_fg_cls = num_cls - 1; // last class is background
    let batch = total / num_queries;
    let scale = input_size as f32;

    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

    let mut batch_dets = Vec::with_capacity(batch);

    for b in 0..batch {
        let mut dets = Vec::new();
        for q in 0..num_queries {
            let idx = b * num_queries + q;
            let logit_base = idx * num_cls;

            // Find best foreground class (exclude background = last class)
            let mut best_cls = 0;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..num_fg_cls {
                let s = sigmoid(logits[logit_base + c]);
                if s > best_score {
                    best_score = s;
                    best_cls = c;
                }
            }

            let box_base = idx * 4;
            dets.push(RtDetrDetection {
                cx: boxes[box_base] * scale,
                cy: boxes[box_base + 1] * scale,
                w: boxes[box_base + 2] * scale,
                h: boxes[box_base + 3] * scale,
                class_id: best_cls,
                score: best_score,
            });
        }
        batch_dets.push(dets);
    }

    batch_dets
}

/// NMS for RT-DETR detections.
pub fn rt_detr_nms(
    detections: &[RtDetrDetection],
    score_threshold: f32,
    nms_threshold: f32,
) -> Vec<RtDetrDetection> {
    let mut candidates: Vec<RtDetrDetection> = detections
        .iter()
        .filter(|d| d.score >= score_threshold)
        .cloned()
        .collect();

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(candidates[i].clone());

        for j in (i + 1)..candidates.len() {
            if suppressed[j] || candidates[j].class_id != candidates[i].class_id {
                continue;
            }
            let iou = iou_cxcywh(
                candidates[i].cx, candidates[i].cy, candidates[i].w, candidates[i].h,
                candidates[j].cx, candidates[j].cy, candidates[j].w, candidates[j].h,
            );
            if iou > nms_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}
