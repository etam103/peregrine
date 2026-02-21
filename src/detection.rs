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
pub const INPUT_SIZE: usize = 416;

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

/// Learnable parameters for a single Conv2d + ReLU + MaxPool block.
struct ConvBlock {
    /// Kernel weights: [out_channels, in_channels, kH, kW]
    weight: Tensor,
    /// Bias: [out_channels]
    bias: Tensor,
}

impl ConvBlock {
    fn new(in_ch: usize, out_ch: usize, kernel_size: usize) -> Self {
        ConvBlock {
            weight: Tensor::randn(&[out_ch, in_ch, kernel_size, kernel_size], true),
            bias: Tensor::zeros(&[out_ch], true),
        }
    }

    /// Forward: Conv2d -> ReLU -> 2x2 MaxPool.
    fn forward(&self, x: &Tensor) -> Tensor {
        x.conv2d(&self.weight, &self.bias).relu().max_pool2d()
    }

    fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }
}

/// Tiny YOLO-style network.
///
/// Architecture (no batch norm — just conv -> relu -> maxpool):
///   Conv2d(3, 16, 3)  -> ReLU -> MaxPool  =>  (416-2)/2 = 207 ... needs padding
///
/// IMPORTANT: The conv2d in tensor.rs uses no padding and stride=1.
/// With 5 rounds of maxpool (each /2) we need the spatial dims to remain
/// even at each stage.  Starting from 416:
///
///   Stage 0: [B, 3, 416, 416]
///   Conv(3, 16, 3):  416 - 3 + 1 = 414 -> MaxPool -> 207 (ODD! breaks pool)
///
/// To work around the no-padding conv, we use kernel_size=1 for intermediate
/// layers so spatial dims are preserved, and kernel_size=3 only where the
/// resulting size stays even before pooling. Alternatively, we pick an input
/// size that works cleanly. With all k=3 convs:
///   After conv: W - 2.  After pool: (W-2)/2.
///   416 -> 414 -> 207  (odd, bad)
///
/// A clean approach: use 1x1 convs everywhere except where we can accommodate
/// the shrinkage, or adjust input to 418 which gives:
///   418 -> 416 -> 208 -> 206 -> 103 (odd again)
///
/// Simplest correct design: use kernel_size=1 convs (which preserve spatial
/// dims perfectly) for the feature extraction layers, and a final 1x1 conv
/// for the detection head. The pooling does the spatial reduction.
/// Input 416 -> pool -> 208 -> pool -> 104 -> pool -> 52 -> pool -> 26
/// -> pool -> 13.  Exactly our target grid.
///
/// We use 1x1 convs for simplicity since the focus is on the detection head
/// logic, not on the backbone architecture.
pub struct YoloNet {
    conv1: ConvBlock, // 3  -> 16,  k=1
    conv2: ConvBlock, // 16 -> 32,  k=1
    conv3: ConvBlock, // 32 -> 64,  k=1
    conv4: ConvBlock, // 64 -> 128, k=1
    conv5: ConvBlock, // 128-> 256, k=1
    // Final 1x1 detection conv (no relu, no pool).
    det_weight: Tensor,
    det_bias: Tensor,
    pub num_classes: usize,
}

impl YoloNet {
    pub fn new(num_classes: usize) -> Self {
        let out_ch = NUM_ANCHORS * (5 + num_classes);
        YoloNet {
            conv1: ConvBlock::new(3, 16, 1),
            conv2: ConvBlock::new(16, 32, 1),
            conv3: ConvBlock::new(32, 64, 1),
            conv4: ConvBlock::new(64, 128, 1),
            conv5: ConvBlock::new(128, 256, 1),
            det_weight: Tensor::randn(&[out_ch, 256, 1, 1], true),
            det_bias: Tensor::zeros(&[out_ch], true),
            num_classes,
        }
    }

    /// Forward pass: returns raw logits of shape
    ///   [batch, num_anchors * (5 + num_classes), 13, 13]
    ///
    /// Input must be [batch, 3, 416, 416].
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 416 -> pool -> 208 -> pool -> 104 -> pool -> 52 -> pool -> 26 -> pool -> 13
        let h1 = self.conv1.forward(x);
        let h2 = self.conv2.forward(&h1);
        let h3 = self.conv3.forward(&h2);
        let h4 = self.conv4.forward(&h3);
        let h5 = self.conv5.forward(&h4);
        // Detection head: 1x1 conv, no activation, no pooling.
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
