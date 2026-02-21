use peregrine::tensor::Tensor;
use peregrine::nn::{TransformerEncoderLayer, TransformerDecoderLayer};

pub const INPUT_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// ResNet Backbone
// ---------------------------------------------------------------------------

struct ResidualBlock {
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
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

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = x.conv2d(&self.conv1_w, &self.conv1_b).relu();
        let h = h.conv2d(&self.conv2_w, &self.conv2_b);

        let skip = if let (Some(w), Some(b)) = (&self.skip_w, &self.skip_b) {
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

struct ResNetBackbone {
    stem_w: Tensor,
    stem_b: Tensor,
    stage2_blocks: Vec<ResidualBlock>,
    stage3_blocks: Vec<ResidualBlock>,
    stage4_blocks: Vec<ResidualBlock>,
}

impl ResNetBackbone {
    fn new() -> Self {
        ResNetBackbone {
            stem_w: Tensor::randn(&[64, 3, 1, 1], true),
            stem_b: Tensor::zeros(&[64], true),
            stage2_blocks: vec![
                ResidualBlock::new(64, 128),
                ResidualBlock::new(128, 128),
            ],
            stage3_blocks: vec![
                ResidualBlock::new(128, 256),
                ResidualBlock::new(256, 256),
            ],
            stage4_blocks: vec![
                ResidualBlock::new(256, 512),
                ResidualBlock::new(512, 512),
            ],
        }
    }

    fn forward(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        let h = x.conv2d(&self.stem_w, &self.stem_b).relu().max_pool2d();

        let mut h = h;
        for block in &self.stage2_blocks {
            h = block.forward(&h);
        }
        let s2 = h.clone();
        let h = s2.max_pool2d();

        let mut h = h;
        for block in &self.stage3_blocks {
            h = block.forward(&h);
        }
        let s3 = h.clone();
        let h = s3.max_pool2d();

        let mut h = h;
        for block in &self.stage4_blocks {
            h = block.forward(&h);
        }
        let s4 = h;

        (s2, s3, s4)
    }

    fn params(&self) -> Vec<&Tensor> {
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
// Channel Projection
// ---------------------------------------------------------------------------

struct ChannelProjection {
    weight: Tensor,
    bias: Tensor,
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

// ---------------------------------------------------------------------------
// RtDetrNet
// ---------------------------------------------------------------------------

pub struct RtDetrNet {
    backbone: ResNetBackbone,
    proj_s2: ChannelProjection,
    proj_s3: ChannelProjection,
    proj_s4: ChannelProjection,
    encoder_layers: Vec<TransformerEncoderLayer>,
    object_queries: Tensor,
    decoder_layers: Vec<TransformerDecoderLayer>,
    cls_w: Tensor,
    cls_b: Tensor,
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
        let cls_out = num_classes + 1;
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

    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let batch = x.shape()[0];

        let (s2, s3, s4) = self.backbone.forward(x);

        let p2 = self.proj_s2.forward(&s2).max_pool2d().max_pool2d();
        let p3 = self.proj_s3.forward(&s3).max_pool2d();
        let p4 = self.proj_s4.forward(&s4);

        let p2_shape = p2.shape();
        let p3_shape = p3.shape();
        let p4_shape = p4.shape();

        let seq2 = p2_shape[2] * p2_shape[3];
        let seq3 = p3_shape[2] * p3_shape[3];
        let seq4 = p4_shape[2] * p4_shape[3];

        let flat2 = p2.flatten().reshape(vec![batch, self.embed_dim, seq2])
            .transpose(1, 2).reshape(vec![batch * seq2, self.embed_dim]);
        let flat3 = p3.flatten().reshape(vec![batch, self.embed_dim, seq3])
            .transpose(1, 2).reshape(vec![batch * seq3, self.embed_dim]);
        let flat4 = p4.flatten().reshape(vec![batch, self.embed_dim, seq4])
            .transpose(1, 2).reshape(vec![batch * seq4, self.embed_dim]);

        let f2 = flat2.reshape(vec![batch, seq2, self.embed_dim]);
        let f3 = flat3.reshape(vec![batch, seq3, self.embed_dim]);
        let f4 = flat4.reshape(vec![batch, seq4, self.embed_dim]);
        let encoder_input = Tensor::concat(&[&f2, &f3, &f4], 1);
        let total_seq = seq2 + seq3 + seq4;
        let mut enc = encoder_input.reshape(vec![batch * total_seq, self.embed_dim]);

        for layer in &self.encoder_layers {
            enc = layer.forward(&enc, batch, total_seq);
        }

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

        for layer in &self.decoder_layers {
            tgt = layer.forward(&tgt, &enc, batch, self.num_queries, total_seq);
        }

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
// Hungarian Matching + Loss
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RtDetrTarget {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub class_id: usize,
}

fn hungarian_match(
    pred_boxes: &[f32],
    pred_logits: &[f32],
    gt_boxes: &[f32],
    gt_classes: &[usize],
    num_preds: usize,
    num_cls: usize,
) -> Vec<(usize, usize)> {
    let num_gt = gt_classes.len();
    if num_gt == 0 {
        return Vec::new();
    }

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

            let bbox_cost = (p_cx - g_cx).abs() + (p_cy - g_cy).abs()
                + (p_w - g_w).abs() + (p_h - g_h).abs();

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

    let mut matched = vec![false; num_preds];
    let mut result = Vec::with_capacity(num_gt);

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

pub fn rt_detr_loss(
    pred_logits: &Tensor,
    pred_boxes: &Tensor,
    targets: &[Vec<RtDetrTarget>],
    num_queries: usize,
    num_classes: usize,
) -> Tensor {
    let logits_shape = pred_logits.shape();
    let batch = logits_shape[0] / num_queries;
    let num_cls = num_classes + 1;
    let bg_class = num_classes;

    let logits_data = pred_logits.data();
    let boxes_data = pred_boxes.data();

    let total_queries = batch * num_queries;
    let mut cls_targets = vec![0.0f32; total_queries * num_cls];
    let mut bbox_indices = Vec::new();
    let mut bbox_targets = Vec::new();

    for b in 0..batch {
        let query_offset = b * num_queries;

        let gt = &targets[b];
        let mut gt_boxes_flat = Vec::with_capacity(gt.len() * 4);
        let mut gt_classes = Vec::with_capacity(gt.len());
        for t in gt {
            gt_boxes_flat.extend_from_slice(&[t.cx, t.cy, t.w, t.h]);
            gt_classes.push(t.class_id);
        }

        let pred_boxes_batch = &boxes_data[query_offset * 4..(query_offset + num_queries) * 4];
        let pred_logits_batch = &logits_data[query_offset * num_cls..(query_offset + num_queries) * num_cls];
        let matches = hungarian_match(
            pred_boxes_batch, pred_logits_batch,
            &gt_boxes_flat, &gt_classes,
            num_queries, num_cls,
        );

        let mut matched_preds = std::collections::HashSet::new();
        for &(pi, gi) in &matches {
            matched_preds.insert(pi);
            let cls = gt_classes[gi];
            cls_targets[(query_offset + pi) * num_cls + cls] = 1.0;

            for d in 0..4 {
                bbox_indices.push((query_offset + pi) * 4 + d);
                bbox_targets.push(gt_boxes_flat[gi * 4 + d]);
            }
        }

        for qi in 0..num_queries {
            if !matched_preds.contains(&qi) {
                cls_targets[(query_offset + qi) * num_cls + bg_class] = 1.0;
            }
        }
    }

    let cls_loss = pred_logits
        .reshape(vec![total_queries * num_cls])
        .bce_with_logits_loss(&cls_targets);

    let bbox_loss = if bbox_indices.is_empty() {
        Tensor::new(vec![0.0], vec![1], false)
    } else {
        let n = bbox_indices.len();
        let pred_matched = pred_boxes
            .reshape(vec![total_queries * 4])
            .select(&bbox_indices);
        let tgt_tensor = Tensor::new(bbox_targets, vec![n], false);
        let diff = pred_matched.add(&tgt_tensor.scale(-1.0));
        let eps = Tensor::new(vec![1e-6; n], vec![n], false);
        let abs_approx = diff.mul(&diff).add(&eps);
        abs_approx.sum().scale(1.0 / n as f32)
    };

    cls_loss.add(&bbox_loss.scale(5.0))
}

// ---------------------------------------------------------------------------
// Decode + NMS
// ---------------------------------------------------------------------------

fn iou(a_cx: f32, a_cy: f32, a_w: f32, a_h: f32,
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

#[derive(Debug, Clone)]
pub struct RtDetrDetection {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub class_id: usize,
    pub score: f32,
}

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
    let num_fg_cls = num_cls - 1;
    let batch = total / num_queries;
    let scale = input_size as f32;

    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

    let mut batch_dets = Vec::with_capacity(batch);

    for b in 0..batch {
        let mut dets = Vec::new();
        for q in 0..num_queries {
            let idx = b * num_queries + q;
            let logit_base = idx * num_cls;

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
            let v = iou(
                candidates[i].cx, candidates[i].cy, candidates[i].w, candidates[i].h,
                candidates[j].cx, candidates[j].cy, candidates[j].w, candidates[j].h,
            );
            if v > nms_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}
