use peregrine::tensor::Tensor;
use std::collections::HashMap;

/// MXFP4 lookup table: 4-bit float values (3 sign/magnitude/exponent combos).
const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// MXFP4 quantized matrix-vector multiply: input [in_dim] × weight [out_dim, in_dim] → output [out_dim].
///
/// Weight is stored as packed nibble blocks + uint8 scales.
/// Each block = 32 FP4 values stored in 16 bytes, plus 1 scale byte.
/// Blocking is along the input (column) dimension.
fn mxfp4_matmul(
    input: &[f32],
    blocks: &[u8],
    scales: &[u8],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let n_blocks_per_row = in_dim / 32;
    let bytes_per_row = n_blocks_per_row * 16; // 16 bytes per block (32 nibbles)
    let mut output = vec![0.0f32; out_dim];

    for r in 0..out_dim {
        let row_block_offset = r * bytes_per_row;
        let row_scale_offset = r * n_blocks_per_row;
        let mut row_sum = 0.0f32;

        for b in 0..n_blocks_per_row {
            let block_start = row_block_offset + b * 16;
            let scale_byte = scales[row_scale_offset + b];
            let scale = f32::from_bits((scale_byte as u32) << 23);
            let input_offset = b * 32;

            let mut block_dot = 0.0f32;
            for k in 0..16 {
                let byte = blocks[block_start + k];
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                block_dot += FP4_LUT[lo] * input[input_offset + 2 * k];
                block_dot += FP4_LUT[hi] * input[input_offset + 2 * k + 1];
            }

            row_sum += block_dot * scale;
        }

        output[r] = row_sum;
    }

    output
}

/// MXFP4 quantized expert: stores mlp1/mlp2 as packed nibble blocks + scales.
pub struct Mxfp4Expert {
    pub mlp1_blocks: Vec<u8>,  // weight [intermediate*2, hidden] packed
    pub mlp1_scales: Vec<u8>,
    pub mlp2_blocks: Vec<u8>,  // weight [hidden, intermediate] packed
    pub mlp2_scales: Vec<u8>,
    pub mlp1_bias: Vec<f32>,   // [intermediate*2]
    pub mlp2_bias: Vec<f32>,   // [hidden]
    intermediate: usize,
    hidden: usize,
    alpha: f32,
    clamp: f32,
}

impl Mxfp4Expert {
    pub fn new(hidden: usize, intermediate: usize, alpha: f32, clamp: f32) -> Self {
        Mxfp4Expert {
            mlp1_blocks: Vec::new(),
            mlp1_scales: Vec::new(),
            mlp2_blocks: Vec::new(),
            mlp2_scales: Vec::new(),
            mlp1_bias: vec![0.0; intermediate * 2],
            mlp2_bias: vec![0.0; hidden],
            intermediate,
            hidden,
            alpha,
            clamp,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // x @ mlp1 + bias → swiglu → x @ mlp2 + bias
        let inter2 = self.intermediate * 2;
        let h = mxfp4_matmul(x, &self.mlp1_blocks, &self.mlp1_scales, inter2, self.hidden);

        let mut biased = vec![0.0f32; inter2];
        for i in 0..inter2 {
            biased[i] = h[i] + self.mlp1_bias[i];
        }

        let activated = swiglu(&biased, self.intermediate, self.alpha, self.clamp);

        let out = mxfp4_matmul(
            &activated,
            &self.mlp2_blocks,
            &self.mlp2_scales,
            self.hidden,
            self.intermediate,
        );

        let mut result = vec![0.0f32; self.hidden];
        for i in 0..self.hidden {
            result[i] = out[i] + self.mlp2_bias[i];
        }
        result
    }

    pub fn load_weights(
        &mut self,
        params: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        prefix: &str,
    ) {
        // Blocks/scales are stored as f32 bytes packed into the binary — reinterpret as u8
        if let Some((_shape, data)) = params.get(&format!("{}.mlp1_blocks", prefix)) {
            self.mlp1_blocks = reinterpret_f32_as_u8(data);
        }
        if let Some((_shape, data)) = params.get(&format!("{}.mlp1_scales", prefix)) {
            self.mlp1_scales = reinterpret_f32_as_u8(data);
        }
        if let Some((_shape, data)) = params.get(&format!("{}.mlp2_blocks", prefix)) {
            self.mlp2_blocks = reinterpret_f32_as_u8(data);
        }
        if let Some((_shape, data)) = params.get(&format!("{}.mlp2_scales", prefix)) {
            self.mlp2_scales = reinterpret_f32_as_u8(data);
        }
        if let Some((_shape, data)) = params.get(&format!("{}.mlp1_bias", prefix)) {
            self.mlp1_bias = data.clone();
        }
        if let Some((_shape, data)) = params.get(&format!("{}.mlp2_bias", prefix)) {
            self.mlp2_bias = data.clone();
        }
    }
}

/// Reinterpret f32 slice as raw bytes (inverse of Python's pack-u8-as-f32).
fn reinterpret_f32_as_u8(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &f in data {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

/// Custom SwiGLU activation for gpt-oss.
/// Input `data` has interleaved gate/linear pairs: [gate0, linear0, gate1, linear1, ...].
/// Output has half the size.
///   gate = data[2*i].min(clamp)  (upper clamp only)
///   linear = data[2*i+1].clamp(-clamp, clamp)
///   activated = gate * sigmoid(alpha * gate)
///   out[i] = activated * (linear + 1.0)
fn swiglu(data: &[f32], out_dim: usize, alpha: f32, clamp: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let gate = data[2 * i].min(clamp);
        let linear = data[2 * i + 1].clamp(-clamp, clamp);
        let activated = gate * (1.0 / (1.0 + (-alpha * gate).exp())); // gate * sigmoid(alpha * gate)
        out[i] = activated * (linear + 1.0);
    }
    out
}

/// Single expert FFN: x @ mlp1 + bias → swiglu → x @ mlp2 + bias.
pub struct Expert {
    pub mlp1: Tensor,      // [hidden, intermediate * 2]
    pub mlp1_bias: Tensor,  // [intermediate * 2]
    pub mlp2: Tensor,      // [intermediate, hidden]
    pub mlp2_bias: Tensor,  // [hidden]
    intermediate: usize,
    alpha: f32,
    clamp: f32,
}

impl Expert {
    pub fn new(hidden: usize, intermediate: usize, alpha: f32, clamp: f32) -> Self {
        Expert {
            mlp1: Tensor::zeros(&[hidden, intermediate * 2], false),
            mlp1_bias: Tensor::zeros(&[intermediate * 2], false),
            mlp2: Tensor::zeros(&[intermediate, hidden], false),
            mlp2_bias: Tensor::zeros(&[hidden], false),
            intermediate,
            alpha,
            clamp,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let hidden = x.shape()[1];

        // x @ mlp1 + bias
        let h = x.matmul(&self.mlp1);
        let h_data = h.data();
        let bias1 = self.mlp1_bias.data();

        let mut biased = vec![0.0f32; h_data.len()];
        let inter2 = self.intermediate * 2;
        for i in 0..h_data.len() {
            biased[i] = h_data[i] + bias1[i % inter2];
        }

        // SwiGLU activation
        let activated = swiglu(&biased, self.intermediate, self.alpha, self.clamp);

        // activated @ mlp2 + bias
        let act_tensor = Tensor::new(activated, vec![1, self.intermediate], false);
        let out = act_tensor.matmul(&self.mlp2);
        let out_data = out.data();
        let bias2 = self.mlp2_bias.data();

        let mut result = vec![0.0f32; hidden];
        for i in 0..hidden {
            result[i] = out_data[i] + bias2[i];
        }

        Tensor::new(result, vec![1, hidden], false)
    }

    pub fn load_weights(
        &mut self,
        params: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        prefix: &str,
    ) {
        if let Some((shape, data)) = params.get(&format!("{}.mlp1", prefix)) {
            self.mlp1 = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.mlp1_bias", prefix)) {
            self.mlp1_bias = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.mlp2", prefix)) {
            self.mlp2 = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.mlp2_bias", prefix)) {
            self.mlp2_bias = Tensor::new(data.clone(), shape.clone(), false);
        }
    }
}

/// Mixture-of-Experts layer with softmax top-k routing.
/// Supports both f32 (full precision) and MXFP4 (quantized) expert paths.
pub struct MoELayer {
    pub gate: Tensor,      // [hidden, num_experts]
    pub gate_bias: Tensor,  // [num_experts]
    pub experts: Vec<Expert>,
    pub mxfp4_experts: Vec<Mxfp4Expert>,
    pub quantized: bool,
    num_experts: usize,
    top_k: usize,
    hidden: usize,
}

impl MoELayer {
    pub fn new(
        hidden: usize,
        intermediate: usize,
        num_experts: usize,
        top_k: usize,
        alpha: f32,
        clamp: f32,
    ) -> Self {
        let experts = (0..num_experts)
            .map(|_| Expert::new(hidden, intermediate, alpha, clamp))
            .collect();
        MoELayer {
            gate: Tensor::zeros(&[hidden, num_experts], false),
            gate_bias: Tensor::zeros(&[num_experts], false),
            experts,
            mxfp4_experts: Vec::new(),
            quantized: false,
            num_experts,
            top_k,
            hidden,
        }
    }

    /// Initialize empty MXFP4 experts (call before loading quantized weights).
    pub fn init_mxfp4_experts(&mut self, intermediate: usize, alpha: f32, clamp: f32) {
        self.mxfp4_experts = (0..self.num_experts)
            .map(|_| Mxfp4Expert::new(self.hidden, intermediate, alpha, clamp))
            .collect();
        self.quantized = true;
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let seq_len = shape[0];

        // Router: logits = x @ gate + bias -> [seq_len, num_experts]
        let logits = x.matmul(&self.gate);
        let logits_data = logits.data();
        let gate_bias_data = self.gate_bias.data();

        let x_data = x.data();
        let mut output_data = vec![0.0f32; seq_len * self.hidden];

        for t in 0..seq_len {
            // Add bias to logits
            let mut token_logits = vec![0.0f32; self.num_experts];
            for e in 0..self.num_experts {
                token_logits[e] = logits_data[t * self.num_experts + e] + gate_bias_data[e];
            }

            // Find top-k expert indices
            let mut expert_indices: Vec<(usize, f32)> = token_logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            expert_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_k_experts: Vec<(usize, f32)> =
                expert_indices[..self.top_k].to_vec();

            // Softmax over top-k logits for weights
            let max_logit = top_k_experts
                .iter()
                .map(|&(_, l)| l)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = top_k_experts
                .iter()
                .map(|&(_, l)| (l - max_logit).exp())
                .sum();
            let weights: Vec<(usize, f32)> = top_k_experts
                .iter()
                .map(|&(idx, l)| (idx, (l - max_logit).exp() / exp_sum))
                .collect();

            let token_slice = &x_data[t * self.hidden..(t + 1) * self.hidden];

            let out_offset = t * self.hidden;
            for &(expert_idx, weight) in &weights {
                if self.quantized {
                    let expert_out = self.mxfp4_experts[expert_idx].forward(token_slice);
                    for d in 0..self.hidden {
                        output_data[out_offset + d] += weight * expert_out[d];
                    }
                } else {
                    let token_input = Tensor::new(
                        token_slice.to_vec(),
                        vec![1, self.hidden],
                        false,
                    );
                    let expert_out = self.experts[expert_idx].forward(&token_input);
                    let expert_data = expert_out.data();
                    for d in 0..self.hidden {
                        output_data[out_offset + d] += weight * expert_data[d];
                    }
                }
            }
        }

        Tensor::new(output_data, vec![seq_len, self.hidden], false)
    }

    pub fn load_weights(
        &mut self,
        params: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        prefix: &str,
    ) {
        if let Some((shape, data)) = params.get(&format!("{}.gate", prefix)) {
            self.gate = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.gate_bias", prefix)) {
            self.gate_bias = Tensor::new(data.clone(), shape.clone(), false);
        }
        if self.quantized {
            for j in 0..self.num_experts {
                let expert_prefix = format!("{}.experts.{}", prefix, j);
                self.mxfp4_experts[j].load_weights(params, &expert_prefix);
            }
        } else {
            for j in 0..self.num_experts {
                let expert_prefix = format!("{}.experts.{}", prefix, j);
                self.experts[j].load_weights(params, &expert_prefix);
            }
        }
    }
}
