use crate::nn::RMSNorm;
use crate::tensor::Tensor;

use super::attention::{KVCache, LlamaAttention};

/// One Llama decoder layer: pre-norm GQA attention + pre-norm SwiGLU FFN with residuals.
pub struct LlamaBlock {
    pub attn_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
    pub attention: LlamaAttention,
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
    gate_up_fused: Option<Tensor>,
    ffn_dim: usize,
}

impl LlamaBlock {
    pub fn new(
        model_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        ffn_dim: usize,
        max_seq_len: usize,
        rope_base: f32,
        rms_eps: f32,
    ) -> Self {
        LlamaBlock {
            attn_norm: RMSNorm::new(model_dim, rms_eps),
            ffn_norm: RMSNorm::new(model_dim, rms_eps),
            attention: LlamaAttention::new(
                model_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                max_seq_len,
                rope_base,
            ),
            gate_proj: Tensor::zeros(&[model_dim, ffn_dim], false),
            up_proj: Tensor::zeros(&[model_dim, ffn_dim], false),
            down_proj: Tensor::zeros(&[ffn_dim, model_dim], false),
            gate_up_fused: None,
            ffn_dim,
        }
    }

    /// Fuse gate_proj and up_proj into a single [model_dim, 2*ffn_dim] weight matrix.
    pub fn fuse_gate_up(&mut self) {
        let gate_data = self.gate_proj.data();
        let up_data = self.up_proj.data();
        let gate_shape = self.gate_proj.shape();
        let model_dim = gate_shape[0];
        let ffn_dim = gate_shape[1];
        self.ffn_dim = ffn_dim;

        let mut fused = vec![0.0f32; model_dim * 2 * ffn_dim];
        for r in 0..model_dim {
            let src_off = r * ffn_dim;
            let dst_off = r * 2 * ffn_dim;
            fused[dst_off..dst_off + ffn_dim]
                .copy_from_slice(&gate_data[src_off..src_off + ffn_dim]);
            fused[dst_off + ffn_dim..dst_off + 2 * ffn_dim]
                .copy_from_slice(&up_data[src_off..src_off + ffn_dim]);
        }
        self.gate_up_fused = Some(Tensor::new(fused, vec![model_dim, 2 * ffn_dim], false));
    }

    pub fn forward(&self, h: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        let normed = self.attn_norm.forward(h);
        let attn_out = self.attention.forward(&normed, kv_cache);
        let h = h.add(&attn_out);

        let normed = self.ffn_norm.forward(&h);

        let (gate_data, up_data, ffn_dim);
        if let Some(ref fused) = self.gate_up_fused {
            let gate_up = normed.matmul(fused);
            let gu_data = gate_up.data();
            let seq_len = gate_up.shape()[0];
            ffn_dim = self.ffn_dim;
            let n = seq_len * ffn_dim;
            let mut g = vec![0.0f32; n];
            let mut u = vec![0.0f32; n];
            for s in 0..seq_len {
                let src = s * 2 * ffn_dim;
                let dst = s * ffn_dim;
                g[dst..dst + ffn_dim].copy_from_slice(&gu_data[src..src + ffn_dim]);
                u[dst..dst + ffn_dim].copy_from_slice(&gu_data[src + ffn_dim..src + 2 * ffn_dim]);
            }
            gate_data = g;
            up_data = u;
        } else {
            let gate = normed.matmul(&self.gate_proj);
            let up = normed.matmul(&self.up_proj);
            ffn_dim = gate.shape()[1];
            gate_data = gate.data().to_vec();
            up_data = up.data().to_vec();
        }

        let n = gate_data.len();
        let mut ffn_act = vec![0.0f32; n];
        let seq_len = n / ffn_dim;

        #[cfg(target_os = "macos")]
        {
            extern "C" { fn vvexpf(result: *mut f32, input: *const f32, count: *const i32); }
            for i in 0..n { ffn_act[i] = -gate_data[i]; }
            let ni = n as i32;
            unsafe { vvexpf(ffn_act.as_mut_ptr(), ffn_act.as_ptr(), &ni); }

            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let chunks4 = n / 4;
                let ones = unsafe { vdupq_n_f32(1.0) };
                for i in 0..chunks4 {
                    let off = i * 4;
                    unsafe {
                        let vg = vld1q_f32(gate_data.as_ptr().add(off));
                        let vu = vld1q_f32(up_data.as_ptr().add(off));
                        let vexp = vld1q_f32(ffn_act.as_ptr().add(off));
                        let denom = vaddq_f32(ones, vexp);
                        let sigmoid = vdivq_f32(ones, denom);
                        let silu = vmulq_f32(vg, sigmoid);
                        let result = vmulq_f32(silu, vu);
                        vst1q_f32(ffn_act.as_mut_ptr().add(off), result);
                    }
                }
                for i in (chunks4 * 4)..n {
                    let g = gate_data[i];
                    ffn_act[i] = g / (1.0 + ffn_act[i]) * up_data[i];
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..n {
                    let g = gate_data[i];
                    ffn_act[i] = g / (1.0 + ffn_act[i]) * up_data[i];
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            for i in 0..n {
                let g = gate_data[i];
                let silu = g / (1.0 + (-g).exp());
                ffn_act[i] = silu * up_data[i];
            }
        }

        let ffn_hidden = Tensor::new(ffn_act, vec![seq_len, ffn_dim], false);
        let ffn_out = ffn_hidden.matmul(&self.down_proj);
        h.add(&ffn_out)
    }
}
