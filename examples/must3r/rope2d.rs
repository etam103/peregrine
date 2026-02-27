/// 2D Rotary Position Embeddings as used in MUSt3R / CroCo / DUSt3R.
///
/// Splits each head dimension into two halves:
///   - First half uses y-positions
///   - Second half uses x-positions
/// Each half applies standard 1D RoPE with frequencies:
///   theta_i = 1.0 / (freq ^ (2*i / half_dim))
///
/// Applied to Q and K tensors AFTER projection, BEFORE attention computation.

pub struct RoPE2D {
    freq: f32, // base frequency, default 100.0
}

impl RoPE2D {
    pub fn new(freq: f32) -> Self {
        RoPE2D { freq }
    }

    /// Create a RoPE2D from num_heads and head_dim (uses default freq=100.0).
    /// This is a convenience constructor used by the decoder.
    pub fn from_config(_num_heads: usize, _head_dim: usize) -> Self {
        RoPE2D { freq: 100.0 }
    }

    /// Apply 2D RoPE in-place on data laid out as [batch * seq_len, embed_dim].
    ///
    /// This variant works with the decoder's data layout where:
    /// - `x_data` is flattened [batch * seq_len, embed_dim] (embed_dim = num_heads * head_dim)
    /// - `positions` is [batch * seq_len * 2] (y, x pairs per token)
    ///
    /// RoPE is applied per-head to the head_dim portion of each token.
    pub fn apply_in_place(
        &self,
        x_data: &mut [f32],
        positions: &[f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        let embed_dim = num_heads * head_dim;
        let total_tokens = batch * seq_len;
        assert_eq!(
            x_data.len(),
            total_tokens * embed_dim,
            "RoPE2D::apply_in_place: x_data length mismatch"
        );

        let half_dim = head_dim / 2;
        assert!(half_dim % 2 == 0, "RoPE2D::apply_in_place: half_dim must be even");
        let num_pairs = half_dim / 2;

        let inv_freq: Vec<f32> = (0..num_pairs)
            .map(|i| 1.0 / self.freq.powf(2.0 * i as f32 / half_dim as f32))
            .collect();

        for b in 0..batch {
            for s in 0..seq_len {
                let token_idx = b * seq_len + s;
                // Use positions from the flat positions array (may be [seq_len*2] or [batch*seq_len*2])
                let pos_idx = if positions.len() == seq_len * 2 {
                    s * 2  // shared positions across batch
                } else {
                    token_idx * 2
                };
                let pos_y = positions[pos_idx];
                let pos_x = positions[pos_idx + 1];

                for h in 0..num_heads {
                    let base = token_idx * embed_dim + h * head_dim;

                    // First half: y-positions
                    for i in 0..num_pairs {
                        let idx0 = base + 2 * i;
                        let idx1 = base + 2 * i + 1;
                        let theta = pos_y * inv_freq[i];
                        let c = theta.cos();
                        let si = theta.sin();
                        let x0 = x_data[idx0];
                        let x1 = x_data[idx1];
                        x_data[idx0] = x0 * c - x1 * si;
                        x_data[idx1] = x0 * si + x1 * c;
                    }

                    // Second half: x-positions
                    for i in 0..num_pairs {
                        let idx0 = base + half_dim + 2 * i;
                        let idx1 = base + half_dim + 2 * i + 1;
                        let theta = pos_x * inv_freq[i];
                        let c = theta.cos();
                        let si = theta.sin();
                        let x0 = x_data[idx0];
                        let x1 = x_data[idx1];
                        x_data[idx0] = x0 * c - x1 * si;
                        x_data[idx1] = x0 * si + x1 * c;
                    }
                }
            }
        }
    }

    /// Apply 2D RoPE to tensor data.
    ///
    /// # Arguments
    /// * `x_data` - flattened [num_heads, seq_len, head_dim] data
    /// * `positions` - flattened [seq_len, 2] data where positions[t] = (y, x)
    /// * `num_heads` - number of attention heads
    /// * `seq_len` - sequence length (number of tokens)
    /// * `head_dim` - dimension per head (must be even)
    ///
    /// # Returns
    /// New Vec<f32> with RoPE applied, same layout as input.
    pub fn apply(
        &self,
        x_data: &[f32],
        positions: &[f32],
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        assert_eq!(
            x_data.len(),
            num_heads * seq_len * head_dim,
            "RoPE2D::apply: x_data length mismatch"
        );
        assert_eq!(
            positions.len(),
            seq_len * 2,
            "RoPE2D::apply: positions length mismatch"
        );
        assert!(
            head_dim % 2 == 0,
            "RoPE2D::apply: head_dim must be even"
        );

        let half_dim = head_dim / 2;
        // Each half must also be even so we can process pairs (2i, 2i+1).
        assert!(
            half_dim % 2 == 0,
            "RoPE2D::apply: half_dim must be even for pair-wise rotation"
        );

        let num_pairs = half_dim / 2; // number of (cos, sin) rotation pairs per half

        // Precompute inverse frequencies for each pair index:
        //   inv_freq[i] = 1.0 / (freq ^ (2*i / half_dim))
        let inv_freq: Vec<f32> = (0..num_pairs)
            .map(|i| 1.0 / self.freq.powf(2.0 * i as f32 / half_dim as f32))
            .collect();

        // Precompute cos/sin tables: [seq_len, num_pairs] for y and x
        // cos_y[t][i] = cos(pos_y[t] * inv_freq[i])
        let mut cos_y = vec![0.0f32; seq_len * num_pairs];
        let mut sin_y = vec![0.0f32; seq_len * num_pairs];
        let mut cos_x = vec![0.0f32; seq_len * num_pairs];
        let mut sin_x = vec![0.0f32; seq_len * num_pairs];

        for t in 0..seq_len {
            let pos_y = positions[t * 2];
            let pos_x = positions[t * 2 + 1];
            for i in 0..num_pairs {
                let theta_y = pos_y * inv_freq[i];
                let theta_x = pos_x * inv_freq[i];
                cos_y[t * num_pairs + i] = theta_y.cos();
                sin_y[t * num_pairs + i] = theta_y.sin();
                cos_x[t * num_pairs + i] = theta_x.cos();
                sin_x[t * num_pairs + i] = theta_x.sin();
            }
        }

        let mut out = vec![0.0f32; x_data.len()];

        for h in 0..num_heads {
            for t in 0..seq_len {
                let base = h * seq_len * head_dim + t * head_dim;

                // First half of head_dim: apply RoPE with y-positions
                for i in 0..num_pairs {
                    let idx0 = base + 2 * i;
                    let idx1 = base + 2 * i + 1;
                    let c = cos_y[t * num_pairs + i];
                    let s = sin_y[t * num_pairs + i];
                    let x0 = x_data[idx0];
                    let x1 = x_data[idx1];
                    out[idx0] = x0 * c - x1 * s;
                    out[idx1] = x0 * s + x1 * c;
                }

                // Second half of head_dim: apply RoPE with x-positions
                for i in 0..num_pairs {
                    let idx0 = base + half_dim + 2 * i;
                    let idx1 = base + half_dim + 2 * i + 1;
                    let c = cos_x[t * num_pairs + i];
                    let s = sin_x[t * num_pairs + i];
                    let x0 = x_data[idx0];
                    let x1 = x_data[idx1];
                    out[idx0] = x0 * c - x1 * s;
                    out[idx1] = x0 * s + x1 * c;
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope2d_zero_positions() {
        // With all-zero positions, cos(0)=1, sin(0)=0, so output should equal input.
        let rope = RoPE2D::new(100.0);
        let num_heads = 2;
        let seq_len = 3;
        let head_dim = 8;
        let x: Vec<f32> = (0..num_heads * seq_len * head_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let positions = vec![0.0f32; seq_len * 2];

        let out = rope.apply(&x, &positions, num_heads, seq_len, head_dim);

        for (a, b) in x.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Zero positions should leave data unchanged: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_rope2d_preserves_norm() {
        // RoPE is an orthogonal rotation, so it preserves vector norms per pair.
        let rope = RoPE2D::new(100.0);
        let num_heads = 1;
        let seq_len = 2;
        let head_dim = 8;
        let x: Vec<f32> = (0..num_heads * seq_len * head_dim)
            .map(|i| ((i + 1) as f32) * 0.3)
            .collect();
        let positions = vec![1.0, 2.0, 3.0, 4.0]; // [seq_len, 2]

        let out = rope.apply(&x, &positions, num_heads, seq_len, head_dim);

        // Check that each (2i, 2i+1) pair preserves its L2 norm.
        for t in 0..seq_len {
            for pair in 0..(head_dim / 2) {
                let idx0 = t * head_dim + 2 * pair;
                let idx1 = idx0 + 1;
                let norm_in = (x[idx0] * x[idx0] + x[idx1] * x[idx1]).sqrt();
                let norm_out = (out[idx0] * out[idx0] + out[idx1] * out[idx1]).sqrt();
                assert!(
                    (norm_in - norm_out).abs() < 1e-5,
                    "RoPE should preserve pair norm: {} vs {}",
                    norm_in,
                    norm_out
                );
            }
        }
    }

    #[test]
    fn test_rope2d_correct_shape() {
        let rope = RoPE2D::new(100.0);
        let num_heads = 4;
        let seq_len = 10;
        let head_dim = 16;
        let x = vec![1.0f32; num_heads * seq_len * head_dim];
        let positions = vec![0.5f32; seq_len * 2];

        let out = rope.apply(&x, &positions, num_heads, seq_len, head_dim);
        assert_eq!(out.len(), x.len());
    }
}
