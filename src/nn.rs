use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// MultiHeadAttention
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

        let q_proj = q.matmul(&self.wq).add_bias(&self.bq);
        let k_proj = k.matmul(&self.wk).add_bias(&self.bk);
        let v_proj = v.matmul(&self.wv).add_bias(&self.bv);

        let q_4d = q_proj.reshape(vec![batch, seq_q, self.num_heads, self.head_dim]);
        let q_4d = q_4d.transpose(1, 2);
        let q_2d = q_4d.reshape(vec![batch * self.num_heads * seq_q, self.head_dim]);

        let k_4d = k_proj.reshape(vec![batch, seq_kv, self.num_heads, self.head_dim]);
        let k_4d = k_4d.transpose(1, 2);
        let k_2d = k_4d.reshape(vec![batch * self.num_heads * seq_kv, self.head_dim]);

        let v_4d = v_proj.reshape(vec![batch, seq_kv, self.num_heads, self.head_dim]);
        let v_4d = v_4d.transpose(1, 2);
        let v_2d = v_4d.reshape(vec![batch * self.num_heads * seq_kv, self.head_dim]);

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_out_data: Vec<f32> = Vec::with_capacity(batch * self.num_heads * seq_q * self.head_dim);
        let q_data = q_2d.data();
        let k_data = k_2d.data();
        let v_data = v_2d.data();

        for bh in 0..(batch * self.num_heads) {
            let q_offset = bh * seq_q * self.head_dim;
            let k_offset = bh * seq_kv * self.head_dim;
            let v_offset = bh * seq_kv * self.head_dim;

            let q_slice = q_data[q_offset..q_offset + seq_q * self.head_dim].to_vec();
            let k_slice = k_data[k_offset..k_offset + seq_kv * self.head_dim].to_vec();
            let v_slice = v_data[v_offset..v_offset + seq_kv * self.head_dim].to_vec();

            let q_t = Tensor::new(q_slice, vec![seq_q, self.head_dim], false);
            let k_t = Tensor::new(k_slice, vec![seq_kv, self.head_dim], false);
            let v_t = Tensor::new(v_slice, vec![seq_kv, self.head_dim], false);

            let k_transposed = k_t.transpose(0, 1);
            let scores = q_t.matmul(&k_transposed).scale(scale);
            let attn_weights = scores.softmax(-1);

            let context = attn_weights.matmul(&v_t);
            attn_out_data.extend(context.data());
        }

        let attn_out = Tensor::new(
            attn_out_data,
            vec![batch, self.num_heads, seq_q, self.head_dim],
            false,
        );
        let attn_out = attn_out.transpose(1, 2);
        let attn_flat = attn_out.reshape(vec![batch * seq_q, embed_dim]);

        attn_flat.matmul(&self.wo).add_bias(&self.bo)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![
            &self.wq, &self.wk, &self.wv, &self.wo,
            &self.bq, &self.bk, &self.bv, &self.bo,
        ]
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.wq", prefix), &self.wq),
            (format!("{}.wk", prefix), &self.wk),
            (format!("{}.wv", prefix), &self.wv),
            (format!("{}.wo", prefix), &self.wo),
            (format!("{}.bq", prefix), &self.bq),
            (format!("{}.bk", prefix), &self.bk),
            (format!("{}.bv", prefix), &self.bv),
            (format!("{}.bo", prefix), &self.bo),
        ]
    }
}

// ---------------------------------------------------------------------------
// TransformerEncoderLayer
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
        let normed = x.layer_norm(&self.ln1_gamma, &self.ln1_beta, self.embed_dim);
        let attn_out = self.mha.forward(&normed, &normed, &normed, batch, seq, seq);
        let x = x.add(&attn_out);

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

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        let mut p = self.mha.named_params(&format!("{}.mha", prefix));
        p.extend([
            (format!("{}.ln1_gamma", prefix), &self.ln1_gamma),
            (format!("{}.ln1_beta", prefix), &self.ln1_beta),
            (format!("{}.ln2_gamma", prefix), &self.ln2_gamma),
            (format!("{}.ln2_beta", prefix), &self.ln2_beta),
            (format!("{}.ffn_w1", prefix), &self.ffn_w1),
            (format!("{}.ffn_b1", prefix), &self.ffn_b1),
            (format!("{}.ffn_w2", prefix), &self.ffn_w2),
            (format!("{}.ffn_b2", prefix), &self.ffn_b2),
        ]);
        p
    }
}

// ---------------------------------------------------------------------------
// TransformerDecoderLayer
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
        let normed = tgt.layer_norm(&self.ln1_gamma, &self.ln1_beta, self.embed_dim);
        let sa_out = self.self_attn.forward(&normed, &normed, &normed, batch, num_queries, num_queries);
        let x = tgt.add(&sa_out);

        let normed = x.layer_norm(&self.ln2_gamma, &self.ln2_beta, self.embed_dim);
        let ca_out = self.cross_attn.forward(&normed, memory, memory, batch, num_queries, seq_kv);
        let x = x.add(&ca_out);

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

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        let mut p = self.self_attn.named_params(&format!("{}.self_attn", prefix));
        p.extend(self.cross_attn.named_params(&format!("{}.cross_attn", prefix)));
        p.extend([
            (format!("{}.ln1_gamma", prefix), &self.ln1_gamma),
            (format!("{}.ln1_beta", prefix), &self.ln1_beta),
            (format!("{}.ln2_gamma", prefix), &self.ln2_gamma),
            (format!("{}.ln2_beta", prefix), &self.ln2_beta),
            (format!("{}.ln3_gamma", prefix), &self.ln3_gamma),
            (format!("{}.ln3_beta", prefix), &self.ln3_beta),
            (format!("{}.ffn_w1", prefix), &self.ffn_w1),
            (format!("{}.ffn_b1", prefix), &self.ffn_b1),
            (format!("{}.ffn_w2", prefix), &self.ffn_w2),
            (format!("{}.ffn_b2", prefix), &self.ffn_b2),
        ]);
        p
    }
}
