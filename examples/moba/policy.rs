/// Recurrent actor-critic policy with LSTM for MOBA.
/// ~45K parameters: obs->128 linear, 128->128 LSTM, actor/critic heads.

use crate::game::{NUM_ACTIONS, OBS_DIM};
use peregrine::nn::{Linear, LSTM};
use peregrine::tensor::Tensor;

pub struct RecurrentActorCritic {
    pub obs_fc: Linear,      // OBS_DIM -> 128
    pub lstm: LSTM,           // 128 -> 128
    pub actor_fc: Linear,     // 128 -> 64
    pub actor_head: Linear,   // 64 -> NUM_ACTIONS
    pub critic_fc: Linear,    // 128 -> 64
    pub critic_head: Linear,  // 64 -> 1
}

impl RecurrentActorCritic {
    pub fn new() -> Self {
        RecurrentActorCritic {
            obs_fc: Linear::new(OBS_DIM, 128),
            lstm: LSTM::new(128, 128),
            actor_fc: Linear::new(128, 64),
            actor_head: Linear::new(64, NUM_ACTIONS),
            critic_fc: Linear::new(128, 64),
            critic_head: Linear::new(64, 1),
        }
    }

    /// Forward pass for a single timestep.
    /// obs: [1, OBS_DIM], h: [1, 128], c: [1, 128]
    /// Returns (logits [1, NUM_ACTIONS], value [1, 1], h_new, c_new)
    pub fn forward_step(
        &self,
        obs: &Tensor,
        h: &Tensor,
        c: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let x = self.obs_fc.forward(obs).relu(); // [1, 128]
        let (_, h_new, c_new) = self.lstm.forward(&x, h, c); // seq_len=1
        let logits = self.actor_head.forward(&self.actor_fc.forward(&h_new).relu());
        let value = self.critic_head.forward(&self.critic_fc.forward(&h_new).relu());
        (logits, value, h_new, c_new)
    }

    /// Forward pass for a sequence (used during BPTT training).
    /// obs_seq: [seq_len, OBS_DIM], h0: [1, 128], c0: [1, 128]
    /// Returns (logits [seq_len, NUM_ACTIONS], values [seq_len, 1], h_n, c_n)
    pub fn forward_seq(
        &self,
        obs_seq: &Tensor,
        h0: &Tensor,
        c0: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let seq_len = obs_seq.shape()[0];

        // Project observations
        // Process each timestep through obs_fc since it expects [1, OBS_DIM]
        let mut projected = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = obs_seq.index_select(0, &[t]); // [1, OBS_DIM]
            projected.push(self.obs_fc.forward(&x_t).relu()); // [1, 128]
        }
        let x_proj = Tensor::stack(&projected, 0).reshape(vec![seq_len, 128]);

        // LSTM over full sequence
        let (lstm_out, h_n, c_n) = self.lstm.forward(&x_proj, h0, c0);

        // Actor/critic heads for each timestep
        let mut logits_list = Vec::with_capacity(seq_len);
        let mut values_list = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let h_t = lstm_out.index_select(0, &[t]); // [1, 128]
            logits_list.push(self.actor_head.forward(&self.actor_fc.forward(&h_t).relu()));
            values_list.push(self.critic_head.forward(&self.critic_fc.forward(&h_t).relu()));
        }

        let logits = Tensor::stack(&logits_list, 0).reshape(vec![seq_len, NUM_ACTIONS]);
        let values = Tensor::stack(&values_list, 0).reshape(vec![seq_len, 1]);

        (logits, values, h_n, c_n)
    }

    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for t in self.obs_fc.params().into_iter()
            .chain(self.lstm.params())
            .chain(self.actor_fc.params())
            .chain(self.actor_head.params())
            .chain(self.critic_fc.params())
            .chain(self.critic_head.params())
        {
            p.push(t.clone());
        }
        p
    }

    pub fn param_count(&self) -> usize {
        self.params().iter().map(|t| t.shape().iter().product::<usize>()).sum()
    }
}

/// Self-play checkpoint manager.
pub struct SelfPlayManager {
    checkpoints: Vec<Vec<f32>>, // flat parameter snapshots
    max_checkpoints: usize,
}

impl SelfPlayManager {
    pub fn new(max_checkpoints: usize) -> Self {
        SelfPlayManager {
            checkpoints: Vec::new(),
            max_checkpoints,
        }
    }

    /// Save current policy parameters as a checkpoint.
    pub fn save_checkpoint(&mut self, params: &[Tensor]) {
        let flat: Vec<f32> = params.iter()
            .flat_map(|t| t.data())
            .collect();
        self.checkpoints.push(flat);
        if self.checkpoints.len() > self.max_checkpoints {
            self.checkpoints.remove(0);
        }
    }

    /// Load a random checkpoint into the opponent network.
    pub fn load_random_checkpoint(&self, params: &[Tensor], rng: &mut u64) -> bool {
        if self.checkpoints.is_empty() {
            return false;
        }
        let idx = (*rng as usize) % self.checkpoints.len();
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;

        let flat = &self.checkpoints[idx];
        let mut offset = 0;
        for p in params {
            let size: usize = p.shape().iter().product();
            let slice = &flat[offset..offset + size];
            let src = Tensor::new(slice.to_vec(), p.shape().clone(), false);
            copy_tensor_data(&src, p);
            offset += size;
        }
        true
    }

    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }
}

/// Copy data from src tensor into dst tensor (same shape).
fn copy_tensor_data(src: &Tensor, dst: &Tensor) {
    let src_data = src.data();
    let dst_data = dst.data();
    assert_eq!(src_data.len(), dst_data.len());
    // Use the copy_params pattern from rl module
    let new_t = Tensor::new(src_data, dst.shape().clone(), true);
    peregrine::rl::copy_params(&[new_t], &[dst.clone()]);
}
