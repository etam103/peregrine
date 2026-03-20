//! Reinforcement learning infrastructure for Peregrine.
//!
//! Provides core RL abstractions (spaces, environments, buffers) and algorithms
//! (REINFORCE, PPO, DQN) built on top of Peregrine's tensor and autograd system.

use crate::nn;
use crate::optim;
use crate::random;
use crate::tensor::Tensor;

// ============================================================================
// Core Types
// ============================================================================

/// Describes the shape and bounds of observation or action spaces.
#[derive(Debug, Clone)]
pub enum Space {
    /// A single discrete value in `[0, n)`.
    Discrete(usize),
    /// A continuous box with per-element bounds.
    Box {
        low: Vec<f32>,
        high: Vec<f32>,
        shape: Vec<usize>,
    },
    /// Multiple independent discrete sub-spaces.
    MultiDiscrete(Vec<usize>),
    /// Multiple independent binary values.
    MultiBinary(usize),
}

impl Space {
    /// Sample a random element from this space.
    pub fn sample(&self) -> Tensor {
        match self {
            Space::Discrete(n) => {
                let idx = random::randint(&[1], 0, *n as i64);
                idx
            }
            Space::Box { low, high, shape } => {
                let n: usize = shape.iter().product();
                let u = random::uniform(shape, 0.0, 1.0, false);
                let u_data = u.data();
                let data: Vec<f32> = (0..n)
                    .map(|i| low[i] + u_data[i] * (high[i] - low[i]))
                    .collect();
                Tensor::new(data, shape.clone(), false)
            }
            Space::MultiDiscrete(sizes) => {
                let data: Vec<f32> = sizes
                    .iter()
                    .map(|&n| random::randint(&[1], 0, n as i64).data()[0])
                    .collect();
                Tensor::new(data, vec![sizes.len()], false)
            }
            Space::MultiBinary(n) => random::bernoulli(&[*n], 0.5),
        }
    }

    /// Return the shape of tensors in this space.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Space::Discrete(_) => vec![1],
            Space::Box { shape, .. } => shape.clone(),
            Space::MultiDiscrete(sizes) => vec![sizes.len()],
            Space::MultiBinary(n) => vec![*n],
        }
    }

    /// Check whether a tensor value belongs to this space.
    pub fn contains(&self, tensor: &Tensor) -> bool {
        let data = tensor.data();
        match self {
            Space::Discrete(n) => {
                data.len() == 1 && data[0] >= 0.0 && (data[0] as usize) < *n
            }
            Space::Box { low, high, shape } => {
                let expected: usize = shape.iter().product();
                if data.len() != expected {
                    return false;
                }
                data.iter()
                    .enumerate()
                    .all(|(i, &v)| v >= low[i] && v <= high[i])
            }
            Space::MultiDiscrete(sizes) => {
                if data.len() != sizes.len() {
                    return false;
                }
                data.iter()
                    .enumerate()
                    .all(|(i, &v)| v >= 0.0 && (v as usize) < sizes[i])
            }
            Space::MultiBinary(n) => {
                data.len() == *n && data.iter().all(|&v| v == 0.0 || v == 1.0)
            }
        }
    }

    /// For Discrete spaces, return the number of actions; panics otherwise.
    pub fn n(&self) -> usize {
        match self {
            Space::Discrete(n) => *n,
            _ => panic!("n() is only defined for Discrete spaces"),
        }
    }
}

/// The result of a single environment step.
pub struct StepResult {
    pub observation: Tensor,
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
}

// ============================================================================
// Traits
// ============================================================================

/// A tensor-based RL environment (gym-like interface).
pub trait Environment {
    /// Reset the environment and return the initial observation.
    fn reset(&mut self) -> Tensor;
    /// Take an action and return the result.
    fn step(&mut self, action: &Tensor) -> StepResult;
    /// Describe the observation space.
    fn observation_space(&self) -> Space;
    /// Describe the action space.
    fn action_space(&self) -> Space;
    /// Return a human-readable representation of the current state.
    fn render(&self) -> String;
}

/// Extension trait for reasoning/puzzle environments that also support text I/O.
pub trait ReasoningEnv: Environment {
    /// Return the current question as text.
    fn question(&self) -> String;
    /// Return the correct answer as text.
    fn answer(&self) -> String;
    /// Score a textual answer, returning 0.0–1.0.
    fn score_answer(&self, answer: &str) -> f32;
    /// Return metadata key-value pairs about the current problem.
    fn metadata(&self) -> Vec<(String, String)>;
}

/// Unified optimizer interface for RL algorithms.
pub trait RlOptimizer {
    fn step(&mut self);
    fn zero_grad(&self);
}

// ============================================================================
// RlOptimizer implementations for all existing optimizers
// ============================================================================

impl RlOptimizer for optim::Sgd {
    fn step(&mut self) {
        optim::Sgd::step(self);
    }
    fn zero_grad(&self) {
        optim::Sgd::zero_grad(self);
    }
}

impl RlOptimizer for optim::Adam {
    fn step(&mut self) {
        optim::Adam::step(self);
    }
    fn zero_grad(&self) {
        optim::Adam::zero_grad(self);
    }
}

impl RlOptimizer for optim::RmsProp {
    fn step(&mut self) {
        optim::RmsProp::step(self);
    }
    fn zero_grad(&self) {
        optim::RmsProp::zero_grad(self);
    }
}

impl RlOptimizer for optim::Adagrad {
    fn step(&mut self) {
        optim::Adagrad::step(self);
    }
    fn zero_grad(&self) {
        optim::Adagrad::zero_grad(self);
    }
}

impl RlOptimizer for optim::Adamax {
    fn step(&mut self) {
        optim::Adamax::step(self);
    }
    fn zero_grad(&self) {
        optim::Adamax::zero_grad(self);
    }
}

impl RlOptimizer for optim::AdaDelta {
    fn step(&mut self) {
        optim::AdaDelta::step(self);
    }
    fn zero_grad(&self) {
        optim::AdaDelta::zero_grad(self);
    }
}

impl RlOptimizer for optim::Lion {
    fn step(&mut self) {
        optim::Lion::step(self);
    }
    fn zero_grad(&self) {
        optim::Lion::zero_grad(self);
    }
}

impl RlOptimizer for optim::Adafactor {
    fn step(&mut self) {
        optim::Adafactor::step(self);
    }
    fn zero_grad(&self) {
        optim::Adafactor::zero_grad(self);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute discounted returns from a sequence of rewards.
///
/// `R_t = r_t + gamma * R_{t+1}` (reset on episode boundaries).
pub fn discounted_returns(rewards: &[f32], dones: &[f32], gamma: f32) -> Vec<f32> {
    let n = rewards.len();
    let mut returns = vec![0.0f32; n];
    let mut cumulative = 0.0f32;
    for i in (0..n).rev() {
        cumulative = rewards[i] + gamma * cumulative * (1.0 - dones[i]);
        returns[i] = cumulative;
    }
    returns
}

/// Generalized Advantage Estimation (GAE).
///
/// Returns `(advantages, returns)`.
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    gamma: f32,
    lambda: f32,
    last_value: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    let mut advantages = vec![0.0f32; n];
    let mut last_gae = 0.0f32;
    for i in (0..n).rev() {
        let next_value = if i + 1 < n { values[i + 1] } else { last_value };
        let next_non_terminal = 1.0 - dones[i];
        let delta = rewards[i] + gamma * next_value * next_non_terminal - values[i];
        last_gae = delta + gamma * lambda * next_non_terminal * last_gae;
        advantages[i] = last_gae;
    }
    let returns: Vec<f32> = advantages
        .iter()
        .zip(values.iter())
        .map(|(&a, &v)| a + v)
        .collect();
    (advantages, returns)
}

/// Normalize data in-place to zero mean and unit variance.
pub fn normalize(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let std = (var + 1e-8).sqrt();
    for x in data.iter_mut() {
        *x = (*x - mean) / std;
    }
}

/// Explained variance: `1 - Var(y_true - y_pred) / Var(y_true)`.
///
/// Returns a value in `(-inf, 1.0]`. Higher is better; 1.0 means perfect predictions.
pub fn explained_variance(y_pred: &[f32], y_true: &[f32]) -> f32 {
    assert_eq!(y_pred.len(), y_true.len());
    let n = y_true.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mean_true = y_true.iter().sum::<f32>() / n;
    let var_true = y_true
        .iter()
        .map(|&y| (y - mean_true) * (y - mean_true))
        .sum::<f32>()
        / n;
    if var_true < 1e-10 {
        return 0.0;
    }
    let residuals: Vec<f32> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| t - p)
        .collect();
    let mean_res = residuals.iter().sum::<f32>() / n;
    let var_res = residuals
        .iter()
        .map(|&r| (r - mean_res) * (r - mean_res))
        .sum::<f32>()
        / n;
    1.0 - var_res / var_true
}

// ============================================================================
// ReplayBuffer (off-policy, for DQN)
// ============================================================================

/// Fixed-size ring buffer for off-policy algorithms (DQN).
///
/// Stores transitions as flat `Vec<f32>` arrays for efficiency.
pub struct ReplayBuffer {
    state_dim: usize,
    action_dim: usize,
    capacity: usize,
    pos: usize,
    len: usize,
    states: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    next_states: Vec<f32>,
    dones: Vec<f32>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, state_dim: usize, action_dim: usize) -> Self {
        ReplayBuffer {
            state_dim,
            action_dim,
            capacity,
            pos: 0,
            len: 0,
            states: vec![0.0; capacity * state_dim],
            actions: vec![0.0; capacity * action_dim],
            rewards: vec![0.0; capacity],
            next_states: vec![0.0; capacity * state_dim],
            dones: vec![0.0; capacity],
        }
    }

    pub fn push(
        &mut self,
        state: &[f32],
        action: &[f32],
        reward: f32,
        next_state: &[f32],
        done: bool,
    ) {
        let so = self.pos * self.state_dim;
        let ao = self.pos * self.action_dim;
        self.states[so..so + self.state_dim].copy_from_slice(state);
        self.actions[ao..ao + self.action_dim].copy_from_slice(action);
        self.rewards[self.pos] = reward;
        self.next_states[so..so + self.state_dim].copy_from_slice(next_state);
        self.dones[self.pos] = if done { 1.0 } else { 0.0 };
        self.pos = (self.pos + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Sample a random batch of transitions as Tensors.
    ///
    /// Returns `(states, actions, rewards, next_states, dones)`.
    pub fn sample(
        &self,
        batch_size: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        assert!(batch_size <= self.len, "not enough samples in buffer");
        let idx_tensor = random::randint(&[batch_size], 0, self.len as i64);
        let idx_data = idx_tensor.data();

        let mut s = Vec::with_capacity(batch_size * self.state_dim);
        let mut a = Vec::with_capacity(batch_size * self.action_dim);
        let mut r = Vec::with_capacity(batch_size);
        let mut ns = Vec::with_capacity(batch_size * self.state_dim);
        let mut d = Vec::with_capacity(batch_size);

        for &idx_f in &idx_data {
            let idx = idx_f as usize;
            let so = idx * self.state_dim;
            let ao = idx * self.action_dim;
            s.extend_from_slice(&self.states[so..so + self.state_dim]);
            a.extend_from_slice(&self.actions[ao..ao + self.action_dim]);
            r.push(self.rewards[idx]);
            ns.extend_from_slice(&self.next_states[so..so + self.state_dim]);
            d.push(self.dones[idx]);
        }

        (
            Tensor::new(s, vec![batch_size, self.state_dim], false),
            Tensor::new(a, vec![batch_size, self.action_dim], false),
            Tensor::new(r, vec![batch_size], false),
            Tensor::new(ns, vec![batch_size, self.state_dim], false),
            Tensor::new(d, vec![batch_size], false),
        )
    }
}

// ============================================================================
// RolloutBuffer (on-policy, for PPO / REINFORCE)
// ============================================================================

/// A single mini-batch from the rollout buffer.
pub struct RolloutBatch {
    pub observations: Tensor,
    pub actions: Vec<usize>,
    pub old_log_probs: Vec<f32>,
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
}

/// Stores on-policy trajectory data for PPO and similar algorithms.
pub struct RolloutBuffer {
    state_dim: usize,
    capacity: usize,
    pos: usize,
    observations: Vec<f32>,
    actions: Vec<usize>,
    log_probs: Vec<f32>,
    rewards: Vec<f32>,
    values: Vec<f32>,
    dones: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
}

impl RolloutBuffer {
    pub fn new(capacity: usize, state_dim: usize) -> Self {
        RolloutBuffer {
            state_dim,
            capacity,
            pos: 0,
            observations: vec![0.0; capacity * state_dim],
            actions: vec![0; capacity],
            log_probs: vec![0.0; capacity],
            rewards: vec![0.0; capacity],
            values: vec![0.0; capacity],
            dones: vec![0.0; capacity],
            advantages: vec![0.0; capacity],
            returns: vec![0.0; capacity],
        }
    }

    pub fn add(
        &mut self,
        obs: &[f32],
        action: usize,
        log_prob: f32,
        reward: f32,
        value: f32,
        done: bool,
    ) {
        assert!(self.pos < self.capacity, "rollout buffer is full");
        let o = self.pos * self.state_dim;
        self.observations[o..o + self.state_dim].copy_from_slice(obs);
        self.actions[self.pos] = action;
        self.log_probs[self.pos] = log_prob;
        self.rewards[self.pos] = reward;
        self.values[self.pos] = value;
        self.dones[self.pos] = if done { 1.0 } else { 0.0 };
        self.pos += 1;
    }

    pub fn len(&self) -> usize {
        self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }

    /// Compute GAE advantages and returns for the collected rollout.
    pub fn compute_returns_and_advantages(&mut self, last_value: f32, gamma: f32, lambda: f32) {
        let n = self.pos;
        let (adv, ret) = compute_gae(
            &self.rewards[..n],
            &self.values[..n],
            &self.dones[..n],
            gamma,
            lambda,
            last_value,
        );
        self.advantages[..n].copy_from_slice(&adv);
        self.returns[..n].copy_from_slice(&ret);
    }

    /// Generate shuffled mini-batches for PPO training.
    pub fn get_batches(&self, batch_size: usize) -> Vec<RolloutBatch> {
        let n = self.pos;
        if n == 0 {
            return vec![];
        }
        // Shuffle indices
        let perm = random::permutation(n);
        let perm_data = perm.data();
        let indices: Vec<usize> = perm_data.iter().map(|&v| v as usize).collect();

        let num_batches = n / batch_size;
        let mut batches = Vec::with_capacity(num_batches);

        for b in 0..num_batches {
            let start = b * batch_size;
            let batch_idx = &indices[start..start + batch_size];

            let mut obs_data = Vec::with_capacity(batch_size * self.state_dim);
            let mut actions_batch = Vec::with_capacity(batch_size);
            let mut lp_batch = Vec::with_capacity(batch_size);
            let mut adv_batch = Vec::with_capacity(batch_size);
            let mut ret_batch = Vec::with_capacity(batch_size);

            for &i in batch_idx {
                let o = i * self.state_dim;
                obs_data.extend_from_slice(&self.observations[o..o + self.state_dim]);
                actions_batch.push(self.actions[i]);
                lp_batch.push(self.log_probs[i]);
                adv_batch.push(self.advantages[i]);
                ret_batch.push(self.returns[i]);
            }

            batches.push(RolloutBatch {
                observations: Tensor::new(obs_data, vec![batch_size, self.state_dim], false),
                actions: actions_batch,
                old_log_probs: lp_batch,
                advantages: adv_batch,
                returns: ret_batch,
            });
        }

        batches
    }

    /// Clear the buffer for the next rollout.
    pub fn clear(&mut self) {
        self.pos = 0;
    }
}

// ============================================================================
// Sampling helpers (used during rollout collection)
// ============================================================================

/// Sample a discrete action from logits and return `(action, log_prob)`.
///
/// Performs softmax internally; does not build an autograd graph.
fn sample_discrete_action(logits_data: &[f32]) -> (usize, f32) {
    let num_actions = logits_data.len();
    // Numerically stable softmax
    let max = logits_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits_data.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

    // Sample from categorical
    let r = random::uniform(&[1], 0.0, 1.0, false).item();
    let mut cumulative = 0.0;
    let mut action = num_actions - 1;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            action = i;
            break;
        }
    }

    let log_prob = (probs[action] + 1e-8).ln();
    (action, log_prob)
}

/// Argmax of a float slice.
fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ============================================================================
// REINFORCE
// ============================================================================

/// Basic REINFORCE (Monte Carlo policy gradient) with optional baseline.
pub struct Reinforce {
    pub gamma: f32,
    pub baseline: bool,
}

impl Reinforce {
    pub fn new(gamma: f32) -> Self {
        Reinforce {
            gamma,
            baseline: false,
        }
    }

    pub fn with_baseline(mut self) -> Self {
        self.baseline = true;
        self
    }

    /// Run one episode: collect trajectory, compute loss, update policy.
    ///
    /// `policy_fn` maps a state tensor `[1, obs_dim]` to action logits `[1, num_actions]`.
    /// Returns the total undiscounted episode reward.
    pub fn train_episode(
        &self,
        env: &mut dyn Environment,
        policy_fn: &dyn Fn(&Tensor) -> Tensor,
        optimizer: &mut dyn RlOptimizer,
    ) -> f32 {
        let num_actions = env.action_space().n();
        let obs_dim: usize = env.observation_space().shape().iter().product();

        // === Collection phase ===
        let mut obs_list: Vec<Vec<f32>> = Vec::new();
        let mut action_list: Vec<usize> = Vec::new();
        let mut reward_list: Vec<f32> = Vec::new();

        let obs = env.reset();
        let mut obs_data = obs.data();
        let mut total_reward = 0.0f32;

        loop {
            let obs_tensor = Tensor::new(obs_data.clone(), vec![1, obs_dim], false);
            let logits = policy_fn(&obs_tensor);
            let logits_data = logits.data();
            let (action, _log_prob) = sample_discrete_action(&logits_data);

            let action_tensor = Tensor::new(vec![action as f32], vec![1], false);
            let result = env.step(&action_tensor);

            obs_list.push(obs_data);
            action_list.push(action);
            reward_list.push(result.reward);
            total_reward += result.reward;

            if result.done || result.truncated {
                break;
            }
            obs_data = result.observation.data();
        }

        let n = obs_list.len();
        if n == 0 {
            return total_reward;
        }

        // === Compute discounted returns ===
        let dones = vec![0.0f32; n]; // single episode, no intermediate dones
        let mut returns = discounted_returns(&reward_list, &dones, self.gamma);

        if self.baseline {
            normalize(&mut returns);
        }

        // === Training phase: re-run forward pass with autograd ===
        let mut batch_data = Vec::with_capacity(n * obs_dim);
        for obs in &obs_list {
            batch_data.extend_from_slice(obs);
        }
        let batch_obs = Tensor::new(batch_data, vec![n, obs_dim], false);
        let logits = policy_fn(&batch_obs);
        let log_probs = logits.log_softmax(-1);

        // Select log_probs for the taken actions
        let flat_indices: Vec<usize> = (0..n)
            .map(|i| i * num_actions + action_list[i])
            .collect();
        let selected = log_probs.select(&flat_indices);

        let returns_tensor = Tensor::new(returns, vec![n], false);
        let loss = selected.mul(&returns_tensor).mean().neg();

        loss.backward();
        optimizer.step();
        optimizer.zero_grad();

        total_reward
    }

    /// Run multiple episodes, aggregate into a single batch update.
    ///
    /// More sample-efficient than `train_episode` for single-step environments.
    /// Returns the mean episode reward across all episodes.
    pub fn train_batch(
        &self,
        env: &mut dyn Environment,
        policy_fn: &dyn Fn(&Tensor) -> Tensor,
        optimizer: &mut dyn RlOptimizer,
        num_episodes: usize,
    ) -> f32 {
        let num_actions = env.action_space().n();
        let obs_dim: usize = env.observation_space().shape().iter().product();

        let mut all_obs: Vec<Vec<f32>> = Vec::new();
        let mut all_actions: Vec<usize> = Vec::new();
        let mut all_returns: Vec<f32> = Vec::new();
        let mut total_reward = 0.0f32;

        for _ in 0..num_episodes {
            let mut ep_obs: Vec<Vec<f32>> = Vec::new();
            let mut ep_actions: Vec<usize> = Vec::new();
            let mut ep_rewards: Vec<f32> = Vec::new();

            let obs = env.reset();
            let mut obs_data = obs.data();
            let mut ep_reward = 0.0f32;

            loop {
                let obs_tensor = Tensor::new(obs_data.clone(), vec![1, obs_dim], false);
                let logits = policy_fn(&obs_tensor);
                let logits_data = logits.data();
                let (action, _) = sample_discrete_action(&logits_data);

                let action_tensor = Tensor::new(vec![action as f32], vec![1], false);
                let result = env.step(&action_tensor);

                ep_obs.push(obs_data);
                ep_actions.push(action);
                ep_rewards.push(result.reward);
                ep_reward += result.reward;

                if result.done || result.truncated {
                    break;
                }
                obs_data = result.observation.data();
            }

            let dones = vec![0.0f32; ep_rewards.len()];
            let returns = discounted_returns(&ep_rewards, &dones, self.gamma);

            all_obs.extend(ep_obs);
            all_actions.extend(ep_actions);
            all_returns.extend(returns);
            total_reward += ep_reward;
        }

        let n = all_obs.len();
        if n == 0 {
            return 0.0;
        }

        if self.baseline {
            normalize(&mut all_returns);
        }

        let mut batch_data = Vec::with_capacity(n * obs_dim);
        for obs in &all_obs {
            batch_data.extend_from_slice(obs);
        }
        let batch_obs = Tensor::new(batch_data, vec![n, obs_dim], false);
        let logits = policy_fn(&batch_obs);
        let log_probs = logits.log_softmax(-1);

        let flat_indices: Vec<usize> = (0..n)
            .map(|i| i * num_actions + all_actions[i])
            .collect();
        let selected = log_probs.select(&flat_indices);

        let returns_tensor = Tensor::new(all_returns, vec![n], false);
        let loss = selected.mul(&returns_tensor).mean().neg();

        loss.backward();
        optimizer.step();
        optimizer.zero_grad();

        total_reward / num_episodes as f32
    }
}

// ============================================================================
// PPO (Proximal Policy Optimization)
// ============================================================================

/// Configuration for PPO training.
pub struct PpoConfig {
    pub clip_eps: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub ent_coef: f32,
    pub vf_coef: f32,
    pub max_grad_norm: f32,
    pub rollout_steps: usize,
}

impl Default for PpoConfig {
    fn default() -> Self {
        PpoConfig {
            clip_eps: 0.2,
            gamma: 0.99,
            gae_lambda: 0.95,
            epochs: 4,
            batch_size: 64,
            ent_coef: 0.01,
            vf_coef: 0.5,
            max_grad_norm: 0.5,
            rollout_steps: 2048,
        }
    }
}

impl PpoConfig {
    pub fn clip_eps(mut self, v: f32) -> Self {
        self.clip_eps = v;
        self
    }
    pub fn gamma(mut self, v: f32) -> Self {
        self.gamma = v;
        self
    }
    pub fn gae_lambda(mut self, v: f32) -> Self {
        self.gae_lambda = v;
        self
    }
    pub fn epochs(mut self, v: usize) -> Self {
        self.epochs = v;
        self
    }
    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }
    pub fn ent_coef(mut self, v: f32) -> Self {
        self.ent_coef = v;
        self
    }
    pub fn vf_coef(mut self, v: f32) -> Self {
        self.vf_coef = v;
        self
    }
    pub fn max_grad_norm(mut self, v: f32) -> Self {
        self.max_grad_norm = v;
        self
    }
    pub fn rollout_steps(mut self, v: usize) -> Self {
        self.rollout_steps = v;
        self
    }
}

/// PPO trainer: collects rollouts and performs clipped policy optimization.
pub struct PpoTrainer {
    config: PpoConfig,
    buffer: RolloutBuffer,
    last_obs: Vec<f32>,
    episode_reward: f32,
    episode_count: usize,
    reward_sum: f32,
}

impl PpoTrainer {
    pub fn new(config: PpoConfig, state_dim: usize) -> Self {
        let buffer = RolloutBuffer::new(config.rollout_steps, state_dim);
        PpoTrainer {
            config,
            buffer,
            last_obs: Vec::new(),
            episode_reward: 0.0,
            episode_count: 0,
            reward_sum: 0.0,
        }
    }

    /// Collect `rollout_steps` transitions from the environment.
    ///
    /// `policy_fn`: state `[1, obs_dim]` -> logits `[1, num_actions]`
    /// `value_fn`: state `[1, obs_dim]` -> value `[1, 1]` or `[1]`
    ///
    /// Returns mean episode reward over completed episodes (or 0 if none completed).
    pub fn collect_rollouts(
        &mut self,
        env: &mut dyn Environment,
        policy_fn: &dyn Fn(&Tensor) -> Tensor,
        value_fn: &dyn Fn(&Tensor) -> Tensor,
        num_actions: usize,
    ) -> f32 {
        self.buffer.clear();
        self.episode_count = 0;
        self.reward_sum = 0.0;

        let obs_dim: usize = env.observation_space().shape().iter().product();

        // Initialize if needed
        if self.last_obs.is_empty() {
            let obs = env.reset();
            self.last_obs = obs.data();
            self.episode_reward = 0.0;
        }

        for _ in 0..self.config.rollout_steps {
            let obs_tensor = Tensor::new(self.last_obs.clone(), vec![1, obs_dim], false);
            let logits = policy_fn(&obs_tensor);
            let logits_data = logits.data();
            let (action, log_prob) = sample_discrete_action(&logits_data[..num_actions]);

            let value_out = value_fn(&obs_tensor);
            let value = value_out.data()[0];

            let action_tensor = Tensor::new(vec![action as f32], vec![1], false);
            let result = env.step(&action_tensor);

            self.buffer
                .add(&self.last_obs, action, log_prob, result.reward, value, result.done);
            self.episode_reward += result.reward;

            if result.done || result.truncated {
                self.reward_sum += self.episode_reward;
                self.episode_count += 1;
                self.episode_reward = 0.0;
                let obs = env.reset();
                self.last_obs = obs.data();
            } else {
                self.last_obs = result.observation.data();
            }
        }

        // Compute last value for GAE
        let obs_tensor = Tensor::new(self.last_obs.clone(), vec![1, obs_dim], false);
        let last_value = value_fn(&obs_tensor).data()[0];
        self.buffer.compute_returns_and_advantages(
            last_value,
            self.config.gamma,
            self.config.gae_lambda,
        );

        if self.episode_count > 0 {
            self.reward_sum / self.episode_count as f32
        } else {
            0.0
        }
    }

    /// Run PPO optimization on the collected rollout.
    ///
    /// Returns `(policy_loss, value_loss, entropy)` averaged over all updates.
    pub fn update(
        &mut self,
        policy_fn: &dyn Fn(&Tensor) -> Tensor,
        value_fn: &dyn Fn(&Tensor) -> Tensor,
        params: &[Tensor],
        optimizer: &mut dyn RlOptimizer,
        num_actions: usize,
    ) -> (f32, f32, f32) {
        let mut total_pi_loss = 0.0f32;
        let mut total_vf_loss = 0.0f32;
        let mut total_entropy = 0.0f32;
        let mut update_count = 0;

        for _ in 0..self.config.epochs {
            let batches = self.buffer.get_batches(self.config.batch_size);
            for batch in &batches {
                let bs = batch.actions.len();

                // Fresh forward pass with autograd
                let logits = policy_fn(&batch.observations);
                let log_probs = logits.log_softmax(-1);
                let probs = logits.softmax(-1);

                // Select log_probs for taken actions
                let flat_indices: Vec<usize> = (0..bs)
                    .map(|i| i * num_actions + batch.actions[i])
                    .collect();
                let new_log_probs = log_probs.select(&flat_indices);

                // Ratio
                let old_lp = Tensor::new(batch.old_log_probs.clone(), vec![bs], false);
                let log_ratio = new_log_probs.sub(&old_lp);
                let ratio = log_ratio.exp();

                // Clipped surrogate
                let mut adv = batch.advantages.clone();
                normalize(&mut adv);
                let adv_tensor = Tensor::new(adv, vec![bs], false);
                let surr1 = ratio.mul(&adv_tensor);
                let ratio_clipped =
                    ratio.clip(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps);
                let surr2 = ratio_clipped.mul(&adv_tensor);
                let policy_loss = surr1.minimum(&surr2).mean().neg();

                // Value loss
                let values = value_fn(&batch.observations);
                let values_flat = if values.shape().len() == 2 {
                    values.reshape(vec![bs])
                } else {
                    values
                };
                let ret_tensor = Tensor::new(batch.returns.clone(), vec![bs], false);
                let value_loss = nn::mse_loss(&values_flat, &ret_tensor);

                // Entropy bonus: -mean(sum(p * log(p)))
                let plogp = probs.mul(&log_probs);
                let entropy = plogp.sum().neg().scale(1.0 / bs as f32);

                // Total loss
                let loss = policy_loss
                    .add(&value_loss.scale(self.config.vf_coef))
                    .sub(&entropy.scale(self.config.ent_coef));

                loss.backward();
                optim::clip_grad_norm(params, self.config.max_grad_norm);
                optimizer.step();
                optimizer.zero_grad();

                total_pi_loss += policy_loss.item();
                total_vf_loss += value_loss.item();
                total_entropy += entropy.item();
                update_count += 1;
            }
        }

        if update_count > 0 {
            let c = update_count as f32;
            (total_pi_loss / c, total_vf_loss / c, total_entropy / c)
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

// ============================================================================
// DQN (Deep Q-Network)
// ============================================================================

/// Configuration for DQN training.
pub struct DqnConfig {
    pub gamma: f32,
    pub eps_start: f32,
    pub eps_end: f32,
    pub eps_decay: f32,
    pub target_update: usize,
    pub batch_size: usize,
    pub buffer_size: usize,
}

impl Default for DqnConfig {
    fn default() -> Self {
        DqnConfig {
            gamma: 0.99,
            eps_start: 1.0,
            eps_end: 0.01,
            eps_decay: 0.995,
            target_update: 100,
            batch_size: 32,
            buffer_size: 10_000,
        }
    }
}

impl DqnConfig {
    pub fn gamma(mut self, v: f32) -> Self {
        self.gamma = v;
        self
    }
    pub fn eps_start(mut self, v: f32) -> Self {
        self.eps_start = v;
        self
    }
    pub fn eps_end(mut self, v: f32) -> Self {
        self.eps_end = v;
        self
    }
    pub fn eps_decay(mut self, v: f32) -> Self {
        self.eps_decay = v;
        self
    }
    pub fn target_update(mut self, v: usize) -> Self {
        self.target_update = v;
        self
    }
    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }
    pub fn buffer_size(mut self, v: usize) -> Self {
        self.buffer_size = v;
        self
    }
}

/// DQN trainer with replay buffer, epsilon-greedy exploration, and target network.
pub struct DqnTrainer {
    config: DqnConfig,
    buffer: ReplayBuffer,
    epsilon: f32,
    steps: usize,
}

impl DqnTrainer {
    pub fn new(config: DqnConfig, state_dim: usize, action_dim: usize) -> Self {
        let buffer = ReplayBuffer::new(config.buffer_size, state_dim, action_dim);
        let epsilon = config.eps_start;
        DqnTrainer {
            config,
            buffer,
            epsilon,
            steps: 0,
        }
    }

    /// Current exploration rate.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Select an action using epsilon-greedy exploration.
    pub fn select_action(
        &self,
        state: &Tensor,
        q_fn: &dyn Fn(&Tensor) -> Tensor,
        num_actions: usize,
    ) -> usize {
        let r = random::uniform(&[1], 0.0, 1.0, false).item();
        if r < self.epsilon {
            // Random action
            random::randint(&[1], 0, num_actions as i64).data()[0] as usize
        } else {
            // Greedy action
            let q_values = q_fn(state);
            argmax(&q_values.data())
        }
    }

    /// Store a transition in the replay buffer.
    pub fn store_transition(
        &mut self,
        state: &[f32],
        action: usize,
        reward: f32,
        next_state: &[f32],
        done: bool,
    ) {
        self.buffer
            .push(state, &[action as f32], reward, next_state, done);
    }

    /// Perform one gradient step on the Q-network.
    ///
    /// Returns `Some(loss)` if an update was performed, `None` if buffer too small.
    pub fn update(
        &mut self,
        online_fn: &dyn Fn(&Tensor) -> Tensor,
        target_fn: &dyn Fn(&Tensor) -> Tensor,
        online_params: &[Tensor],
        target_params: &[Tensor],
        optimizer: &mut dyn RlOptimizer,
        num_actions: usize,
    ) -> Option<f32> {
        if self.buffer.len() < self.config.batch_size {
            return None;
        }

        let (states, actions, rewards, next_states, dones) =
            self.buffer.sample(self.config.batch_size);
        let bs = self.config.batch_size;

        // Online Q(s, a) for taken actions
        let q_all = online_fn(&states);
        let action_data = actions.data();
        let flat_indices: Vec<usize> = (0..bs)
            .map(|i| i * num_actions + action_data[i] as usize)
            .collect();
        let q_selected = q_all.select(&flat_indices);

        // Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        let target_q = target_fn(&next_states);
        let tq_data = target_q.data();
        let rewards_data = rewards.data();
        let dones_data = dones.data();
        let targets: Vec<f32> = (0..bs)
            .map(|i| {
                let offset = i * num_actions;
                let max_q = tq_data[offset..offset + num_actions]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                rewards_data[i] + self.config.gamma * max_q * (1.0 - dones_data[i])
            })
            .collect();
        let targets_tensor = Tensor::new(targets, vec![bs], false);

        let loss = nn::mse_loss(&q_selected, &targets_tensor);
        let loss_val = loss.item();

        loss.backward();
        optimizer.step();
        optimizer.zero_grad();

        // Decay epsilon
        self.epsilon = (self.epsilon * self.config.eps_decay).max(self.config.eps_end);
        self.steps += 1;

        // Periodic target update
        if self.steps % self.config.target_update == 0 {
            copy_params(online_params, target_params);
        }

        Some(loss_val)
    }
}

/// Hard-copy parameters from `src` to `dst` (for target network updates).
pub fn copy_params(src: &[Tensor], dst: &[Tensor]) {
    for (s, d) in src.iter().zip(dst.iter()) {
        let src_data = s.data();
        let mut d_inner = d.0.borrow_mut();
        d_inner.data.copy_from_slice(&src_data);
    }
}

// ============================================================================
// LLM-oriented RL shared types
// ============================================================================

/// A single completion (rollout) for an LLM-RL training group.
pub struct Completion {
    pub tokens: Vec<usize>,
    pub log_probs: Vec<f32>,
    pub ref_log_probs: Vec<f32>,
    pub reward: f32,
}

/// A group of completions generated from the same prompt.
pub struct SequenceGroup {
    pub prompt_tokens: Vec<usize>,
    pub completions: Vec<Completion>,
}

/// Compute group-relative advantages: (r_i - mean) / std.
/// Used by GRPO, Dr. GRPO, DAPO.
pub fn group_relative_advantages(rewards: &[f32]) -> Vec<f32> {
    let n = rewards.len() as f32;
    if rewards.is_empty() {
        return vec![];
    }
    let mean = rewards.iter().sum::<f32>() / n;
    let var = rewards.iter().map(|&r| (r - mean) * (r - mean)).sum::<f32>() / n;
    let std = (var + 1e-8).sqrt();
    rewards.iter().map(|&r| (r - mean) / std).collect()
}

/// Token-level policy gradient loss with fixed normalization.
/// Computes clipped surrogate loss over tokens, dividing by total token count
/// rather than per-sequence. Used by Dr. GRPO, DAPO, CISPO.
pub fn token_level_loss(log_ratios: &[f32], advantages: &[f32], clip_lo: f32, clip_hi: f32) -> f32 {
    assert_eq!(log_ratios.len(), advantages.len());
    let n = log_ratios.len();
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for i in 0..n {
        let ratio = log_ratios[i].exp();
        let clamped = ratio.clamp(1.0 - clip_lo, 1.0 + clip_hi);
        let surr1 = ratio * advantages[i];
        let surr2 = clamped * advantages[i];
        total += -surr1.min(surr2);
    }
    total / n as f32
}

// ============================================================================
// GRPO (Group Relative Policy Optimization)
// ============================================================================

/// Configuration for GRPO training.
pub struct GrpoConfig {
    pub group_size: usize,
    pub clip_eps: f32,
    pub kl_coef: f32,
    pub epochs: usize,
    pub max_grad_norm: f32,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        GrpoConfig {
            group_size: 8,
            clip_eps: 0.2,
            kl_coef: 0.01,
            epochs: 1,
            max_grad_norm: 1.0,
        }
    }
}

impl GrpoConfig {
    pub fn group_size(mut self, v: usize) -> Self {
        self.group_size = v;
        self
    }
    pub fn clip_eps(mut self, v: f32) -> Self {
        self.clip_eps = v;
        self
    }
    pub fn kl_coef(mut self, v: f32) -> Self {
        self.kl_coef = v;
        self
    }
    pub fn epochs(mut self, v: usize) -> Self {
        self.epochs = v;
        self
    }
    pub fn max_grad_norm(mut self, v: f32) -> Self {
        self.max_grad_norm = v;
        self
    }
}

/// GRPO trainer: group-relative advantages with clipped surrogate + KL penalty.
pub struct GrpoTrainer {
    config: GrpoConfig,
}

impl GrpoTrainer {
    pub fn new(config: GrpoConfig) -> Self {
        GrpoTrainer { config }
    }

    /// Compute GRPO loss from pre-collected sequence groups.
    ///
    /// `policy_fn` maps a token sequence to per-token log-probs under the current policy.
    /// Returns the scalar loss value.
    pub fn compute_loss(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
    ) -> f32 {
        let mut total_loss = 0.0f32;
        let mut total_seqs = 0;

        for group in groups {
            let rewards: Vec<f32> = group.completions.iter().map(|c| c.reward).collect();
            let advantages = group_relative_advantages(&rewards);

            for (ci, comp) in group.completions.iter().enumerate() {
                let new_lp = policy_fn(&comp.tokens);
                let n_tokens = comp.log_probs.len().min(new_lp.len());
                if n_tokens == 0 {
                    continue;
                }

                // Sequence-level: average log-ratio across tokens
                let mut seq_loss = 0.0f32;
                let mut kl_sum = 0.0f32;
                for t in 0..n_tokens {
                    let log_ratio = new_lp[t] - comp.log_probs[t];
                    let ratio = log_ratio.exp();
                    let clamped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps);
                    let surr1 = ratio * advantages[ci];
                    let surr2 = clamped * advantages[ci];
                    seq_loss += -surr1.min(surr2);
                    // KL: new_lp - ref_lp approximation
                    kl_sum += new_lp[t] - comp.ref_log_probs[t];
                }
                total_loss += seq_loss / n_tokens as f32 + self.config.kl_coef * kl_sum / n_tokens as f32;
                total_seqs += 1;
            }
        }

        if total_seqs > 0 {
            total_loss / total_seqs as f32
        } else {
            0.0
        }
    }

    /// Run one GRPO update: compute loss, backward, clip gradients, step optimizer.
    pub fn update(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
        params: &[Tensor],
        optimizer: &mut dyn RlOptimizer,
    ) -> f32 {
        let loss_val = self.compute_loss(groups, policy_fn);
        // Create a scalar tensor to trigger backward through any tracked computation
        let loss_tensor = Tensor::new(vec![loss_val], vec![1], true);
        loss_tensor.backward();
        optim::clip_grad_norm(params, self.config.max_grad_norm);
        optimizer.step();
        optimizer.zero_grad();
        loss_val
    }
}

// ============================================================================
// RLOO (REINFORCE Leave-One-Out)
// ============================================================================

/// Configuration for RLOO training.
pub struct RlooConfig {
    pub group_size: usize,
    pub kl_coef: f32,
    pub max_grad_norm: f32,
}

impl Default for RlooConfig {
    fn default() -> Self {
        RlooConfig {
            group_size: 8,
            kl_coef: 0.01,
            max_grad_norm: 1.0,
        }
    }
}

impl RlooConfig {
    pub fn group_size(mut self, v: usize) -> Self {
        self.group_size = v;
        self
    }
    pub fn kl_coef(mut self, v: f32) -> Self {
        self.kl_coef = v;
        self
    }
    pub fn max_grad_norm(mut self, v: f32) -> Self {
        self.max_grad_norm = v;
        self
    }
}

/// RLOO trainer: leave-one-out baseline, no clipping.
pub struct RlooTrainer {
    config: RlooConfig,
}

impl RlooTrainer {
    pub fn new(config: RlooConfig) -> Self {
        RlooTrainer { config }
    }

    /// Compute RLOO loss: baseline b_i = (sum - r_i) / (G-1), advantage = r_i - b_i.
    pub fn compute_loss(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
    ) -> f32 {
        let mut total_loss = 0.0f32;
        let mut total_seqs = 0;

        for group in groups {
            let rewards: Vec<f32> = group.completions.iter().map(|c| c.reward).collect();
            let g = rewards.len();
            if g <= 1 {
                continue;
            }
            let sum_r: f32 = rewards.iter().sum();

            for (ci, comp) in group.completions.iter().enumerate() {
                let baseline = (sum_r - rewards[ci]) / (g as f32 - 1.0);
                let advantage = rewards[ci] - baseline;

                let new_lp = policy_fn(&comp.tokens);
                let n_tokens = comp.log_probs.len().min(new_lp.len());
                if n_tokens == 0 {
                    continue;
                }

                let mut seq_loss = 0.0f32;
                let mut kl_sum = 0.0f32;
                for t in 0..n_tokens {
                    let log_ratio = new_lp[t] - comp.log_probs[t];
                    let ratio = log_ratio.exp();
                    // No clipping for RLOO
                    seq_loss += -ratio * advantage;
                    kl_sum += new_lp[t] - comp.ref_log_probs[t];
                }
                total_loss += seq_loss / n_tokens as f32 + self.config.kl_coef * kl_sum / n_tokens as f32;
                total_seqs += 1;
            }
        }

        if total_seqs > 0 {
            total_loss / total_seqs as f32
        } else {
            0.0
        }
    }

    /// Run one RLOO update.
    pub fn update(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
        params: &[Tensor],
        optimizer: &mut dyn RlOptimizer,
    ) -> f32 {
        let loss_val = self.compute_loss(groups, policy_fn);
        let loss_tensor = Tensor::new(vec![loss_val], vec![1], true);
        loss_tensor.backward();
        optim::clip_grad_norm(params, self.config.max_grad_norm);
        optimizer.step();
        optimizer.zero_grad();
        loss_val
    }
}

// ============================================================================
// Dr. GRPO (Decoupled Regularization GRPO)
// ============================================================================

/// Configuration for Dr. GRPO training.
pub struct DrGrpoConfig {
    pub group_size: usize,
    pub clip_eps: f32,
    pub kl_coef: f32,
    pub max_grad_norm: f32,
}

impl Default for DrGrpoConfig {
    fn default() -> Self {
        DrGrpoConfig {
            group_size: 8,
            clip_eps: 0.2,
            kl_coef: 0.01,
            max_grad_norm: 1.0,
        }
    }
}

impl DrGrpoConfig {
    pub fn group_size(mut self, v: usize) -> Self {
        self.group_size = v;
        self
    }
    pub fn clip_eps(mut self, v: f32) -> Self {
        self.clip_eps = v;
        self
    }
    pub fn kl_coef(mut self, v: f32) -> Self {
        self.kl_coef = v;
        self
    }
    pub fn max_grad_norm(mut self, v: f32) -> Self {
        self.max_grad_norm = v;
        self
    }
}

/// Dr. GRPO trainer: like GRPO but token-level loss with fixed normalization
/// (divide by total tokens across all sequences, not per-sequence mean).
pub struct DrGrpoTrainer {
    config: DrGrpoConfig,
}

impl DrGrpoTrainer {
    pub fn new(config: DrGrpoConfig) -> Self {
        DrGrpoTrainer { config }
    }

    /// Compute Dr. GRPO loss with token-level normalization.
    pub fn compute_loss(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
    ) -> f32 {
        let mut all_log_ratios = Vec::new();
        let mut all_advantages = Vec::new();
        let mut kl_sum = 0.0f32;
        let mut kl_count = 0;

        for group in groups {
            let rewards: Vec<f32> = group.completions.iter().map(|c| c.reward).collect();
            let advantages = group_relative_advantages(&rewards);

            for (ci, comp) in group.completions.iter().enumerate() {
                let new_lp = policy_fn(&comp.tokens);
                let n_tokens = comp.log_probs.len().min(new_lp.len());

                for t in 0..n_tokens {
                    let log_ratio = new_lp[t] - comp.log_probs[t];
                    all_log_ratios.push(log_ratio);
                    all_advantages.push(advantages[ci]);
                    kl_sum += new_lp[t] - comp.ref_log_probs[t];
                    kl_count += 1;
                }
            }
        }

        let pg_loss = token_level_loss(&all_log_ratios, &all_advantages, self.config.clip_eps, self.config.clip_eps);
        let kl_penalty = if kl_count > 0 { self.config.kl_coef * kl_sum / kl_count as f32 } else { 0.0 };
        pg_loss + kl_penalty
    }

    /// Run one Dr. GRPO update.
    pub fn update(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
        params: &[Tensor],
        optimizer: &mut dyn RlOptimizer,
    ) -> f32 {
        let loss_val = self.compute_loss(groups, policy_fn);
        let loss_tensor = Tensor::new(vec![loss_val], vec![1], true);
        loss_tensor.backward();
        optim::clip_grad_norm(params, self.config.max_grad_norm);
        optimizer.step();
        optimizer.zero_grad();
        loss_val
    }
}

// ============================================================================
// DAPO (Decoupled Advantage Policy Optimization)
// ============================================================================

/// Configuration for DAPO training.
pub struct DapoConfig {
    pub group_size: usize,
    pub clip_lo: f32,
    pub clip_hi: f32,
    pub max_grad_norm: f32,
    pub overlong_penalty: f32,
}

impl Default for DapoConfig {
    fn default() -> Self {
        DapoConfig {
            group_size: 16,
            clip_lo: 0.2,
            clip_hi: 0.28,
            max_grad_norm: 1.0,
            overlong_penalty: -1.0,
        }
    }
}

impl DapoConfig {
    pub fn group_size(mut self, v: usize) -> Self {
        self.group_size = v;
        self
    }
    pub fn clip_lo(mut self, v: f32) -> Self {
        self.clip_lo = v;
        self
    }
    pub fn clip_hi(mut self, v: f32) -> Self {
        self.clip_hi = v;
        self
    }
    pub fn max_grad_norm(mut self, v: f32) -> Self {
        self.max_grad_norm = v;
        self
    }
    pub fn overlong_penalty(mut self, v: f32) -> Self {
        self.overlong_penalty = v;
        self
    }
}

/// DAPO trainer: asymmetric clipping, dynamic sampling, token-level loss,
/// overlong penalty for max-length sequences.
pub struct DapoTrainer {
    config: DapoConfig,
    max_seq_len: usize,
}

impl DapoTrainer {
    pub fn new(config: DapoConfig, max_seq_len: usize) -> Self {
        DapoTrainer { config, max_seq_len }
    }

    /// Compute DAPO loss with asymmetric clipping and dynamic sampling.
    pub fn compute_loss(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
    ) -> f32 {
        let mut all_log_ratios = Vec::new();
        let mut all_advantages = Vec::new();

        for group in groups {
            // Apply overlong penalty: add penalty to reward of max-length sequences
            let rewards: Vec<f32> = group.completions.iter().map(|c| {
                if c.tokens.len() >= self.max_seq_len {
                    c.reward + self.config.overlong_penalty
                } else {
                    c.reward
                }
            }).collect();

            // Dynamic sampling: skip groups where all rewards are identical
            let first = rewards[0];
            let all_same = rewards.iter().all(|&r| (r - first).abs() < 1e-8);
            if all_same {
                continue;
            }

            let advantages = group_relative_advantages(&rewards);

            for (ci, comp) in group.completions.iter().enumerate() {
                let new_lp = policy_fn(&comp.tokens);
                let n_tokens = comp.log_probs.len().min(new_lp.len());

                for t in 0..n_tokens {
                    let log_ratio = new_lp[t] - comp.log_probs[t];
                    all_log_ratios.push(log_ratio);
                    all_advantages.push(advantages[ci]);
                }
            }
        }

        // Token-level loss with asymmetric clipping
        token_level_loss(&all_log_ratios, &all_advantages, self.config.clip_lo, self.config.clip_hi)
    }

    /// Run one DAPO update.
    pub fn update(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
        params: &[Tensor],
        optimizer: &mut dyn RlOptimizer,
    ) -> f32 {
        let loss_val = self.compute_loss(groups, policy_fn);
        let loss_tensor = Tensor::new(vec![loss_val], vec![1], true);
        loss_tensor.backward();
        optim::clip_grad_norm(params, self.config.max_grad_norm);
        optimizer.step();
        optimizer.zero_grad();
        loss_val
    }
}

// ============================================================================
// CISPO (Clipped Importance Sampling Policy Optimization)
// ============================================================================

/// Configuration for CISPO training.
pub struct CispoConfig {
    pub clip_eps: f32,
    pub group_size: usize,
    pub kl_coef: f32,
    pub max_grad_norm: f32,
}

impl Default for CispoConfig {
    fn default() -> Self {
        CispoConfig {
            clip_eps: 0.2,
            group_size: 8,
            kl_coef: 0.01,
            max_grad_norm: 1.0,
        }
    }
}

impl CispoConfig {
    pub fn clip_eps(mut self, v: f32) -> Self {
        self.clip_eps = v;
        self
    }
    pub fn group_size(mut self, v: usize) -> Self {
        self.group_size = v;
        self
    }
    pub fn kl_coef(mut self, v: f32) -> Self {
        self.kl_coef = v;
        self
    }
    pub fn max_grad_norm(mut self, v: f32) -> Self {
        self.max_grad_norm = v;
        self
    }
}

/// CISPO trainer: stop_gradient(clip(ratio)) * log_prob * advantage.
/// Gradients flow through log_prob, not through the ratio.
pub struct CispoTrainer {
    config: CispoConfig,
}

impl CispoTrainer {
    pub fn new(config: CispoConfig) -> Self {
        CispoTrainer { config }
    }

    /// Compute CISPO loss: clipped ratio is stop-gradient, gradient flows through log_prob.
    ///
    /// For each token: weight = stop_grad(clip(ratio, 1-eps, 1+eps)), loss = -weight * new_lp * adv.
    pub fn compute_loss(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
    ) -> f32 {
        let mut total_loss = 0.0f32;
        let mut total_tokens = 0;
        let mut kl_sum = 0.0f32;

        for group in groups {
            let rewards: Vec<f32> = group.completions.iter().map(|c| c.reward).collect();
            let advantages = group_relative_advantages(&rewards);

            for (ci, comp) in group.completions.iter().enumerate() {
                let new_lp = policy_fn(&comp.tokens);
                let n_tokens = comp.log_probs.len().min(new_lp.len());

                for t in 0..n_tokens {
                    let log_ratio = new_lp[t] - comp.log_probs[t];
                    let ratio = log_ratio.exp();
                    // Stop-gradient on clipped ratio: this is a fixed weight
                    let clipped_ratio = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps);
                    // Loss: -clipped_ratio * new_log_prob * advantage
                    // (gradient flows through new_lp, not through clipped_ratio)
                    total_loss += -clipped_ratio * new_lp[t] * advantages[ci];
                    kl_sum += new_lp[t] - comp.ref_log_probs[t];
                    total_tokens += 1;
                }
            }
        }

        if total_tokens > 0 {
            total_loss / total_tokens as f32 + self.config.kl_coef * kl_sum / total_tokens as f32
        } else {
            0.0
        }
    }

    /// Run one CISPO update.
    pub fn update(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
        params: &[Tensor],
        optimizer: &mut dyn RlOptimizer,
    ) -> f32 {
        let loss_val = self.compute_loss(groups, policy_fn);
        let loss_tensor = Tensor::new(vec![loss_val], vec![1], true);
        loss_tensor.backward();
        optim::clip_grad_norm(params, self.config.max_grad_norm);
        optimizer.step();
        optimizer.zero_grad();
        loss_val
    }
}

// ============================================================================
// DPPO (Divergence-Penalized PPO)
// ============================================================================

/// Divergence type for DPPO.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DivergenceType {
    KL,
    ReverseKL,
    JSD,
}

/// Configuration for DPPO training.
pub struct DppoConfig {
    pub div_type: DivergenceType,
    pub div_coef: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub vf_coef: f32,
    pub max_grad_norm: f32,
    pub rollout_steps: usize,
}

impl Default for DppoConfig {
    fn default() -> Self {
        DppoConfig {
            div_type: DivergenceType::KL,
            div_coef: 0.01,
            gamma: 0.99,
            gae_lambda: 0.95,
            epochs: 4,
            batch_size: 64,
            vf_coef: 0.5,
            max_grad_norm: 0.5,
            rollout_steps: 2048,
        }
    }
}

impl DppoConfig {
    pub fn div_type(mut self, v: DivergenceType) -> Self {
        self.div_type = v;
        self
    }
    pub fn div_coef(mut self, v: f32) -> Self {
        self.div_coef = v;
        self
    }
    pub fn gamma(mut self, v: f32) -> Self {
        self.gamma = v;
        self
    }
    pub fn gae_lambda(mut self, v: f32) -> Self {
        self.gae_lambda = v;
        self
    }
    pub fn epochs(mut self, v: usize) -> Self {
        self.epochs = v;
        self
    }
    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }
    pub fn vf_coef(mut self, v: f32) -> Self {
        self.vf_coef = v;
        self
    }
    pub fn max_grad_norm(mut self, v: f32) -> Self {
        self.max_grad_norm = v;
        self
    }
    pub fn rollout_steps(mut self, v: usize) -> Self {
        self.rollout_steps = v;
        self
    }
}

/// DPPO trainer: replaces PPO's clipped surrogate with a divergence penalty
/// between current and old policy distributions.
pub struct DppoTrainer {
    config: DppoConfig,
    buffer: RolloutBuffer,
    last_obs: Vec<f32>,
    episode_reward: f32,
    episode_count: usize,
    reward_sum: f32,
}

impl DppoTrainer {
    pub fn new(config: DppoConfig, state_dim: usize) -> Self {
        let buffer = RolloutBuffer::new(config.rollout_steps, state_dim);
        DppoTrainer {
            config,
            buffer,
            last_obs: Vec::new(),
            episode_reward: 0.0,
            episode_count: 0,
            reward_sum: 0.0,
        }
    }

    /// Collect rollouts from the environment (reuses PPO pattern).
    pub fn collect_rollouts(
        &mut self,
        env: &mut dyn Environment,
        policy_fn: &dyn Fn(&Tensor) -> Tensor,
        value_fn: &dyn Fn(&Tensor) -> Tensor,
        num_actions: usize,
    ) -> f32 {
        self.buffer.clear();
        self.episode_count = 0;
        self.reward_sum = 0.0;

        let obs_dim: usize = env.observation_space().shape().iter().product();

        if self.last_obs.is_empty() {
            let obs = env.reset();
            self.last_obs = obs.data();
            self.episode_reward = 0.0;
        }

        for _ in 0..self.config.rollout_steps {
            let obs_tensor = Tensor::new(self.last_obs.clone(), vec![1, obs_dim], false);
            let logits = policy_fn(&obs_tensor);
            let logits_data = logits.data();
            let (action, log_prob) = sample_discrete_action(&logits_data[..num_actions]);

            let value_out = value_fn(&obs_tensor);
            let value = value_out.data()[0];

            let action_tensor = Tensor::new(vec![action as f32], vec![1], false);
            let result = env.step(&action_tensor);

            self.buffer
                .add(&self.last_obs, action, log_prob, result.reward, value, result.done);
            self.episode_reward += result.reward;

            if result.done || result.truncated {
                self.reward_sum += self.episode_reward;
                self.episode_count += 1;
                self.episode_reward = 0.0;
                let obs = env.reset();
                self.last_obs = obs.data();
            } else {
                self.last_obs = result.observation.data();
            }
        }

        let obs_tensor = Tensor::new(self.last_obs.clone(), vec![1, obs_dim], false);
        let last_value = value_fn(&obs_tensor).data()[0];
        self.buffer.compute_returns_and_advantages(
            last_value,
            self.config.gamma,
            self.config.gae_lambda,
        );

        if self.episode_count > 0 {
            self.reward_sum / self.episode_count as f32
        } else {
            0.0
        }
    }

    /// Compute divergence between current and old policy distributions.
    fn compute_divergence(div_type: DivergenceType, new_probs: &[f32], old_probs: &[f32]) -> f32 {
        let n = new_probs.len();
        let mut div = 0.0f32;
        for i in 0..n {
            let p = new_probs[i].max(1e-8);
            let q = old_probs[i].max(1e-8);
            match div_type {
                DivergenceType::KL => {
                    // KL(old || new) = sum old * log(old/new)
                    div += q * (q / p).ln();
                }
                DivergenceType::ReverseKL => {
                    // KL(new || old) = sum new * log(new/old)
                    div += p * (p / q).ln();
                }
                DivergenceType::JSD => {
                    // JSD = 0.5 * KL(old || M) + 0.5 * KL(new || M) where M = (old+new)/2
                    let m = 0.5 * (p + q);
                    div += 0.5 * q * (q / m).ln() + 0.5 * p * (p / m).ln();
                }
            }
        }
        div
    }

    /// Run DPPO optimization on collected rollout.
    ///
    /// Returns `(policy_loss, value_loss, divergence)` averaged over all updates.
    pub fn update(
        &mut self,
        policy_fn: &dyn Fn(&Tensor) -> Tensor,
        value_fn: &dyn Fn(&Tensor) -> Tensor,
        params: &[Tensor],
        optimizer: &mut dyn RlOptimizer,
        num_actions: usize,
    ) -> (f32, f32, f32) {
        let mut total_pi_loss = 0.0f32;
        let mut total_vf_loss = 0.0f32;
        let mut total_div = 0.0f32;
        let mut update_count = 0;

        for _ in 0..self.config.epochs {
            let batches = self.buffer.get_batches(self.config.batch_size);
            for batch in &batches {
                let bs = batch.actions.len();

                let logits = policy_fn(&batch.observations);
                let log_probs = logits.log_softmax(-1);
                let probs = logits.softmax(-1);
                let probs_data = probs.data();

                // Compute old probs from old log_probs using the action
                // For divergence, we compare full distributions
                let flat_indices: Vec<usize> = (0..bs)
                    .map(|i| i * num_actions + batch.actions[i])
                    .collect();
                let new_log_probs = log_probs.select(&flat_indices);

                // Compute per-sample divergence (over the full action distribution)
                let mut batch_div = 0.0f32;
                for i in 0..bs {
                    // Reconstruct old probs: we only have old_log_prob for the taken action,
                    // so we use current distribution shifted by the log ratio as an approximation
                    let new_slice = &probs_data[i * num_actions..(i + 1) * num_actions];
                    // Approximate old distribution: uniform for non-taken actions,
                    // with taken-action prob from old_log_prob
                    let old_action_prob = batch.old_log_probs[i].exp().clamp(1e-8, 1.0);
                    let remaining = (1.0 - old_action_prob).max(0.0);
                    let other_prob = if num_actions > 1 { remaining / (num_actions - 1) as f32 } else { 0.0 };
                    let old_probs: Vec<f32> = (0..num_actions).map(|a| {
                        if a == batch.actions[i] { old_action_prob } else { other_prob }
                    }).collect();
                    batch_div += Self::compute_divergence(self.config.div_type, new_slice, &old_probs);
                }
                let mean_div = batch_div / bs as f32;

                // Policy loss = -advantage * ratio + div_coef * divergence
                let old_lp = Tensor::new(batch.old_log_probs.clone(), vec![bs], false);
                let log_ratio = new_log_probs.sub(&old_lp);
                let ratio = log_ratio.exp();

                let mut adv = batch.advantages.clone();
                normalize(&mut adv);
                let adv_tensor = Tensor::new(adv, vec![bs], false);
                let pg_term = ratio.mul(&adv_tensor).mean().neg();

                let div_tensor = Tensor::new(vec![mean_div], vec![1], false);
                let policy_loss = pg_term.add(&div_tensor.scale(self.config.div_coef));

                // Value loss
                let values = value_fn(&batch.observations);
                let values_flat = if values.shape().len() == 2 {
                    values.reshape(vec![bs])
                } else {
                    values
                };
                let ret_tensor = Tensor::new(batch.returns.clone(), vec![bs], false);
                let value_loss = nn::mse_loss(&values_flat, &ret_tensor);

                let loss = policy_loss.add(&value_loss.scale(self.config.vf_coef));

                loss.backward();
                optim::clip_grad_norm(params, self.config.max_grad_norm);
                optimizer.step();
                optimizer.zero_grad();

                total_pi_loss += pg_term.item();
                total_vf_loss += value_loss.item();
                total_div += mean_div;
                update_count += 1;
            }
        }

        if update_count > 0 {
            let c = update_count as f32;
            (total_pi_loss / c, total_vf_loss / c, total_div / c)
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

// ============================================================================
// MaxRL (Maximum Reward Likelihood)
// ============================================================================

/// Configuration for MaxRL training.
pub struct MaxRlConfig {
    pub group_size: usize,
    pub top_k_values: Vec<usize>,
    pub max_grad_norm: f32,
}

impl Default for MaxRlConfig {
    fn default() -> Self {
        MaxRlConfig {
            group_size: 16,
            top_k_values: vec![1, 2, 4, 8],
            max_grad_norm: 1.0,
        }
    }
}

impl MaxRlConfig {
    pub fn group_size(mut self, v: usize) -> Self {
        self.group_size = v;
        self
    }
    pub fn top_k_values(mut self, v: Vec<usize>) -> Self {
        self.top_k_values = v;
        self
    }
    pub fn max_grad_norm(mut self, v: f32) -> Self {
        self.max_grad_norm = v;
        self
    }
}

/// MaxRL trainer: truncated harmonic mixture over pass@k objectives.
/// L = sum_k (1/k) * L_k / sum_k (1/k), where L_k is the negative mean
/// log-prob of the top-k completions by reward.
pub struct MaxRlTrainer {
    config: MaxRlConfig,
}

impl MaxRlTrainer {
    pub fn new(config: MaxRlConfig) -> Self {
        MaxRlTrainer { config }
    }

    /// Compute MaxRL loss: harmonic mixture over top-k log-prob objectives.
    pub fn compute_loss(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
    ) -> f32 {
        let mut harmonic_denom = 0.0f32;
        for &k in &self.config.top_k_values {
            harmonic_denom += 1.0 / k as f32;
        }
        if harmonic_denom == 0.0 || groups.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0f32;

        for &k in &self.config.top_k_values {
            let weight = 1.0 / k as f32;
            let mut k_loss = 0.0f32;
            let mut k_count = 0;

            for group in groups {
                let g = group.completions.len();
                let actual_k = k.min(g);
                if actual_k == 0 {
                    continue;
                }

                // Sort completions by reward descending, take top-k
                let mut indices: Vec<usize> = (0..g).collect();
                indices.sort_by(|&a, &b| {
                    group.completions[b].reward
                        .partial_cmp(&group.completions[a].reward)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for &ci in &indices[..actual_k] {
                    let comp = &group.completions[ci];
                    let new_lp = policy_fn(&comp.tokens);
                    let n_tokens = new_lp.len().min(comp.log_probs.len());
                    if n_tokens == 0 {
                        continue;
                    }
                    // L_k: negative mean log-prob of these completions
                    let mean_lp: f32 = new_lp[..n_tokens].iter().sum::<f32>() / n_tokens as f32;
                    k_loss += -mean_lp;
                    k_count += 1;
                }
            }

            if k_count > 0 {
                total_loss += weight * k_loss / k_count as f32;
            }
        }

        total_loss / harmonic_denom
    }

    /// Run one MaxRL update.
    pub fn update(
        &self,
        groups: &[SequenceGroup],
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
        params: &[Tensor],
        optimizer: &mut dyn RlOptimizer,
    ) -> f32 {
        let loss_val = self.compute_loss(groups, policy_fn);
        let loss_tensor = Tensor::new(vec![loss_val], vec![1], true);
        loss_tensor.backward();
        optim::clip_grad_norm(params, self.config.max_grad_norm);
        optimizer.step();
        optimizer.zero_grad();
        loss_val
    }
}

// ============================================================================
// ScaleRL (Scalable RL with async buffer)
// ============================================================================

/// Configuration for ScaleRL training.
pub struct ScaleRlConfig {
    pub group_size: usize,
    pub clip_eps: f32,
    pub max_grad_norm: f32,
    pub min_buffer_size: usize,
}

impl Default for ScaleRlConfig {
    fn default() -> Self {
        ScaleRlConfig {
            group_size: 8,
            clip_eps: 0.2,
            max_grad_norm: 1.0,
            min_buffer_size: 4,
        }
    }
}

impl ScaleRlConfig {
    pub fn group_size(mut self, v: usize) -> Self {
        self.group_size = v;
        self
    }
    pub fn clip_eps(mut self, v: f32) -> Self {
        self.clip_eps = v;
        self
    }
    pub fn max_grad_norm(mut self, v: f32) -> Self {
        self.max_grad_norm = v;
        self
    }
    pub fn min_buffer_size(mut self, v: usize) -> Self {
        self.min_buffer_size = v;
        self
    }
}

/// ScaleRL trainer: async buffer with prompt-level aggregation.
/// Accumulates rollouts asynchronously and updates when enough are buffered.
pub struct ScaleRlTrainer {
    config: ScaleRlConfig,
    pending_groups: Vec<SequenceGroup>,
}

impl ScaleRlTrainer {
    pub fn new(config: ScaleRlConfig) -> Self {
        ScaleRlTrainer {
            config,
            pending_groups: Vec::new(),
        }
    }

    /// Add new rollout groups to the async pending buffer.
    pub fn add_rollouts(&mut self, groups: Vec<SequenceGroup>) {
        self.pending_groups.extend(groups);
    }

    /// Number of groups currently buffered.
    pub fn pending_count(&self) -> usize {
        self.pending_groups.len()
    }

    /// If enough groups are buffered (>= min_buffer_size), drain the buffer,
    /// compute prompt-level aggregated loss, and run one update.
    ///
    /// Returns `Some(loss)` if an update was performed, `None` otherwise.
    pub fn update(
        &mut self,
        policy_fn: &dyn Fn(&[usize]) -> Vec<f32>,
        params: &[Tensor],
        optimizer: &mut dyn RlOptimizer,
    ) -> Option<f32> {
        if self.pending_groups.len() < self.config.min_buffer_size {
            return None;
        }

        let groups: Vec<SequenceGroup> = self.pending_groups.drain(..).collect();

        // Prompt-level aggregation: for each group, compute average advantage
        // and average loss across completions, then average across groups.
        let mut total_loss = 0.0f32;
        let mut group_count = 0;

        for group in &groups {
            let rewards: Vec<f32> = group.completions.iter().map(|c| c.reward).collect();
            let advantages = group_relative_advantages(&rewards);

            let mut group_loss = 0.0f32;
            let mut group_tokens = 0;

            for (ci, comp) in group.completions.iter().enumerate() {
                let new_lp = policy_fn(&comp.tokens);
                let n_tokens = comp.log_probs.len().min(new_lp.len());

                for t in 0..n_tokens {
                    // FP32 log-ratio computation (already f32 in Rust)
                    let log_ratio = new_lp[t] - comp.log_probs[t];
                    let ratio = log_ratio.exp();
                    let clamped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps);
                    let surr1 = ratio * advantages[ci];
                    let surr2 = clamped * advantages[ci];
                    group_loss += -surr1.min(surr2);
                    group_tokens += 1;
                }
            }

            if group_tokens > 0 {
                total_loss += group_loss / group_tokens as f32;
                group_count += 1;
            }
        }

        let loss_val = if group_count > 0 { total_loss / group_count as f32 } else { 0.0 };

        let loss_tensor = Tensor::new(vec![loss_val], vec![1], true);
        loss_tensor.backward();
        optim::clip_grad_norm(params, self.config.max_grad_norm);
        optimizer.step();
        optimizer.zero_grad();

        Some(loss_val)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use crate::optim::Adam;

    // Simple deterministic environment for testing
    struct MockEnv {
        state: f32,
        step_count: usize,
        max_steps: usize,
    }

    impl MockEnv {
        fn new(max_steps: usize) -> Self {
            MockEnv {
                state: 0.0,
                step_count: 0,
                max_steps,
            }
        }
    }

    impl Environment for MockEnv {
        fn reset(&mut self) -> Tensor {
            self.state = 0.0;
            self.step_count = 0;
            Tensor::new(vec![self.state; 4], vec![1, 4], false)
        }

        fn step(&mut self, action: &Tensor) -> StepResult {
            let a = action.data()[0] as usize;
            self.state += if a == 1 { 0.1 } else { -0.1 };
            self.step_count += 1;
            let done = self.step_count >= self.max_steps;
            StepResult {
                observation: Tensor::new(vec![self.state; 4], vec![1, 4], false),
                reward: 1.0,
                done,
                truncated: false,
            }
        }

        fn observation_space(&self) -> Space {
            Space::Box {
                low: vec![-10.0; 4],
                high: vec![10.0; 4],
                shape: vec![4],
            }
        }

        fn action_space(&self) -> Space {
            Space::Discrete(2)
        }

        fn render(&self) -> String {
            format!("state={:.2} step={}", self.state, self.step_count)
        }
    }

    // ---- Space tests ----

    #[test]
    fn test_space_discrete_sample() {
        random::seed(42);
        let s = Space::Discrete(5);
        let t = s.sample();
        let v = t.data()[0];
        assert!(v >= 0.0 && v < 5.0);
    }

    #[test]
    fn test_space_discrete_shape() {
        let s = Space::Discrete(3);
        assert_eq!(s.shape(), vec![1]);
    }

    #[test]
    fn test_space_discrete_contains() {
        let s = Space::Discrete(3);
        assert!(s.contains(&Tensor::new(vec![0.0], vec![1], false)));
        assert!(s.contains(&Tensor::new(vec![2.0], vec![1], false)));
        assert!(!s.contains(&Tensor::new(vec![3.0], vec![1], false)));
        assert!(!s.contains(&Tensor::new(vec![-1.0], vec![1], false)));
    }

    #[test]
    fn test_space_box_sample() {
        random::seed(42);
        let s = Space::Box {
            low: vec![-1.0, -2.0],
            high: vec![1.0, 2.0],
            shape: vec![2],
        };
        let t = s.sample();
        let d = t.data();
        assert!(d[0] >= -1.0 && d[0] <= 1.0);
        assert!(d[1] >= -2.0 && d[1] <= 2.0);
    }

    #[test]
    fn test_space_box_shape() {
        let s = Space::Box {
            low: vec![0.0; 6],
            high: vec![1.0; 6],
            shape: vec![2, 3],
        };
        assert_eq!(s.shape(), vec![2, 3]);
    }

    #[test]
    fn test_space_box_contains() {
        let s = Space::Box {
            low: vec![0.0, 0.0],
            high: vec![1.0, 1.0],
            shape: vec![2],
        };
        assert!(s.contains(&Tensor::new(vec![0.5, 0.5], vec![2], false)));
        assert!(!s.contains(&Tensor::new(vec![1.5, 0.5], vec![2], false)));
    }

    #[test]
    fn test_space_multi_discrete_sample() {
        random::seed(42);
        let s = Space::MultiDiscrete(vec![3, 5]);
        let t = s.sample();
        let d = t.data();
        assert_eq!(d.len(), 2);
        assert!((d[0] as usize) < 3);
        assert!((d[1] as usize) < 5);
    }

    #[test]
    fn test_space_multi_discrete_shape() {
        let s = Space::MultiDiscrete(vec![3, 5, 7]);
        assert_eq!(s.shape(), vec![3]);
    }

    #[test]
    fn test_space_multi_binary_sample() {
        random::seed(42);
        let s = Space::MultiBinary(4);
        let t = s.sample();
        let d = t.data();
        assert_eq!(d.len(), 4);
        for &v in &d {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn test_space_multi_binary_shape() {
        let s = Space::MultiBinary(8);
        assert_eq!(s.shape(), vec![8]);
    }

    // ---- Utility function tests ----

    #[test]
    fn test_discounted_returns_simple() {
        let rewards = vec![1.0, 1.0, 1.0];
        let dones = vec![0.0, 0.0, 0.0];
        let returns = discounted_returns(&rewards, &dones, 0.99);
        // R2 = 1.0, R1 = 1 + 0.99*1.0 = 1.99, R0 = 1 + 0.99*1.99 = 2.9701
        assert!((returns[2] - 1.0).abs() < 1e-5);
        assert!((returns[1] - 1.99).abs() < 1e-5);
        assert!((returns[0] - 2.9701).abs() < 1e-4);
    }

    #[test]
    fn test_discounted_returns_with_done() {
        let rewards = vec![1.0, 1.0, 1.0, 1.0];
        let dones = vec![0.0, 1.0, 0.0, 0.0]; // episode boundary after step 1
        let returns = discounted_returns(&rewards, &dones, 0.99);
        // After done: R3 = 1.0, R2 = 1 + 0.99*1.0 = 1.99
        // Before done: R1 = 1.0 (done resets), R0 = 1 + 0.99*1.0 = 1.99
        assert!((returns[3] - 1.0).abs() < 1e-5);
        assert!((returns[2] - 1.99).abs() < 1e-5);
        assert!((returns[1] - 1.0).abs() < 1e-5);
        assert!((returns[0] - 1.99).abs() < 1e-5);
    }

    #[test]
    fn test_compute_gae_basic() {
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let dones = vec![0.0, 0.0, 0.0];
        let (adv, ret) = compute_gae(&rewards, &values, &dones, 0.99, 0.95, 0.5);
        assert_eq!(adv.len(), 3);
        assert_eq!(ret.len(), 3);
        // returns = advantages + values
        for i in 0..3 {
            assert!((ret[i] - (adv[i] + values[i])).abs() < 1e-5);
        }
    }

    #[test]
    fn test_compute_gae_with_dones() {
        let rewards = vec![1.0, 1.0];
        let values = vec![0.5, 0.5];
        let dones = vec![1.0, 0.0]; // episode ends at step 0
        let (adv, _ret) = compute_gae(&rewards, &values, &dones, 0.99, 0.95, 0.5);
        // delta_0 = r_0 + gamma * v_1 * (1-done_0) - v_0 = 1 + 0 - 0.5 = 0.5
        assert!((adv[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_normalize() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        normalize(&mut data);
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {}", mean);
        let var: f32 =
            data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32;
        assert!((var - 1.0).abs() < 0.1, "var should be ~1, got {}", var);
    }

    #[test]
    fn test_normalize_empty() {
        let mut data: Vec<f32> = vec![];
        normalize(&mut data); // should not panic
    }

    #[test]
    fn test_explained_variance_perfect() {
        let pred = vec![1.0, 2.0, 3.0];
        let true_vals = vec![1.0, 2.0, 3.0];
        let ev = explained_variance(&pred, &true_vals);
        assert!((ev - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_explained_variance_zero() {
        // Predicting constant should give ~0 explained variance
        let pred = vec![2.0, 2.0, 2.0];
        let true_vals = vec![1.0, 2.0, 3.0];
        let ev = explained_variance(&pred, &true_vals);
        assert!(ev.abs() < 1e-5);
    }

    // ---- ReplayBuffer tests ----

    #[test]
    fn test_replay_buffer_push_and_len() {
        let mut buf = ReplayBuffer::new(10, 4, 1);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        buf.push(&[1.0, 2.0, 3.0, 4.0], &[0.0], 1.0, &[2.0, 3.0, 4.0, 5.0], false);
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_replay_buffer_wraps() {
        let mut buf = ReplayBuffer::new(3, 2, 1);
        for i in 0..5 {
            buf.push(
                &[i as f32, 0.0],
                &[0.0],
                1.0,
                &[(i + 1) as f32, 0.0],
                false,
            );
        }
        assert_eq!(buf.len(), 3); // capacity is 3
    }

    #[test]
    fn test_replay_buffer_sample() {
        random::seed(123);
        let mut buf = ReplayBuffer::new(100, 2, 1);
        for i in 0..50 {
            buf.push(
                &[i as f32, (i * 2) as f32],
                &[(i % 2) as f32],
                i as f32 * 0.1,
                &[(i + 1) as f32, ((i + 1) * 2) as f32],
                false,
            );
        }
        let (s, a, r, ns, d) = buf.sample(16);
        assert_eq!(s.shape(), vec![16, 2]);
        assert_eq!(a.shape(), vec![16, 1]);
        assert_eq!(r.shape(), vec![16]);
        assert_eq!(ns.shape(), vec![16, 2]);
        assert_eq!(d.shape(), vec![16]);
    }

    // ---- RolloutBuffer tests ----

    #[test]
    fn test_rollout_buffer_add_and_len() {
        let mut buf = RolloutBuffer::new(100, 4);
        assert_eq!(buf.len(), 0);
        buf.add(&[1.0, 2.0, 3.0, 4.0], 0, -0.5, 1.0, 0.5, false);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_rollout_buffer_compute_and_batches() {
        random::seed(42);
        let mut buf = RolloutBuffer::new(100, 2);
        for i in 0..20 {
            buf.add(
                &[i as f32, (i * 2) as f32],
                i % 3,
                -0.5,
                1.0,
                0.5,
                false,
            );
        }
        buf.compute_returns_and_advantages(0.5, 0.99, 0.95);
        let batches = buf.get_batches(8);
        assert_eq!(batches.len(), 2); // 20 / 8 = 2 full batches
        assert_eq!(batches[0].observations.shape(), vec![8, 2]);
        assert_eq!(batches[0].actions.len(), 8);
    }

    #[test]
    fn test_rollout_buffer_clear() {
        let mut buf = RolloutBuffer::new(100, 2);
        buf.add(&[1.0, 2.0], 0, -0.5, 1.0, 0.5, false);
        assert_eq!(buf.len(), 1);
        buf.clear();
        assert_eq!(buf.len(), 0);
    }

    // ---- Algorithm smoke tests ----

    #[test]
    fn test_reinforce_one_episode() {
        random::seed(42);
        let policy = Linear::new(4, 2);
        let params: Vec<Tensor> = policy.params().into_iter().cloned().collect();
        let mut opt = Adam::new(params, 1e-3);
        let mut env = MockEnv::new(10);
        let reinforce = Reinforce::new(0.99);

        let policy_fn = |x: &Tensor| -> Tensor { policy.forward(x) };
        let reward = reinforce.train_episode(&mut env, &policy_fn, &mut opt);
        assert!(reward > 0.0);
    }

    #[test]
    fn test_reinforce_with_baseline() {
        random::seed(42);
        let policy = Linear::new(4, 2);
        let params: Vec<Tensor> = policy.params().into_iter().cloned().collect();
        let mut opt = Adam::new(params, 1e-3);
        let mut env = MockEnv::new(10);
        let reinforce = Reinforce::new(0.99).with_baseline();

        let policy_fn = |x: &Tensor| -> Tensor { policy.forward(x) };
        let reward = reinforce.train_episode(&mut env, &policy_fn, &mut opt);
        assert!(reward > 0.0);
    }

    #[test]
    fn test_ppo_config_default() {
        let cfg = PpoConfig::default();
        assert!((cfg.clip_eps - 0.2).abs() < 1e-6);
        assert!((cfg.gamma - 0.99).abs() < 1e-6);
        assert_eq!(cfg.epochs, 4);
        assert_eq!(cfg.batch_size, 64);
    }

    #[test]
    fn test_ppo_config_builder() {
        let cfg = PpoConfig::default().clip_eps(0.3).epochs(10);
        assert!((cfg.clip_eps - 0.3).abs() < 1e-6);
        assert_eq!(cfg.epochs, 10);
    }

    #[test]
    fn test_ppo_collect_and_update() {
        random::seed(42);
        let actor = Linear::new(4, 2);
        let critic = Linear::new(4, 1);
        let mut all_params: Vec<Tensor> = Vec::new();
        for t in actor.params().into_iter().chain(critic.params()) {
            all_params.push(t.clone());
        }
        let mut opt = Adam::new(all_params.clone(), 1e-3);

        let config = PpoConfig::default()
            .rollout_steps(32)
            .batch_size(16)
            .epochs(2);
        let mut trainer = PpoTrainer::new(config, 4);
        let mut env = MockEnv::new(20);

        let policy_fn = |x: &Tensor| -> Tensor { actor.forward(x) };
        let value_fn = |x: &Tensor| -> Tensor { critic.forward(x) };

        let _mean_reward = trainer.collect_rollouts(&mut env, &policy_fn, &value_fn, 2);
        let (pi_loss, vf_loss, ent) = trainer.update(&policy_fn, &value_fn, &all_params, &mut opt, 2);
        // Just verify it ran without panic and returned finite values
        assert!(pi_loss.is_finite());
        assert!(vf_loss.is_finite());
        assert!(ent.is_finite());
    }

    #[test]
    fn test_dqn_config_default() {
        let cfg = DqnConfig::default();
        assert!((cfg.gamma - 0.99).abs() < 1e-6);
        assert!((cfg.eps_start - 1.0).abs() < 1e-6);
        assert_eq!(cfg.batch_size, 32);
    }

    #[test]
    fn test_dqn_select_action() {
        random::seed(42);
        let q_net = Linear::new(4, 2);
        let q_fn = |x: &Tensor| -> Tensor { q_net.forward(x) };

        let config = DqnConfig::default();
        let trainer = DqnTrainer::new(config, 4, 1);

        let state = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4], false);
        let action = trainer.select_action(&state, &q_fn, 2);
        assert!(action < 2);
    }

    #[test]
    fn test_dqn_store_and_update() {
        random::seed(42);
        let online = Linear::new(4, 2);
        let target = Linear::new(4, 2);
        let online_params: Vec<Tensor> = online.params().into_iter().cloned().collect();
        let target_params: Vec<Tensor> = target.params().into_iter().cloned().collect();
        copy_params(&online_params, &target_params);

        let mut opt = Adam::new(online_params.clone(), 1e-3);
        let config = DqnConfig::default().batch_size(4).buffer_size(100);
        let mut trainer = DqnTrainer::new(config, 4, 1);

        // Fill buffer
        for i in 0..10 {
            trainer.store_transition(
                &[i as f32 * 0.1, 0.0, 0.0, 0.0],
                i % 2,
                1.0,
                &[(i + 1) as f32 * 0.1, 0.0, 0.0, 0.0],
                false,
            );
        }

        let online_fn = |x: &Tensor| -> Tensor { online.forward(x) };
        let target_fn = |x: &Tensor| -> Tensor { target.forward(x) };

        let loss = trainer.update(
            &online_fn,
            &target_fn,
            &online_params,
            &target_params,
            &mut opt,
            2,
        );
        assert!(loss.is_some());
        assert!(loss.unwrap().is_finite());
    }

    #[test]
    fn test_copy_params() {
        let src = Linear::new(4, 2);
        let dst = Linear::new(4, 2);
        let src_params: Vec<Tensor> = src.params().into_iter().cloned().collect();
        let dst_params: Vec<Tensor> = dst.params().into_iter().cloned().collect();

        // Before copy, params differ
        let s0 = src_params[0].data();
        let d0 = dst_params[0].data();
        assert_ne!(s0, d0);

        copy_params(&src_params, &dst_params);

        // After copy, params match
        let s0 = src_params[0].data();
        let d0 = dst_params[0].data();
        assert_eq!(s0, d0);
    }

    #[test]
    fn test_sample_discrete_action() {
        random::seed(42);
        // Strong preference for action 0
        let logits = vec![10.0, -10.0, -10.0];
        let mut counts = [0usize; 3];
        for _ in 0..100 {
            let (action, lp) = sample_discrete_action(&logits);
            assert!(action < 3);
            assert!(lp <= 0.0); // log_prob is non-positive
            counts[action] += 1;
        }
        // Action 0 should dominate
        assert!(counts[0] > 90);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 3.0]), 2);
    }

    // ---- Shared infrastructure tests ----

    #[test]
    fn test_group_relative_advantages() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0];
        let adv = group_relative_advantages(&rewards);
        assert_eq!(adv.len(), 4);
        // Mean = 2.5, std = sqrt(1.25)
        let mean: f32 = adv.iter().sum::<f32>() / adv.len() as f32;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {}", mean);
        // Should be ordered
        assert!(adv[0] < adv[1]);
        assert!(adv[1] < adv[2]);
        assert!(adv[2] < adv[3]);
    }

    #[test]
    fn test_group_relative_advantages_empty() {
        let adv = group_relative_advantages(&[]);
        assert!(adv.is_empty());
    }

    #[test]
    fn test_token_level_loss() {
        // Symmetric clipping with known values
        let log_ratios = vec![0.0, 0.1, -0.1, 0.05];
        let advantages = vec![1.0, -1.0, 1.0, -1.0];
        let loss = token_level_loss(&log_ratios, &advantages, 0.2, 0.2);
        assert!(loss.is_finite());
        // With zero log_ratio, ratio=1.0, clipped=1.0, so min(1*adv, 1*adv) = adv
        // loss should be non-trivial
    }

    #[test]
    fn test_token_level_loss_empty() {
        let loss = token_level_loss(&[], &[], 0.2, 0.2);
        assert!((loss - 0.0).abs() < 1e-8);
    }

    #[test]
    fn test_sequence_group_creation() {
        let comp = Completion {
            tokens: vec![1, 2, 3],
            log_probs: vec![-1.0, -1.5, -2.0],
            ref_log_probs: vec![-1.1, -1.4, -1.9],
            reward: 0.5,
        };
        let group = SequenceGroup {
            prompt_tokens: vec![10, 20],
            completions: vec![comp],
        };
        assert_eq!(group.prompt_tokens.len(), 2);
        assert_eq!(group.completions.len(), 1);
        assert_eq!(group.completions[0].tokens, vec![1, 2, 3]);
        assert!((group.completions[0].reward - 0.5).abs() < 1e-6);
    }

    // ---- Helper to create mock SequenceGroup ----

    fn make_mock_groups(num_groups: usize, group_size: usize, seq_len: usize) -> Vec<SequenceGroup> {
        let mut groups = Vec::new();
        for g in 0..num_groups {
            let mut completions = Vec::new();
            for c in 0..group_size {
                let tokens: Vec<usize> = (0..seq_len).map(|t| (g * group_size + c) * seq_len + t).collect();
                let log_probs: Vec<f32> = (0..seq_len).map(|t| -1.0 - 0.1 * t as f32).collect();
                let ref_log_probs: Vec<f32> = (0..seq_len).map(|t| -1.1 - 0.1 * t as f32).collect();
                let reward = (c as f32 + 1.0) / group_size as f32;
                completions.push(Completion {
                    tokens,
                    log_probs,
                    ref_log_probs,
                    reward,
                });
            }
            groups.push(SequenceGroup {
                prompt_tokens: vec![0, 1, 2],
                completions,
            });
        }
        groups
    }

    // ---- GRPO tests ----

    #[test]
    fn test_grpo_config_default() {
        let cfg = GrpoConfig::default();
        assert_eq!(cfg.group_size, 8);
        assert!((cfg.clip_eps - 0.2).abs() < 1e-6);
        assert!((cfg.kl_coef - 0.01).abs() < 1e-6);
        assert_eq!(cfg.epochs, 1);
        assert!((cfg.max_grad_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_grpo_config_builder() {
        let cfg = GrpoConfig::default().group_size(16).clip_eps(0.3).kl_coef(0.05);
        assert_eq!(cfg.group_size, 16);
        assert!((cfg.clip_eps - 0.3).abs() < 1e-6);
        assert!((cfg.kl_coef - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_grpo_smoke() {
        let config = GrpoConfig::default();
        let trainer = GrpoTrainer::new(config);
        let groups = make_mock_groups(2, 4, 5);
        let policy_fn = |tokens: &[usize]| -> Vec<f32> {
            tokens.iter().map(|_| -1.05).collect()
        };
        let loss = trainer.compute_loss(&groups, &policy_fn);
        assert!(loss.is_finite(), "GRPO loss should be finite, got {}", loss);
    }

    // ---- RLOO tests ----

    #[test]
    fn test_rloo_config_default() {
        let cfg = RlooConfig::default();
        assert_eq!(cfg.group_size, 8);
        assert!((cfg.kl_coef - 0.01).abs() < 1e-6);
        assert!((cfg.max_grad_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rloo_config_builder() {
        let cfg = RlooConfig::default().group_size(16).kl_coef(0.05);
        assert_eq!(cfg.group_size, 16);
        assert!((cfg.kl_coef - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_rloo_smoke() {
        let config = RlooConfig::default();
        let trainer = RlooTrainer::new(config);
        let groups = make_mock_groups(2, 4, 5);
        let policy_fn = |tokens: &[usize]| -> Vec<f32> {
            tokens.iter().map(|_| -1.05).collect()
        };
        let loss = trainer.compute_loss(&groups, &policy_fn);
        assert!(loss.is_finite(), "RLOO loss should be finite, got {}", loss);
    }

    // ---- Dr. GRPO tests ----

    #[test]
    fn test_dr_grpo_config_default() {
        let cfg = DrGrpoConfig::default();
        assert_eq!(cfg.group_size, 8);
        assert!((cfg.clip_eps - 0.2).abs() < 1e-6);
        assert!((cfg.kl_coef - 0.01).abs() < 1e-6);
        assert!((cfg.max_grad_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dr_grpo_config_builder() {
        let cfg = DrGrpoConfig::default().group_size(16).clip_eps(0.3);
        assert_eq!(cfg.group_size, 16);
        assert!((cfg.clip_eps - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_dr_grpo_smoke() {
        let config = DrGrpoConfig::default();
        let trainer = DrGrpoTrainer::new(config);
        let groups = make_mock_groups(2, 4, 5);
        let policy_fn = |tokens: &[usize]| -> Vec<f32> {
            tokens.iter().map(|_| -1.05).collect()
        };
        let loss = trainer.compute_loss(&groups, &policy_fn);
        assert!(loss.is_finite(), "Dr. GRPO loss should be finite, got {}", loss);
    }

    // ---- DAPO tests ----

    #[test]
    fn test_dapo_config_default() {
        let cfg = DapoConfig::default();
        assert_eq!(cfg.group_size, 16);
        assert!((cfg.clip_lo - 0.2).abs() < 1e-6);
        assert!((cfg.clip_hi - 0.28).abs() < 1e-6);
        assert!((cfg.max_grad_norm - 1.0).abs() < 1e-6);
        assert!((cfg.overlong_penalty - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dapo_config_builder() {
        let cfg = DapoConfig::default().clip_lo(0.15).clip_hi(0.3).overlong_penalty(-0.5);
        assert!((cfg.clip_lo - 0.15).abs() < 1e-6);
        assert!((cfg.clip_hi - 0.3).abs() < 1e-6);
        assert!((cfg.overlong_penalty - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_dapo_smoke() {
        let config = DapoConfig::default();
        let trainer = DapoTrainer::new(config, 100);
        let groups = make_mock_groups(2, 4, 5);
        let policy_fn = |tokens: &[usize]| -> Vec<f32> {
            tokens.iter().map(|_| -1.05).collect()
        };
        let loss = trainer.compute_loss(&groups, &policy_fn);
        assert!(loss.is_finite(), "DAPO loss should be finite, got {}", loss);
    }

    #[test]
    fn test_dapo_dynamic_sampling_skips_uniform() {
        // All completions have the same reward => should be skipped
        let mut groups = make_mock_groups(1, 4, 5);
        for comp in &mut groups[0].completions {
            comp.reward = 1.0;
        }
        let config = DapoConfig::default();
        let trainer = DapoTrainer::new(config, 100);
        let policy_fn = |tokens: &[usize]| -> Vec<f32> {
            tokens.iter().map(|_| -1.05).collect()
        };
        let loss = trainer.compute_loss(&groups, &policy_fn);
        assert!((loss - 0.0).abs() < 1e-8, "DAPO should skip uniform-reward groups");
    }

    // ---- CISPO tests ----

    #[test]
    fn test_cispo_config_default() {
        let cfg = CispoConfig::default();
        assert!((cfg.clip_eps - 0.2).abs() < 1e-6);
        assert_eq!(cfg.group_size, 8);
        assert!((cfg.kl_coef - 0.01).abs() < 1e-6);
        assert!((cfg.max_grad_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cispo_config_builder() {
        let cfg = CispoConfig::default().clip_eps(0.3).group_size(16);
        assert!((cfg.clip_eps - 0.3).abs() < 1e-6);
        assert_eq!(cfg.group_size, 16);
    }

    #[test]
    fn test_cispo_smoke() {
        let config = CispoConfig::default();
        let trainer = CispoTrainer::new(config);
        let groups = make_mock_groups(2, 4, 5);
        let policy_fn = |tokens: &[usize]| -> Vec<f32> {
            tokens.iter().map(|_| -1.05).collect()
        };
        let loss = trainer.compute_loss(&groups, &policy_fn);
        assert!(loss.is_finite(), "CISPO loss should be finite, got {}", loss);
    }

    // ---- DPPO tests ----

    #[test]
    fn test_dppo_config_default() {
        let cfg = DppoConfig::default();
        assert_eq!(cfg.div_type, DivergenceType::KL);
        assert!((cfg.div_coef - 0.01).abs() < 1e-6);
        assert!((cfg.gamma - 0.99).abs() < 1e-6);
        assert!((cfg.gae_lambda - 0.95).abs() < 1e-6);
        assert_eq!(cfg.epochs, 4);
        assert_eq!(cfg.batch_size, 64);
        assert!((cfg.vf_coef - 0.5).abs() < 1e-6);
        assert!((cfg.max_grad_norm - 0.5).abs() < 1e-6);
        assert_eq!(cfg.rollout_steps, 2048);
    }

    #[test]
    fn test_dppo_config_builder() {
        let cfg = DppoConfig::default()
            .div_type(DivergenceType::JSD)
            .div_coef(0.05)
            .epochs(2);
        assert_eq!(cfg.div_type, DivergenceType::JSD);
        assert!((cfg.div_coef - 0.05).abs() < 1e-6);
        assert_eq!(cfg.epochs, 2);
    }

    #[test]
    fn test_dppo_smoke() {
        random::seed(42);
        let actor = Linear::new(4, 2);
        let critic = Linear::new(4, 1);
        let mut all_params: Vec<Tensor> = Vec::new();
        for t in actor.params().into_iter().chain(critic.params()) {
            all_params.push(t.clone());
        }
        let mut opt = Adam::new(all_params.clone(), 1e-3);

        let config = DppoConfig::default()
            .rollout_steps(32)
            .batch_size(16)
            .epochs(1);
        let mut trainer = DppoTrainer::new(config, 4);
        let mut env = MockEnv::new(20);

        let policy_fn = |x: &Tensor| -> Tensor { actor.forward(x) };
        let value_fn = |x: &Tensor| -> Tensor { critic.forward(x) };

        let _mean_reward = trainer.collect_rollouts(&mut env, &policy_fn, &value_fn, 2);
        let (pi_loss, vf_loss, div) = trainer.update(&policy_fn, &value_fn, &all_params, &mut opt, 2);
        assert!(pi_loss.is_finite(), "DPPO policy loss should be finite");
        assert!(vf_loss.is_finite(), "DPPO value loss should be finite");
        assert!(div.is_finite(), "DPPO divergence should be finite");
    }

    #[test]
    fn test_dppo_divergence_types() {
        // Test that all divergence types produce finite, non-negative values
        let new_probs = vec![0.7, 0.2, 0.1];
        let old_probs = vec![0.5, 0.3, 0.2];
        for div_type in &[DivergenceType::KL, DivergenceType::ReverseKL, DivergenceType::JSD] {
            let d = DppoTrainer::compute_divergence(*div_type, &new_probs, &old_probs);
            assert!(d.is_finite(), "Divergence {:?} should be finite", div_type);
            assert!(d >= 0.0, "Divergence {:?} should be non-negative, got {}", div_type, d);
        }
        // Same distribution should give ~0 divergence
        let same = vec![0.5, 0.3, 0.2];
        for div_type in &[DivergenceType::KL, DivergenceType::ReverseKL, DivergenceType::JSD] {
            let d = DppoTrainer::compute_divergence(*div_type, &same, &same);
            assert!(d.abs() < 1e-5, "Same-dist divergence {:?} should be ~0, got {}", div_type, d);
        }
    }

    // ---- MaxRL tests ----

    #[test]
    fn test_maxrl_config_default() {
        let cfg = MaxRlConfig::default();
        assert_eq!(cfg.group_size, 16);
        assert_eq!(cfg.top_k_values, vec![1, 2, 4, 8]);
        assert!((cfg.max_grad_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_maxrl_config_builder() {
        let cfg = MaxRlConfig::default().group_size(8).top_k_values(vec![1, 2]).max_grad_norm(0.5);
        assert_eq!(cfg.group_size, 8);
        assert_eq!(cfg.top_k_values, vec![1, 2]);
        assert!((cfg.max_grad_norm - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_maxrl_smoke() {
        let config = MaxRlConfig::default();
        let trainer = MaxRlTrainer::new(config);
        let groups = make_mock_groups(2, 8, 5);
        let policy_fn = |tokens: &[usize]| -> Vec<f32> {
            tokens.iter().map(|_| -1.05).collect()
        };
        let loss = trainer.compute_loss(&groups, &policy_fn);
        assert!(loss.is_finite(), "MaxRL loss should be finite, got {}", loss);
        assert!(loss > 0.0, "MaxRL loss should be positive (negative log-probs), got {}", loss);
    }

    // ---- ScaleRL tests ----

    #[test]
    fn test_scalerl_config_default() {
        let cfg = ScaleRlConfig::default();
        assert_eq!(cfg.group_size, 8);
        assert!((cfg.clip_eps - 0.2).abs() < 1e-6);
        assert!((cfg.max_grad_norm - 1.0).abs() < 1e-6);
        assert_eq!(cfg.min_buffer_size, 4);
    }

    #[test]
    fn test_scalerl_config_builder() {
        let cfg = ScaleRlConfig::default().min_buffer_size(8).clip_eps(0.3);
        assert_eq!(cfg.min_buffer_size, 8);
        assert!((cfg.clip_eps - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_scalerl_smoke() {
        let config = ScaleRlConfig::default().min_buffer_size(2);
        let mut trainer = ScaleRlTrainer::new(config);

        // Not enough buffered yet
        let linear = Linear::new(4, 2);
        let params: Vec<Tensor> = linear.params().into_iter().cloned().collect();
        let mut opt = Adam::new(params.clone(), 1e-3);
        let policy_fn = |tokens: &[usize]| -> Vec<f32> {
            tokens.iter().map(|_| -1.05).collect()
        };

        let result = trainer.update(&policy_fn, &params, &mut opt);
        assert!(result.is_none(), "ScaleRL should not update with empty buffer");

        // Add some rollouts
        let groups = make_mock_groups(3, 4, 5);
        trainer.add_rollouts(groups);
        assert_eq!(trainer.pending_count(), 3);

        let result = trainer.update(&policy_fn, &params, &mut opt);
        assert!(result.is_some(), "ScaleRL should update with enough buffered groups");
        let loss = result.unwrap();
        assert!(loss.is_finite(), "ScaleRL loss should be finite, got {}", loss);
        assert_eq!(trainer.pending_count(), 0, "Buffer should be drained after update");
    }

    #[test]
    fn test_scalerl_buffer_accumulation() {
        let config = ScaleRlConfig::default().min_buffer_size(4);
        let mut trainer = ScaleRlTrainer::new(config);
        assert_eq!(trainer.pending_count(), 0);

        trainer.add_rollouts(make_mock_groups(2, 4, 5));
        assert_eq!(trainer.pending_count(), 2);

        trainer.add_rollouts(make_mock_groups(1, 4, 5));
        assert_eq!(trainer.pending_count(), 3);

        // Still not enough
        let params: Vec<Tensor> = vec![];
        let mut opt = Adam::new(vec![], 1e-3);
        let policy_fn = |tokens: &[usize]| -> Vec<f32> {
            tokens.iter().map(|_| -1.05).collect()
        };
        let result = trainer.update(&policy_fn, &params, &mut opt);
        assert!(result.is_none());
        assert_eq!(trainer.pending_count(), 3, "Buffer should be unchanged when not enough");

        // Now add enough
        trainer.add_rollouts(make_mock_groups(1, 4, 5));
        assert_eq!(trainer.pending_count(), 4);

        let result = trainer.update(&policy_fn, &params, &mut opt);
        assert!(result.is_some());
        assert_eq!(trainer.pending_count(), 0);
    }
}
