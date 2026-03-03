//! MOBA 3v3: LSTM-based PPO with self-play on a single-lane MOBA.
//!
//! Usage:
//!   cargo run --example moba --release -- train        # Train vs scripted opponent
//!   cargo run --example moba --release -- selfplay     # Train with self-play
//!   cargo run --example moba --release -- watch        # Record & render a game
//!   cargo run --example moba --release -- video        # Export replay as MP4 video

mod entities;
mod env;
mod game;
mod policy;
mod render;

use env::{MobaEnv, OpponentPolicy};
use game::{FrameSnapshot, NUM_ACTIONS, OBS_DIM};
use policy::{RecurrentActorCritic, SelfPlayManager};
use render::TrainingMetrics;

use peregrine::nn;
use peregrine::optim::{clip_grad_norm, Adam};
use peregrine::random;
use peregrine::rl::Environment;
use peregrine::tensor::Tensor;

// Training hyperparameters
const ROLLOUT_STEPS: usize = 1536; // 512 ticks x 3 heroes
const BPTT_WINDOW: usize = 16;
const BATCH_SIZE: usize = 128;
const PPO_EPOCHS: usize = 4;
const LR: f32 = 3e-4;
const CLIP_EPS: f32 = 0.2;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
const ENT_COEF: f32 = 0.02;
const VF_COEF: f32 = 0.5;
const MAX_GRAD_NORM: f32 = 0.5;
const HIDDEN_SIZE: usize = 128;

/// Transition stored during rollout collection.
struct Transition {
    obs: Vec<f32>,
    action: usize,
    log_prob: f32,
    reward: f32,
    value: f32,
    done: bool,
}

/// Sample an action from logits, returning (action, log_prob).
fn sample_action(logits: &Tensor) -> (usize, f32) {
    let ld = logits.data();
    let num_actions = ld.len();

    // Softmax
    let max_l = ld.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = ld.iter().map(|&x| (x - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Sample
    let r = random::uniform(&[1], 0.0, 1.0, false).data()[0];
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

/// Compute GAE advantages and returns.
fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    last_value: f32,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    let mut advantages = vec![0.0f32; n];
    let mut gae = 0.0f32;

    for t in (0..n).rev() {
        let next_value = if t == n - 1 { last_value } else { values[t + 1] };
        let next_non_terminal = if t == n - 1 {
            if dones[t] { 0.0 } else { 1.0 }
        } else {
            if dones[t] { 0.0 } else { 1.0 }
        };
        let delta = rewards[t] + gamma * next_value * next_non_terminal - values[t];
        gae = delta + gamma * lambda * next_non_terminal * gae;
        advantages[t] = gae;
    }

    let returns: Vec<f32> = advantages
        .iter()
        .zip(values.iter())
        .map(|(a, v)| a + v)
        .collect();

    (advantages, returns)
}

/// Normalize advantages in-place.
fn normalize_advantages(adv: &mut [f32]) {
    let n = adv.len() as f32;
    let mean: f32 = adv.iter().sum::<f32>() / n;
    let var: f32 = adv.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / n;
    let std = var.sqrt() + 1e-8;
    for a in adv.iter_mut() {
        *a = (*a - mean) / std;
    }
}

/// Collect rollouts without autograd tracking.
fn collect_rollouts(
    env: &mut MobaEnv,
    net: &RecurrentActorCritic,
    rollout_steps: usize,
) -> Vec<Transition> {
    let mut buffer = Vec::with_capacity(rollout_steps);
    let mut h_states = vec![
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
    ];
    let mut c_states = vec![
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
    ];

    let mut obs_data = env.reset().data();
    let mut hero_cursor = 0usize;

    for _ in 0..rollout_steps {
        let obs_tensor = Tensor::new(obs_data.clone(), vec![1, OBS_DIM], false);
        let h = &h_states[hero_cursor];
        let c = &c_states[hero_cursor];

        let (logits, value, h_new, c_new) = net.forward_step(&obs_tensor, h, c);
        let (action, log_prob) = sample_action(&logits);
        let val = value.data()[0];

        let action_tensor = Tensor::new(vec![action as f32], vec![1], false);
        let result = env.step(&action_tensor);

        buffer.push(Transition {
            obs: obs_data,
            action,
            log_prob,
            reward: result.reward,
            value: val,
            done: result.done,
        });

        // Detach hidden states (no grad tracking during collection)
        h_states[hero_cursor] = Tensor::new(h_new.data(), vec![1, HIDDEN_SIZE], false);
        c_states[hero_cursor] = Tensor::new(c_new.data(), vec![1, HIDDEN_SIZE], false);

        if result.done {
            obs_data = env.reset().data();
            hero_cursor = 0;
            for i in 0..3 {
                h_states[i] = Tensor::zeros(&[1, HIDDEN_SIZE], false);
                c_states[i] = Tensor::zeros(&[1, HIDDEN_SIZE], false);
            }
        } else {
            obs_data = result.observation.data();
            hero_cursor = (hero_cursor + 1) % 3;
        }
    }

    buffer
}

/// PPO update with truncated BPTT.
fn ppo_update(
    net: &RecurrentActorCritic,
    buffer: &[Transition],
    params: &[Tensor],
    optimizer: &mut Adam,
) -> (f32, f32, f32) {
    // Extract arrays
    let rewards: Vec<f32> = buffer.iter().map(|t| t.reward).collect();
    let values: Vec<f32> = buffer.iter().map(|t| t.value).collect();
    let dones: Vec<bool> = buffer.iter().map(|t| t.done).collect();

    let last_value = if buffer.last().map_or(true, |t| t.done) {
        0.0
    } else {
        buffer.last().unwrap().value
    };

    let (mut advantages, returns) = compute_gae(&rewards, &values, &dones, last_value, GAMMA, GAE_LAMBDA);
    normalize_advantages(&mut advantages);

    let n = buffer.len();
    let mut total_pi = 0.0f32;
    let mut total_vf = 0.0f32;
    let mut total_ent = 0.0f32;
    let mut updates = 0;

    // Simple xorshift for shuffling
    let mut rng = 12345u64;

    for _epoch in 0..PPO_EPOCHS {
        // Generate batch indices (sequential windows for BPTT, then shuffle)
        let num_windows = n / BPTT_WINDOW;
        let mut window_starts: Vec<usize> = (0..num_windows).map(|i| i * BPTT_WINDOW).collect();

        // Fisher-Yates shuffle
        for i in (1..window_starts.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            window_starts.swap(i, j);
        }

        // Process windows in mini-batches
        for batch_start in (0..window_starts.len()).step_by(BATCH_SIZE / BPTT_WINDOW) {
            let batch_end = (batch_start + BATCH_SIZE / BPTT_WINDOW).min(window_starts.len());
            let batch_windows = &window_starts[batch_start..batch_end];

            let mut batch_pi_loss = 0.0f32;
            let mut batch_vf_loss = 0.0f32;
            let mut batch_entropy = 0.0f32;
            let mut batch_count = 0;

            for &start in batch_windows {
                let end = (start + BPTT_WINDOW).min(n);
                let window_len = end - start;
                if window_len < 2 {
                    continue;
                }

                // Check if window crosses an episode boundary; if so, reset h/c at boundary
                // For simplicity, use zero-initialized hidden state at window start
                let h0 = Tensor::zeros(&[1, HIDDEN_SIZE], false);
                let c0 = Tensor::zeros(&[1, HIDDEN_SIZE], false);

                // Build observation sequence tensor
                let mut obs_flat = Vec::with_capacity(window_len * OBS_DIM);
                for t in start..end {
                    obs_flat.extend_from_slice(&buffer[t].obs);
                }
                let obs_seq = Tensor::new(obs_flat, vec![window_len, OBS_DIM], false);

                // Forward pass with autograd through LSTM
                let (logits, vals, _, _) = net.forward_seq(&obs_seq, &h0, &c0);

                // Compute losses for this window
                let log_probs = logits.log_softmax(-1); // [window_len, NUM_ACTIONS]
                let probs = logits.softmax(-1);

                // Select log probs for taken actions
                let flat_indices: Vec<usize> = (0..window_len)
                    .map(|i| i * NUM_ACTIONS + buffer[start + i].action)
                    .collect();
                let new_log_probs = log_probs.select(&flat_indices); // [window_len]

                // Old log probs
                let old_lp: Vec<f32> = (start..end).map(|t| buffer[t].log_prob).collect();
                let old_lp_tensor = Tensor::new(old_lp, vec![window_len], false);

                // Ratio
                let log_ratio = new_log_probs.sub(&old_lp_tensor);
                let ratio = log_ratio.exp();

                // Advantages for this window
                let win_adv: Vec<f32> = (start..end).map(|t| advantages[t]).collect();
                let adv_tensor = Tensor::new(win_adv, vec![window_len], false);

                // Clipped surrogate
                let surr1 = ratio.mul(&adv_tensor);
                let ratio_clipped = ratio.clip(1.0 - CLIP_EPS, 1.0 + CLIP_EPS);
                let surr2 = ratio_clipped.mul(&adv_tensor);
                let pi_loss = surr1.minimum(&surr2).mean().neg();

                // Value loss
                let values_flat = vals.reshape(vec![window_len]);
                let win_ret: Vec<f32> = (start..end).map(|t| returns[t]).collect();
                let ret_tensor = Tensor::new(win_ret, vec![window_len], false);
                let vf_loss = nn::mse_loss(&values_flat, &ret_tensor);

                // Entropy
                let plogp = probs.mul(&log_probs);
                let entropy = plogp.sum().neg().scale(1.0 / window_len as f32);

                // Total loss
                let loss = pi_loss
                    .add(&vf_loss.scale(VF_COEF))
                    .sub(&entropy.scale(ENT_COEF));

                loss.backward();

                batch_pi_loss += pi_loss.data()[0];
                batch_vf_loss += vf_loss.data()[0];
                batch_entropy += entropy.data()[0];
                batch_count += 1;
            }

            if batch_count > 0 {
                clip_grad_norm(params, MAX_GRAD_NORM);
                optimizer.step();
                optimizer.zero_grad();

                total_pi += batch_pi_loss / batch_count as f32;
                total_vf += batch_vf_loss / batch_count as f32;
                total_ent += batch_entropy / batch_count as f32;
                updates += 1;
            }
        }
    }

    if updates > 0 {
        (
            total_pi / updates as f32,
            total_vf / updates as f32,
            total_ent / updates as f32,
        )
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Record a full game for visualization.
fn record_game(net: &RecurrentActorCritic, seed: u64) -> Vec<FrameSnapshot> {
    let mut env = MobaEnv::new(seed);
    let mut frames = Vec::new();

    let mut h_states = vec![
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
    ];
    let mut c_states = vec![
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
        Tensor::zeros(&[1, HIDDEN_SIZE], false),
    ];

    let mut obs_data = env.reset().data();
    let mut hero_cursor = 0usize;

    frames.push(env.game.snapshot());

    loop {
        let obs_tensor = Tensor::new(obs_data.clone(), vec![1, OBS_DIM], false);
        let (logits, _, h_new, c_new) = net.forward_step(&obs_tensor, &h_states[hero_cursor], &c_states[hero_cursor]);

        let ld = logits.data();
        let action = ld
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        h_states[hero_cursor] = Tensor::new(h_new.data(), vec![1, HIDDEN_SIZE], false);
        c_states[hero_cursor] = Tensor::new(c_new.data(), vec![1, HIDDEN_SIZE], false);

        let action_tensor = Tensor::new(vec![action as f32], vec![1], false);
        let result = env.step(&action_tensor);

        // Record frame after tick (every 3rd step when hero_cursor wraps to 0)
        if hero_cursor == 2 {
            frames.push(env.game.snapshot());
        }

        if result.done {
            break;
        }

        obs_data = result.observation.data();
        hero_cursor = (hero_cursor + 1) % 3;
    }

    frames
}

fn train(mode: &str) {
    let selfplay = mode == "selfplay";
    let total_iterations = if selfplay { 200 } else { 100 };

    println!("=== MOBA 3v3 PPO Training ({}) ===\n", mode);
    random::seed(42);

    let net = RecurrentActorCritic::new();
    let params = net.params();
    let param_count = net.param_count();
    println!("Network: {} parameters", param_count);

    let mut optimizer = Adam::new(params.clone(), LR);
    let mut env = MobaEnv::new(123);

    let mut metrics = TrainingMetrics::new();
    let mut sp_manager = SelfPlayManager::new(10);
    let mut rng_seed = 42u64;

    for iteration in 0..total_iterations {
        // Self-play: save checkpoint and load opponent from past checkpoint
        if selfplay {
            if iteration > 0 && iteration % 20 == 0 {
                sp_manager.save_checkpoint(&params);
            }
            if sp_manager.num_checkpoints() > 0 {
                // Create a fresh opponent network owned by the closure
                let opp_net = RecurrentActorCritic::new();
                let opp_params = opp_net.params();
                sp_manager.load_random_checkpoint(&opp_params, &mut rng_seed);

                let opp_h = std::cell::RefCell::new(vec![
                    Tensor::zeros(&[1, HIDDEN_SIZE], false),
                    Tensor::zeros(&[1, HIDDEN_SIZE], false),
                    Tensor::zeros(&[1, HIDDEN_SIZE], false),
                ]);
                let opp_c = std::cell::RefCell::new(vec![
                    Tensor::zeros(&[1, HIDDEN_SIZE], false),
                    Tensor::zeros(&[1, HIDDEN_SIZE], false),
                    Tensor::zeros(&[1, HIDDEN_SIZE], false),
                ]);
                env.set_opponent(OpponentPolicy::Callback(Box::new(
                    move |obs: &[f32], hero_id: usize| {
                        let idx = hero_id - 3;
                        let obs_t = Tensor::new(obs.to_vec(), vec![1, OBS_DIM], false);
                        let h = opp_h.borrow()[idx].clone();
                        let c = opp_c.borrow()[idx].clone();
                        let (logits, _, h_new, c_new) =
                            opp_net.forward_step(&obs_t, &h, &c);
                        opp_h.borrow_mut()[idx] =
                            Tensor::new(h_new.data(), vec![1, HIDDEN_SIZE], false);
                        opp_c.borrow_mut()[idx] =
                            Tensor::new(c_new.data(), vec![1, HIDDEN_SIZE], false);
                        let ld = logits.data();
                        ld.iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .unwrap()
                            .0
                    },
                )));
            }
        }

        // Collect rollouts
        let buffer = collect_rollouts(&mut env, &net, ROLLOUT_STEPS);

        // Compute stats from buffer
        let mean_reward: f32 = buffer.iter().map(|t| t.reward).sum::<f32>() / buffer.len() as f32;
        let win_rate = if env.games_played > 0 {
            env.wins as f32 / env.games_played as f32
        } else {
            0.0
        };

        // PPO update
        let (pi_loss, vf_loss, entropy) = ppo_update(&net, &buffer, &params, &mut optimizer);

        metrics.rewards.push(mean_reward);
        metrics.win_rates.push(win_rate);
        metrics.pi_losses.push(pi_loss);
        metrics.vf_losses.push(vf_loss);
        metrics.entropies.push(entropy);

        if (iteration + 1) % 5 == 0 || iteration == 0 {
            println!(
                "Iter {:3}: reward={:7.4}  win_rate={:.2}  pi_loss={:7.4}  vf_loss={:7.4}  entropy={:.4}  games={}",
                iteration + 1,
                mean_reward,
                win_rate,
                pi_loss,
                vf_loss,
                entropy,
                env.games_played,
            );
        }
    }

    // Save learning curves
    render::write_learning_curves("rl_moba.html", &metrics);

    // Record and save a game replay
    println!("\nRecording evaluation game...");
    let frames = record_game(&net, 999);
    render::write_replay_animation("rl_moba_anim.html", &frames);

    println!("\nTraining complete. Final win rate: {:.2}", metrics.win_rates.last().unwrap_or(&0.0));
}

fn watch() {
    println!("=== MOBA 3v3 — Watch Mode ===\n");
    random::seed(42);

    let net = RecurrentActorCritic::new();
    println!("Recording game with random policy ({} params)...", net.param_count());

    let frames = record_game(&net, 42);
    render::write_replay_animation("rl_moba_anim.html", &frames);
    println!("Recorded {} frames", frames.len());
}

fn video() {
    println!("=== MOBA 3v3 — Video Export ===\n");
    random::seed(42);

    let net = RecurrentActorCritic::new();
    println!("Recording game with random policy ({} params)...", net.param_count());

    let frames = record_game(&net, 42);
    println!("Recorded {} frames, rendering video...", frames.len());
    render::write_video("rl_moba.mp4", &frames);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("train");

    match mode {
        "train" => train("train"),
        "selfplay" => train("selfplay"),
        "watch" => watch(),
        "video" => video(),
        _ => {
            println!("Usage: moba [train|selfplay|watch|video]");
            println!("  train     — Train vs scripted opponent");
            println!("  selfplay  — Train with self-play");
            println!("  watch     — Record & render a game");
            println!("  video     — Export game replay as MP4 video");
        }
    }
}
