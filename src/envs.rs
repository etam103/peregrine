//! Reinforcement learning environments for Peregrine.
//!
//! Classic control, grid/navigation, reasoning/math, logic, and game environments.
//! All implement the [`Environment`] trait from [`crate::rl`]; reasoning environments
//! additionally implement [`ReasoningEnv`].

use crate::rl::{Environment, ReasoningEnv, Space, StepResult};
use crate::tensor::Tensor;

// ============================================================================
// Local PRNG (SplitMix64) — deterministic per-environment, avoids global state
// ============================================================================

struct LocalRng {
    state: u64,
}

impl LocalRng {
    fn new(seed: u64) -> Self {
        LocalRng {
            state: if seed == 0 { 0xdeadbeefcafe } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform float in `[low, high)`.
    fn range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + self.next_f32() * (high - low)
    }

    /// Uniform integer in `[0, n)`.
    fn range_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform integer in `[low, high)`.
    fn range_i64(&mut self, low: i64, high: i64) -> i64 {
        low + (self.next_u64() % (high - low) as u64) as i64
    }

    /// Returns `true` with probability `p`.
    fn bernoulli(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }
}

// ============================================================================
// CartPole
// ============================================================================

/// Classic CartPole balancing environment.
///
/// Observation: `[x, x_dot, theta, theta_dot]` (4D).
/// Actions: `0` = push left, `1` = push right.
/// Reward: +1 per step. Episode ends when pole angle exceeds 12 degrees
/// or cart leaves `[-2.4, 2.4]`.
pub struct CartPole {
    gravity: f32,
    _masscart: f32,
    masspole: f32,
    total_mass: f32,
    length: f32,
    polemass_length: f32,
    force_mag: f32,
    dt: f32,
    theta_threshold: f32,
    x_threshold: f32,
    max_steps: usize,
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
    step_count: usize,
    rng: LocalRng,
}

impl CartPole {
    pub fn new(seed: u64) -> Self {
        let masscart = 1.0;
        let masspole = 0.1;
        let length = 0.5;
        CartPole {
            gravity: 9.8,
            _masscart: masscart,
            masspole,
            total_mass: masscart + masspole,
            length,
            polemass_length: masspole * length,
            force_mag: 10.0,
            dt: 0.02,
            theta_threshold: 12.0f32.to_radians(),
            x_threshold: 2.4,
            max_steps: 500,
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            step_count: 0,
            rng: LocalRng::new(seed),
        }
    }

    fn obs_tensor(&self) -> Tensor {
        Tensor::new(
            vec![self.x, self.x_dot, self.theta, self.theta_dot],
            vec![1, 4],
            false,
        )
    }

}

impl Environment for CartPole {
    fn reset(&mut self) -> Tensor {
        self.x = self.rng.range_f32(-0.05, 0.05);
        self.x_dot = self.rng.range_f32(-0.05, 0.05);
        self.theta = self.rng.range_f32(-0.05, 0.05);
        self.theta_dot = self.rng.range_f32(-0.05, 0.05);
        self.step_count = 0;
        self.obs_tensor()
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        let a = action.data()[0] as usize;
        let force = if a == 1 {
            self.force_mag
        } else {
            -self.force_mag
        };

        let costh = self.theta.cos();
        let sinth = self.theta.sin();
        let temp = (force + self.polemass_length * self.theta_dot * self.theta_dot * sinth)
            / self.total_mass;
        let theta_acc = (self.gravity * sinth - costh * temp)
            / (self.length * (4.0 / 3.0 - self.masspole * costh * costh / self.total_mass));
        let x_acc = temp - self.polemass_length * theta_acc * costh / self.total_mass;

        self.x += self.dt * self.x_dot;
        self.x_dot += self.dt * x_acc;
        self.theta += self.dt * self.theta_dot;
        self.theta_dot += self.dt * theta_acc;
        self.step_count += 1;

        let truncated = self.step_count >= self.max_steps;
        let failed = self.theta.abs() > self.theta_threshold || self.x.abs() > self.x_threshold;
        let done = failed;
        let reward = if failed { 0.0 } else { 1.0 };

        StepResult {
            observation: self.obs_tensor(),
            reward,
            done,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-4.8, -f32::MAX, -0.42, -f32::MAX],
            high: vec![4.8, f32::MAX, 0.42, f32::MAX],
            shape: vec![4],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(2)
    }

    fn render(&self) -> String {
        let pos = ((self.x + 2.4) / 4.8 * 40.0) as usize;
        let pos = pos.min(39);
        let mut line = vec![' '; 40];
        line[pos] = if self.theta.abs() < self.theta_threshold / 2.0 {
            '|'
        } else {
            '/'
        };
        format!(
            "[{}] x={:.2} th={:.1}deg step={}",
            line.iter().collect::<String>(),
            self.x,
            self.theta.to_degrees(),
            self.step_count
        )
    }
}

// ============================================================================
// MountainCar
// ============================================================================

/// Classic MountainCar environment.
///
/// Observation: `[position, velocity]` (2D).
/// Actions: `0` = push left, `1` = no push, `2` = push right.
/// Reward: -1 per step. Done when position >= 0.5.
pub struct MountainCar {
    position: f32,
    velocity: f32,
    max_steps: usize,
    step_count: usize,
    rng: LocalRng,
}

impl MountainCar {
    pub fn new(seed: u64) -> Self {
        MountainCar {
            position: -0.5,
            velocity: 0.0,
            max_steps: 200,
            step_count: 0,
            rng: LocalRng::new(seed),
        }
    }

    fn obs_tensor(&self) -> Tensor {
        Tensor::new(vec![self.position, self.velocity], vec![1, 2], false)
    }
}

impl Environment for MountainCar {
    fn reset(&mut self) -> Tensor {
        self.position = self.rng.range_f32(-0.6, -0.4);
        self.velocity = 0.0;
        self.step_count = 0;
        self.obs_tensor()
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        let a = action.data()[0] as i32;
        let force = 0.001;
        let gravity = 0.0025;

        self.velocity += (a - 1) as f32 * force - (3.0 * self.position).cos() * gravity;
        self.velocity = self.velocity.clamp(-0.07, 0.07);
        self.position += self.velocity;
        self.position = self.position.clamp(-1.2, 0.6);
        if self.position <= -1.2 {
            self.velocity = 0.0;
        }
        self.step_count += 1;

        let done = self.position >= 0.5;
        let truncated = self.step_count >= self.max_steps;

        StepResult {
            observation: self.obs_tensor(),
            reward: -1.0,
            done,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-1.2, -0.07],
            high: vec![0.6, 0.07],
            shape: vec![2],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(3)
    }

    fn render(&self) -> String {
        let norm = (self.position + 1.2) / 1.8;
        let pos = (norm * 40.0) as usize;
        let pos = pos.min(39);
        let mut line = vec!['-'; 40];
        line[pos] = 'C';
        // Draw the hill
        for i in 0..40 {
            let x = -1.2 + i as f32 * 1.8 / 40.0;
            let height = (3.0 * x).sin();
            if height > 0.3 && line[i] != 'C' {
                line[i] = '^';
            }
        }
        format!(
            "[{}] pos={:.3} vel={:.4} step={}",
            line.iter().collect::<String>(),
            self.position,
            self.velocity,
            self.step_count
        )
    }
}

// ============================================================================
// GridWorld
// ============================================================================

/// NxN grid navigation environment.
///
/// Observation: flat grid of `size*size` floats (0=empty, 0.33=agent, 0.67=obstacle, 1.0=goal).
/// Actions: `0`=up, `1`=down, `2`=left, `3`=right.
/// Rewards: +1 goal, -1 obstacle, -0.01 step cost.
pub struct GridWorld {
    size: usize,
    agent_pos: (usize, usize),
    goal_pos: (usize, usize),
    obstacles: Vec<(usize, usize)>,
    done: bool,
    step_count: usize,
    max_steps: usize,
    rng: LocalRng,
}

impl GridWorld {
    pub fn new(size: usize, num_obstacles: usize, seed: u64) -> Self {
        let mut gw = GridWorld {
            size,
            agent_pos: (0, 0),
            goal_pos: (size - 1, size - 1),
            obstacles: Vec::new(),
            done: false,
            step_count: 0,
            max_steps: size * size * 4,
            rng: LocalRng::new(seed),
        };
        gw.place_obstacles(num_obstacles);
        gw
    }

    fn place_obstacles(&mut self, num_obstacles: usize) {
        self.obstacles.clear();
        let mut count = 0;
        let mut attempts = 0;
        while count < num_obstacles && attempts < num_obstacles * 10 {
            let r = self.rng.range_usize(self.size);
            let c = self.rng.range_usize(self.size);
            if (r, c) != self.agent_pos
                && (r, c) != self.goal_pos
                && !self.obstacles.contains(&(r, c))
            {
                self.obstacles.push((r, c));
                count += 1;
            }
            attempts += 1;
        }
    }

    fn obs_tensor(&self) -> Tensor {
        let n = self.size * self.size;
        let mut grid = vec![0.0f32; n];
        // Mark obstacles
        for &(r, c) in &self.obstacles {
            grid[r * self.size + c] = 0.67;
        }
        // Mark goal
        grid[self.goal_pos.0 * self.size + self.goal_pos.1] = 1.0;
        // Mark agent (overwrites if overlapping)
        grid[self.agent_pos.0 * self.size + self.agent_pos.1] = 0.33;
        Tensor::new(grid, vec![1, n], false)
    }
}

impl Environment for GridWorld {
    fn reset(&mut self) -> Tensor {
        self.agent_pos = (0, 0);
        self.done = false;
        self.step_count = 0;
        self.obs_tensor()
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        if self.done {
            return StepResult {
                observation: self.obs_tensor(),
                reward: 0.0,
                done: true,
                truncated: false,
            };
        }

        let a = action.data()[0] as usize;
        let (r, c) = self.agent_pos;
        let new_pos = match a {
            0 => (r.saturating_sub(1), c),            // up
            1 => ((r + 1).min(self.size - 1), c),     // down
            2 => (r, c.saturating_sub(1)),             // left
            3 => (r, (c + 1).min(self.size - 1)),     // right
            _ => (r, c),
        };
        self.agent_pos = new_pos;
        self.step_count += 1;

        let reward;
        if self.agent_pos == self.goal_pos {
            reward = 1.0;
            self.done = true;
        } else if self.obstacles.contains(&self.agent_pos) {
            reward = -1.0;
            self.done = true;
        } else {
            reward = -0.01;
        }

        let truncated = self.step_count >= self.max_steps;
        if truncated {
            self.done = true;
        }

        StepResult {
            observation: self.obs_tensor(),
            reward,
            done: self.done,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        let n = self.size * self.size;
        Space::Box {
            low: vec![0.0; n],
            high: vec![1.0; n],
            shape: vec![n],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(4)
    }

    fn render(&self) -> String {
        let mut lines = Vec::new();
        for r in 0..self.size {
            let mut row = String::new();
            for c in 0..self.size {
                if (r, c) == self.agent_pos {
                    row.push('A');
                } else if (r, c) == self.goal_pos {
                    row.push('G');
                } else if self.obstacles.contains(&(r, c)) {
                    row.push('#');
                } else {
                    row.push('.');
                }
            }
            lines.push(row);
        }
        lines.join("\n")
    }
}

// ============================================================================
// FrozenLake
// ============================================================================

/// FrozenLake environment with stochastic transitions.
///
/// Grid cells: `S`=start, `F`=frozen, `H`=hole, `G`=goal.
/// Observation: one-hot position vector.
/// Actions: `0`=left, `1`=down, `2`=right, `3`=up.
/// Movement is stochastic: 1/3 intended direction, 1/3 each perpendicular.
pub struct FrozenLake {
    size: usize,
    grid: Vec<u8>, // b'S', b'F', b'H', b'G'
    agent_pos: usize,
    done: bool,
    step_count: usize,
    max_steps: usize,
    rng: LocalRng,
}

impl FrozenLake {
    /// Create a 4x4 FrozenLake with the classic layout.
    pub fn new_4x4(seed: u64) -> Self {
        // Classic 4x4 layout:
        // S F F F
        // F H F H
        // F F F F
        // H F F G
        let grid = vec![
            b'S', b'F', b'F', b'F', b'F', b'H', b'F', b'H', b'F', b'F', b'F', b'F', b'H',
            b'F', b'F', b'G',
        ];
        FrozenLake {
            size: 4,
            grid,
            agent_pos: 0,
            done: false,
            step_count: 0,
            max_steps: 100,
            rng: LocalRng::new(seed),
        }
    }

    /// Create an 8x8 FrozenLake with a random layout.
    pub fn new_8x8(seed: u64) -> Self {
        let mut rng = LocalRng::new(seed);
        let size = 8;
        let n = size * size;
        let mut grid = vec![b'F'; n];
        grid[0] = b'S';
        grid[n - 1] = b'G';
        // Place some holes
        let num_holes = 12;
        let mut placed = 0;
        while placed < num_holes {
            let pos = rng.range_usize(n);
            if pos != 0 && pos != n - 1 && grid[pos] == b'F' {
                grid[pos] = b'H';
                placed += 1;
            }
        }
        FrozenLake {
            size,
            grid,
            agent_pos: 0,
            done: false,
            step_count: 0,
            max_steps: 200,
            rng,
        }
    }

    fn obs_tensor(&self) -> Tensor {
        let n = self.size * self.size;
        let mut data = vec![0.0f32; n];
        data[self.agent_pos] = 1.0;
        Tensor::new(data, vec![1, n], false)
    }

    fn pos_to_rc(&self, pos: usize) -> (usize, usize) {
        (pos / self.size, pos % self.size)
    }

    fn rc_to_pos(&self, r: usize, c: usize) -> usize {
        r * self.size + c
    }

    fn move_in_dir(&self, pos: usize, dir: usize) -> usize {
        let (r, c) = self.pos_to_rc(pos);
        match dir {
            0 => self.rc_to_pos(r, c.saturating_sub(1)),            // left
            1 => self.rc_to_pos((r + 1).min(self.size - 1), c),     // down
            2 => self.rc_to_pos(r, (c + 1).min(self.size - 1)),     // right
            3 => self.rc_to_pos(r.saturating_sub(1), c),            // up
            _ => pos,
        }
    }
}

impl Environment for FrozenLake {
    fn reset(&mut self) -> Tensor {
        self.agent_pos = 0;
        self.done = false;
        self.step_count = 0;
        self.obs_tensor()
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        if self.done {
            return StepResult {
                observation: self.obs_tensor(),
                reward: 0.0,
                done: true,
                truncated: false,
            };
        }

        let intended = action.data()[0] as usize;
        // Stochastic: 1/3 intended, 1/3 clockwise, 1/3 counter-clockwise
        let r = self.rng.next_f32();
        let actual_dir = if r < 1.0 / 3.0 {
            intended
        } else if r < 2.0 / 3.0 {
            (intended + 1) % 4 // clockwise
        } else {
            (intended + 3) % 4 // counter-clockwise
        };

        self.agent_pos = self.move_in_dir(self.agent_pos, actual_dir);
        self.step_count += 1;

        let cell = self.grid[self.agent_pos];
        let (reward, done) = match cell {
            b'H' => (0.0, true),
            b'G' => (1.0, true),
            _ => (0.0, false),
        };
        self.done = done;
        let truncated = self.step_count >= self.max_steps;
        if truncated {
            self.done = true;
        }

        StepResult {
            observation: self.obs_tensor(),
            reward,
            done: self.done,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        let n = self.size * self.size;
        Space::Box {
            low: vec![0.0; n],
            high: vec![1.0; n],
            shape: vec![n],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(4)
    }

    fn render(&self) -> String {
        let mut lines = Vec::new();
        for r in 0..self.size {
            let mut row = String::new();
            for c in 0..self.size {
                let pos = self.rc_to_pos(r, c);
                if pos == self.agent_pos {
                    row.push('A');
                } else {
                    row.push(self.grid[pos] as char);
                }
            }
            lines.push(row);
        }
        lines.join("\n")
    }
}

// ============================================================================
// BasicArithmetic (ReasoningEnv)
// ============================================================================

/// Single-operation arithmetic: `a op b = ?`.
///
/// Tensor interface: obs = `[a, op_id, b]`, action = answer index.
/// Text interface: `question()` = "What is 7 + 3?", `score_answer("10")` = 1.0.
pub struct BasicArithmetic {
    pub max_val: i64,
    pub num_answer_bins: usize,
    pub answer_offset: i64,
    pub a: i64,
    pub b: i64,
    pub op: usize,     // 0=+, 1=-, 2=*
    pub correct: i64,
    allowed_ops: Vec<usize>,
    done: bool,
    rng: LocalRng,
}

impl BasicArithmetic {
    /// Create with operands in `[0, max_val]` and all operations (+, -, *).
    pub fn new(max_val: i64, seed: u64) -> Self {
        Self::with_ops(max_val, vec![0, 1, 2], seed)
    }

    /// Create with only addition (`a + b`). Smaller action space, easier to learn.
    pub fn addition_only(max_val: i64, seed: u64) -> Self {
        Self::with_ops(max_val, vec![0], seed)
    }

    /// Create with a specific set of operations (0=+, 1=-, 2=*).
    pub fn with_ops(max_val: i64, ops: Vec<usize>, seed: u64) -> Self {
        let lo = if ops.contains(&1) { -max_val } else { 0 };
        let hi = if ops.contains(&2) {
            max_val * max_val
        } else {
            max_val * 2
        };
        let num_bins = (hi - lo + 1) as usize;
        let mut env = BasicArithmetic {
            max_val,
            num_answer_bins: num_bins,
            answer_offset: lo,
            a: 0,
            b: 0,
            op: 0,
            correct: 0,
            allowed_ops: ops,
            done: false,
            rng: LocalRng::new(seed),
        };
        env.generate();
        env
    }

    fn generate(&mut self) {
        self.a = self.rng.range_i64(0, self.max_val + 1);
        self.b = self.rng.range_i64(0, self.max_val + 1);
        let idx = self.rng.range_usize(self.allowed_ops.len());
        self.op = self.allowed_ops[idx];
        self.correct = match self.op {
            0 => self.a + self.b,
            1 => self.a - self.b,
            2 => self.a * self.b,
            _ => unreachable!(),
        };
        self.done = false;
    }

    fn op_str(&self) -> &str {
        match self.op {
            0 => "+",
            1 => "-",
            2 => "*",
            _ => "?",
        }
    }
}

impl Environment for BasicArithmetic {
    fn reset(&mut self) -> Tensor {
        self.generate();
        Tensor::new(
            vec![self.a as f32, self.op as f32, self.b as f32],
            vec![1, 3],
            false,
        )
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        let answer_idx = action.data()[0] as i64;
        let answer = answer_idx + self.answer_offset;
        let reward = if answer == self.correct { 1.0 } else { 0.0 };
        self.done = true;

        StepResult {
            observation: Tensor::new(
                vec![self.a as f32, self.op as f32, self.b as f32],
                vec![1, 3],
                false,
            ),
            reward,
            done: true,
            truncated: false,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0, 0.0, 0.0],
            high: vec![self.max_val as f32, 2.0, self.max_val as f32],
            shape: vec![3],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(self.num_answer_bins)
    }

    fn render(&self) -> String {
        format!(
            "{} {} {} = ? (correct: {})",
            self.a,
            self.op_str(),
            self.b,
            self.correct
        )
    }
}

impl ReasoningEnv for BasicArithmetic {
    fn question(&self) -> String {
        format!("What is {} {} {}?", self.a, self.op_str(), self.b)
    }

    fn answer(&self) -> String {
        self.correct.to_string()
    }

    fn score_answer(&self, answer: &str) -> f32 {
        match answer.trim().parse::<i64>() {
            Ok(v) if v == self.correct => 1.0,
            _ => 0.0,
        }
    }

    fn metadata(&self) -> Vec<(String, String)> {
        vec![
            ("type".into(), "basic_arithmetic".into()),
            ("op".into(), self.op_str().into()),
            ("difficulty".into(), self.max_val.to_string()),
        ]
    }
}

// ============================================================================
// ChainArithmetic (ReasoningEnv)
// ============================================================================

/// Multi-operation chain arithmetic: `a op1 b op2 c ...` (left-to-right evaluation).
///
/// Configurable chain length and operand range.
pub struct ChainArithmetic {
    max_val: i64,
    chain_len: usize,
    num_answer_bins: usize,
    answer_offset: i64,
    operands: Vec<i64>,
    ops: Vec<usize>,
    correct: i64,
    done: bool,
    rng: LocalRng,
}

impl ChainArithmetic {
    pub fn new(max_val: i64, chain_len: usize, seed: u64) -> Self {
        assert!(chain_len >= 2, "chain_len must be >= 2");
        let range = max_val.pow(chain_len as u32) * 2 + 1;
        let num_bins = range.min(1001) as usize;
        let offset = -(num_bins as i64 / 2);
        let mut env = ChainArithmetic {
            max_val,
            chain_len,
            num_answer_bins: num_bins,
            answer_offset: offset,
            operands: Vec::new(),
            ops: Vec::new(),
            correct: 0,
            done: false,
            rng: LocalRng::new(seed),
        };
        env.generate();
        env
    }

    fn generate(&mut self) {
        self.operands.clear();
        self.ops.clear();
        for _ in 0..self.chain_len {
            self.operands
                .push(self.rng.range_i64(0, self.max_val + 1));
        }
        for _ in 0..self.chain_len - 1 {
            self.ops.push(self.rng.range_usize(3));
        }
        // Evaluate left-to-right
        let mut result = self.operands[0];
        for i in 0..self.ops.len() {
            result = match self.ops[i] {
                0 => result + self.operands[i + 1],
                1 => result - self.operands[i + 1],
                2 => result * self.operands[i + 1],
                _ => unreachable!(),
            };
        }
        self.correct = result;
        self.done = false;
    }

    fn op_str(op: usize) -> &'static str {
        match op {
            0 => "+",
            1 => "-",
            2 => "*",
            _ => "?",
        }
    }

    fn expr_string(&self) -> String {
        let mut s = self.operands[0].to_string();
        for i in 0..self.ops.len() {
            s.push_str(&format!(" {} {}", Self::op_str(self.ops[i]), self.operands[i + 1]));
        }
        s
    }
}

impl Environment for ChainArithmetic {
    fn reset(&mut self) -> Tensor {
        self.generate();
        // Observation: interleaved [operand, op, operand, op, ..., operand]
        let mut data = Vec::new();
        for i in 0..self.chain_len {
            data.push(self.operands[i] as f32);
            if i < self.ops.len() {
                data.push(self.ops[i] as f32);
            }
        }
        let len = data.len();
        Tensor::new(data, vec![1, len], false)
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        let answer_idx = action.data()[0] as i64;
        let answer = answer_idx + self.answer_offset;
        let reward = if answer == self.correct { 1.0 } else { 0.0 };
        self.done = true;

        let mut data = Vec::new();
        for i in 0..self.chain_len {
            data.push(self.operands[i] as f32);
            if i < self.ops.len() {
                data.push(self.ops[i] as f32);
            }
        }
        let len = data.len();

        StepResult {
            observation: Tensor::new(data, vec![1, len], false),
            reward,
            done: true,
            truncated: false,
        }
    }

    fn observation_space(&self) -> Space {
        let dim = self.chain_len * 2 - 1;
        Space::Box {
            low: vec![0.0; dim],
            high: vec![self.max_val as f32; dim],
            shape: vec![dim],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(self.num_answer_bins)
    }

    fn render(&self) -> String {
        format!("{} = ? (correct: {})", self.expr_string(), self.correct)
    }
}

impl ReasoningEnv for ChainArithmetic {
    fn question(&self) -> String {
        format!("What is {}?", self.expr_string())
    }

    fn answer(&self) -> String {
        self.correct.to_string()
    }

    fn score_answer(&self, answer: &str) -> f32 {
        match answer.trim().parse::<i64>() {
            Ok(v) if v == self.correct => 1.0,
            _ => 0.0,
        }
    }

    fn metadata(&self) -> Vec<(String, String)> {
        vec![
            ("type".into(), "chain_arithmetic".into()),
            ("chain_len".into(), self.chain_len.to_string()),
        ]
    }
}

// ============================================================================
// NumberSorting (ReasoningEnv)
// ============================================================================

/// Sort N numbers in ascending order.
///
/// Text: "Sort: 5, 2, 8, 1" -> "1, 2, 5, 8".
/// Tensor: obs = the N numbers, action sequence predicts sorted permutation.
pub struct NumberSorting {
    count: usize,
    max_val: i64,
    numbers: Vec<i64>,
    sorted: Vec<i64>,
    done: bool,
    rng: LocalRng,
}

impl NumberSorting {
    pub fn new(count: usize, max_val: i64, seed: u64) -> Self {
        let mut env = NumberSorting {
            count,
            max_val,
            numbers: Vec::new(),
            sorted: Vec::new(),
            done: false,
            rng: LocalRng::new(seed),
        };
        env.generate();
        env
    }

    fn generate(&mut self) {
        self.numbers.clear();
        for _ in 0..self.count {
            self.numbers.push(self.rng.range_i64(0, self.max_val + 1));
        }
        self.sorted = self.numbers.clone();
        self.sorted.sort();
        self.done = false;
    }
}

impl Environment for NumberSorting {
    fn reset(&mut self) -> Tensor {
        self.generate();
        let data: Vec<f32> = self.numbers.iter().map(|&v| v as f32).collect();
        Tensor::new(data, vec![1, self.count], false)
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        // Action encodes the predicted sorted sequence as N values
        let pred = action.data();
        let mut correct = 0;
        let check_len = pred.len().min(self.count);
        for i in 0..check_len {
            if (pred[i] as i64) == self.sorted[i] {
                correct += 1;
            }
        }
        let reward = correct as f32 / self.count as f32;
        self.done = true;

        let data: Vec<f32> = self.numbers.iter().map(|&v| v as f32).collect();
        StepResult {
            observation: Tensor::new(data, vec![1, self.count], false),
            reward,
            done: true,
            truncated: false,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0; self.count],
            high: vec![self.max_val as f32; self.count],
            shape: vec![self.count],
        }
    }

    fn action_space(&self) -> Space {
        Space::MultiDiscrete(vec![self.max_val as usize + 1; self.count])
    }

    fn render(&self) -> String {
        let nums: Vec<String> = self.numbers.iter().map(|v| v.to_string()).collect();
        format!("Sort: {}", nums.join(", "))
    }
}

impl ReasoningEnv for NumberSorting {
    fn question(&self) -> String {
        let nums: Vec<String> = self.numbers.iter().map(|v| v.to_string()).collect();
        format!("Sort these numbers: {}", nums.join(", "))
    }

    fn answer(&self) -> String {
        let nums: Vec<String> = self.sorted.iter().map(|v| v.to_string()).collect();
        nums.join(", ")
    }

    fn score_answer(&self, answer: &str) -> f32 {
        let parsed: Vec<i64> = answer
            .split(|c: char| c == ',' || c == ' ')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if parsed == self.sorted {
            1.0
        } else {
            // Partial credit
            let mut correct = 0;
            for (i, &v) in parsed.iter().enumerate() {
                if i < self.sorted.len() && v == self.sorted[i] {
                    correct += 1;
                }
            }
            correct as f32 / self.count as f32
        }
    }

    fn metadata(&self) -> Vec<(String, String)> {
        vec![
            ("type".into(), "number_sorting".into()),
            ("count".into(), self.count.to_string()),
        ]
    }
}

// ============================================================================
// SequenceCompletion (ReasoningEnv)
// ============================================================================

/// Predict the next element in a sequence (arithmetic, geometric, or Fibonacci-like).
///
/// Text: "Complete: 2, 4, 6, 8, ?" -> "10"
pub struct SequenceCompletion {
    seq_len: usize,
    max_val: i64,
    sequence: Vec<i64>,
    answer_val: i64,
    seq_type: usize, // 0=arithmetic, 1=geometric, 2=fibonacci
    done: bool,
    rng: LocalRng,
}

impl SequenceCompletion {
    pub fn new(seq_len: usize, max_val: i64, seed: u64) -> Self {
        let mut env = SequenceCompletion {
            seq_len,
            max_val,
            sequence: Vec::new(),
            answer_val: 0,
            seq_type: 0,
            done: false,
            rng: LocalRng::new(seed),
        };
        env.generate();
        env
    }

    fn generate(&mut self) {
        self.seq_type = self.rng.range_usize(3);
        self.sequence.clear();

        match self.seq_type {
            0 => {
                // Arithmetic: a, a+d, a+2d, ...
                let a = self.rng.range_i64(0, self.max_val / 2);
                let d = self.rng.range_i64(1, (self.max_val / (self.seq_len as i64 + 1)).max(2));
                for i in 0..self.seq_len {
                    self.sequence.push(a + d * i as i64);
                }
                self.answer_val = a + d * self.seq_len as i64;
            }
            1 => {
                // Geometric: a, a*r, a*r^2, ...
                let a = self.rng.range_i64(1, 5);
                let r = self.rng.range_i64(2, 4);
                for i in 0..self.seq_len {
                    self.sequence.push(a * r.pow(i as u32));
                }
                self.answer_val = a * r.pow(self.seq_len as u32);
            }
            2 => {
                // Fibonacci-like: a, b, a+b, a+2b, ...
                let a = self.rng.range_i64(1, 5);
                let b = self.rng.range_i64(1, 5);
                self.sequence.push(a);
                if self.seq_len > 1 {
                    self.sequence.push(b);
                }
                for i in 2..self.seq_len {
                    let next = self.sequence[i - 1] + self.sequence[i - 2];
                    self.sequence.push(next);
                }
                if self.seq_len >= 2 {
                    self.answer_val =
                        self.sequence[self.seq_len - 1] + self.sequence[self.seq_len - 2];
                } else {
                    self.answer_val = a; // degenerate
                }
            }
            _ => unreachable!(),
        }
        self.done = false;
    }
}

impl Environment for SequenceCompletion {
    fn reset(&mut self) -> Tensor {
        self.generate();
        let data: Vec<f32> = self.sequence.iter().map(|&v| v as f32).collect();
        Tensor::new(data, vec![1, self.seq_len], false)
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        let predicted = action.data()[0] as i64;
        let reward = if predicted == self.answer_val {
            1.0
        } else {
            0.0
        };
        self.done = true;

        let data: Vec<f32> = self.sequence.iter().map(|&v| v as f32).collect();
        StepResult {
            observation: Tensor::new(data, vec![1, self.seq_len], false),
            reward,
            done: true,
            truncated: false,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0; self.seq_len],
            high: vec![self.max_val as f32; self.seq_len],
            shape: vec![self.seq_len],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete((self.max_val * 2 + 1) as usize) // wide range for answers
    }

    fn render(&self) -> String {
        let nums: Vec<String> = self.sequence.iter().map(|v| v.to_string()).collect();
        format!("{}, ? (correct: {})", nums.join(", "), self.answer_val)
    }
}

impl ReasoningEnv for SequenceCompletion {
    fn question(&self) -> String {
        let nums: Vec<String> = self.sequence.iter().map(|v| v.to_string()).collect();
        format!("Complete the sequence: {}, ?", nums.join(", "))
    }

    fn answer(&self) -> String {
        self.answer_val.to_string()
    }

    fn score_answer(&self, answer: &str) -> f32 {
        match answer.trim().parse::<i64>() {
            Ok(v) if v == self.answer_val => 1.0,
            _ => 0.0,
        }
    }

    fn metadata(&self) -> Vec<(String, String)> {
        let seq_name = match self.seq_type {
            0 => "arithmetic",
            1 => "geometric",
            2 => "fibonacci",
            _ => "unknown",
        };
        vec![
            ("type".into(), "sequence_completion".into()),
            ("pattern".into(), seq_name.into()),
        ]
    }
}

// ============================================================================
// PropositionalLogic (ReasoningEnv)
// ============================================================================

/// Evaluate propositional logic expressions.
///
/// Generates random expressions like `(A AND B) OR (NOT C)` with random variable
/// assignments, and asks for the boolean result.
pub struct PropositionalLogic {
    max_depth: usize,
    num_vars: usize,
    vars: Vec<bool>,
    var_names: Vec<String>,
    expr: LogicExpr,
    correct: bool,
    done: bool,
    rng: LocalRng,
}

#[derive(Clone, Debug)]
enum LogicExpr {
    Var(usize),
    Not(Box<LogicExpr>),
    And(Box<LogicExpr>, Box<LogicExpr>),
    Or(Box<LogicExpr>, Box<LogicExpr>),
}

impl LogicExpr {
    fn eval(&self, vars: &[bool]) -> bool {
        match self {
            LogicExpr::Var(i) => vars[*i],
            LogicExpr::Not(e) => !e.eval(vars),
            LogicExpr::And(a, b) => a.eval(vars) && b.eval(vars),
            LogicExpr::Or(a, b) => a.eval(vars) || b.eval(vars),
        }
    }

    fn to_string(&self, var_names: &[String]) -> String {
        match self {
            LogicExpr::Var(i) => var_names[*i].clone(),
            LogicExpr::Not(e) => format!("NOT {}", e.to_string(var_names)),
            LogicExpr::And(a, b) => {
                format!(
                    "({} AND {})",
                    a.to_string(var_names),
                    b.to_string(var_names)
                )
            }
            LogicExpr::Or(a, b) => {
                format!(
                    "({} OR {})",
                    a.to_string(var_names),
                    b.to_string(var_names)
                )
            }
        }
    }
}

impl PropositionalLogic {
    pub fn new(max_depth: usize, num_vars: usize, seed: u64) -> Self {
        let var_names: Vec<String> = (0..num_vars)
            .map(|i| ((b'A' + i as u8) as char).to_string())
            .collect();
        let mut env = PropositionalLogic {
            max_depth,
            num_vars,
            vars: vec![false; num_vars],
            var_names,
            expr: LogicExpr::Var(0),
            correct: false,
            done: false,
            rng: LocalRng::new(seed),
        };
        env.generate();
        env
    }

    fn generate(&mut self) {
        // Random variable assignments
        for v in &mut self.vars {
            *v = self.rng.bernoulli(0.5);
        }
        // Generate random expression
        self.expr = self.gen_expr(self.max_depth);
        self.correct = self.expr.eval(&self.vars);
        self.done = false;
    }

    fn gen_expr(&mut self, depth: usize) -> LogicExpr {
        if depth == 0 || self.rng.next_f32() < 0.3 {
            return LogicExpr::Var(self.rng.range_usize(self.num_vars));
        }
        match self.rng.range_usize(3) {
            0 => LogicExpr::Not(Box::new(self.gen_expr(depth - 1))),
            1 => LogicExpr::And(
                Box::new(self.gen_expr(depth - 1)),
                Box::new(self.gen_expr(depth - 1)),
            ),
            2 => LogicExpr::Or(
                Box::new(self.gen_expr(depth - 1)),
                Box::new(self.gen_expr(depth - 1)),
            ),
            _ => unreachable!(),
        }
    }
}

impl Environment for PropositionalLogic {
    fn reset(&mut self) -> Tensor {
        self.generate();
        // Observation: variable values (0/1)
        let data: Vec<f32> = self.vars.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        Tensor::new(data, vec![1, self.num_vars], false)
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        let pred = action.data()[0] as usize;
        let pred_bool = pred != 0;
        let reward = if pred_bool == self.correct { 1.0 } else { 0.0 };
        self.done = true;

        let data: Vec<f32> = self.vars.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        StepResult {
            observation: Tensor::new(data, vec![1, self.num_vars], false),
            reward,
            done: true,
            truncated: false,
        }
    }

    fn observation_space(&self) -> Space {
        Space::MultiBinary(self.num_vars)
    }

    fn action_space(&self) -> Space {
        Space::Discrete(2) // true/false
    }

    fn render(&self) -> String {
        let assignments: Vec<String> = self
            .var_names
            .iter()
            .zip(&self.vars)
            .map(|(n, &v)| format!("{}={}", n, v))
            .collect();
        format!(
            "{} where {} (correct: {})",
            self.expr.to_string(&self.var_names),
            assignments.join(", "),
            self.correct
        )
    }
}

impl ReasoningEnv for PropositionalLogic {
    fn question(&self) -> String {
        let assignments: Vec<String> = self
            .var_names
            .iter()
            .zip(&self.vars)
            .map(|(n, &v)| format!("{}={}", n, v))
            .collect();
        format!(
            "Evaluate: {} where {}",
            self.expr.to_string(&self.var_names),
            assignments.join(", ")
        )
    }

    fn answer(&self) -> String {
        if self.correct {
            "true".into()
        } else {
            "false".into()
        }
    }

    fn score_answer(&self, answer: &str) -> f32 {
        let a = answer.trim().to_lowercase();
        let pred = match a.as_str() {
            "true" | "1" | "yes" => true,
            "false" | "0" | "no" => false,
            _ => return 0.0,
        };
        if pred == self.correct {
            1.0
        } else {
            0.0
        }
    }

    fn metadata(&self) -> Vec<(String, String)> {
        vec![
            ("type".into(), "propositional_logic".into()),
            ("depth".into(), self.max_depth.to_string()),
            ("vars".into(), self.num_vars.to_string()),
        ]
    }
}

// ============================================================================
// TicTacToe
// ============================================================================

/// TicTacToe environment: player X (agent) vs player O (opponent).
///
/// Observation: 9 cells, each encoded as: 0=empty, 1=X(agent), -1=O(opponent).
/// Action: cell index 0-8.
/// Reward: +1 win, -1 loss, 0 draw, -10 invalid move.
pub struct TicTacToe {
    board: [i8; 9], // 0=empty, 1=X, -1=O
    done: bool,
    step_count: usize,
    opponent: TicTacToeOpponent,
    rng: LocalRng,
}

/// Opponent strategy.
pub enum TicTacToeOpponent {
    Random,
    Minimax,
}

impl TicTacToe {
    pub fn new(opponent: TicTacToeOpponent, seed: u64) -> Self {
        TicTacToe {
            board: [0; 9],
            done: false,
            step_count: 0,
            opponent,
            rng: LocalRng::new(seed),
        }
    }

    fn obs_tensor(&self) -> Tensor {
        let data: Vec<f32> = self.board.iter().map(|&v| v as f32).collect();
        Tensor::new(data, vec![1, 9], false)
    }

    fn check_winner(&self) -> Option<i8> {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];
        for line in &LINES {
            let s = self.board[line[0]] + self.board[line[1]] + self.board[line[2]];
            if s == 3 {
                return Some(1);
            }
            if s == -3 {
                return Some(-1);
            }
        }
        None
    }

    fn is_full(&self) -> bool {
        self.board.iter().all(|&v| v != 0)
    }

    fn empty_cells(&self) -> Vec<usize> {
        self.board
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 0)
            .map(|(i, _)| i)
            .collect()
    }

    fn opponent_move(&mut self) {
        let empty = self.empty_cells();
        if empty.is_empty() {
            return;
        }
        let cell = match self.opponent {
            TicTacToeOpponent::Random => {
                let idx = self.rng.range_usize(empty.len());
                empty[idx]
            }
            TicTacToeOpponent::Minimax => self.minimax_move(),
        };
        self.board[cell] = -1;
    }

    fn minimax_move(&mut self) -> usize {
        let empty = self.empty_cells();
        let mut best_score = i32::MAX;
        let mut best_move = empty[0];
        for &cell in &empty {
            self.board[cell] = -1;
            let score = self.minimax(true, 0);
            self.board[cell] = 0;
            if score < best_score {
                best_score = score;
                best_move = cell;
            }
        }
        best_move
    }

    fn minimax(&mut self, is_x: bool, depth: i32) -> i32 {
        if let Some(w) = self.check_winner() {
            return w as i32 * 10 - depth * w as i32; // prefer faster wins
        }
        if self.is_full() {
            return 0;
        }
        let empty = self.empty_cells();
        if is_x {
            let mut best = i32::MIN;
            for cell in &empty {
                self.board[*cell] = 1;
                let score = self.minimax(false, depth + 1);
                self.board[*cell] = 0;
                best = best.max(score);
            }
            best
        } else {
            let mut best = i32::MAX;
            for cell in &empty {
                self.board[*cell] = -1;
                let score = self.minimax(true, depth + 1);
                self.board[*cell] = 0;
                best = best.min(score);
            }
            best
        }
    }
}

impl Environment for TicTacToe {
    fn reset(&mut self) -> Tensor {
        self.board = [0; 9];
        self.done = false;
        self.step_count = 0;
        self.obs_tensor()
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        if self.done {
            return StepResult {
                observation: self.obs_tensor(),
                reward: 0.0,
                done: true,
                truncated: false,
            };
        }

        let cell = action.data()[0] as usize;
        self.step_count += 1;

        // Invalid move
        if cell >= 9 || self.board[cell] != 0 {
            self.done = true;
            return StepResult {
                observation: self.obs_tensor(),
                reward: -10.0,
                done: true,
                truncated: false,
            };
        }

        // Agent plays X
        self.board[cell] = 1;

        // Check if agent wins
        if let Some(1) = self.check_winner() {
            self.done = true;
            return StepResult {
                observation: self.obs_tensor(),
                reward: 1.0,
                done: true,
                truncated: false,
            };
        }

        // Check for draw
        if self.is_full() {
            self.done = true;
            return StepResult {
                observation: self.obs_tensor(),
                reward: 0.0,
                done: true,
                truncated: false,
            };
        }

        // Opponent plays O
        self.opponent_move();

        // Check if opponent wins
        if let Some(-1) = self.check_winner() {
            self.done = true;
            return StepResult {
                observation: self.obs_tensor(),
                reward: -1.0,
                done: true,
                truncated: false,
            };
        }

        // Check for draw after opponent move
        if self.is_full() {
            self.done = true;
            return StepResult {
                observation: self.obs_tensor(),
                reward: 0.0,
                done: true,
                truncated: false,
            };
        }

        StepResult {
            observation: self.obs_tensor(),
            reward: 0.0,
            done: false,
            truncated: false,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-1.0; 9],
            high: vec![1.0; 9],
            shape: vec![9],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(9)
    }

    fn render(&self) -> String {
        let sym = |v: i8| match v {
            1 => 'X',
            -1 => 'O',
            _ => '.',
        };
        format!(
            "{} {} {}\n{} {} {}\n{} {} {}",
            sym(self.board[0]),
            sym(self.board[1]),
            sym(self.board[2]),
            sym(self.board[3]),
            sym(self.board[4]),
            sym(self.board[5]),
            sym(self.board[6]),
            sym(self.board[7]),
            sym(self.board[8]),
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::Environment;

    // ---- CartPole ----

    #[test]
    fn test_cartpole_reset() {
        let mut env = CartPole::new(42);
        let obs = env.reset();
        assert_eq!(obs.shape(), vec![1, 4]);
        let d = obs.data();
        for &v in &d {
            assert!(v.abs() < 0.1); // initial state is small
        }
    }

    #[test]
    fn test_cartpole_step() {
        let mut env = CartPole::new(42);
        env.reset();
        let action = Tensor::new(vec![1.0], vec![1], false);
        let result = env.step(&action);
        assert_eq!(result.observation.shape(), vec![1, 4]);
        assert!(!result.done || result.truncated);
    }

    #[test]
    fn test_cartpole_done_eventually() {
        let mut env = CartPole::new(42);
        env.reset();
        // Always push right — should eventually fail
        let action = Tensor::new(vec![1.0], vec![1], false);
        let mut done = false;
        for _ in 0..1000 {
            let result = env.step(&action);
            if result.done || result.truncated {
                done = true;
                break;
            }
        }
        assert!(done);
    }

    #[test]
    fn test_cartpole_spaces() {
        let env = CartPole::new(42);
        match env.observation_space() {
            Space::Box { shape, .. } => assert_eq!(shape, vec![4]),
            _ => panic!("expected Box space"),
        }
        assert_eq!(env.action_space().n(), 2);
    }

    #[test]
    fn test_cartpole_render() {
        let mut env = CartPole::new(42);
        env.reset();
        let r = env.render();
        assert!(r.contains("x="));
    }

    // ---- MountainCar ----

    #[test]
    fn test_mountaincar_reset() {
        let mut env = MountainCar::new(42);
        let obs = env.reset();
        assert_eq!(obs.shape(), vec![1, 2]);
        let d = obs.data();
        assert!(d[0] >= -0.6 && d[0] <= -0.4);
        assert!((d[1]).abs() < 1e-6);
    }

    #[test]
    fn test_mountaincar_step() {
        let mut env = MountainCar::new(42);
        env.reset();
        let action = Tensor::new(vec![2.0], vec![1], false); // push right
        let result = env.step(&action);
        assert_eq!(result.observation.shape(), vec![1, 2]);
        assert!((result.reward - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_mountaincar_spaces() {
        let env = MountainCar::new(42);
        assert_eq!(env.action_space().n(), 3);
    }

    #[test]
    fn test_mountaincar_truncation() {
        let mut env = MountainCar::new(42);
        env.reset();
        let action = Tensor::new(vec![1.0], vec![1], false); // no push
        let mut truncated = false;
        for _ in 0..300 {
            let result = env.step(&action);
            if result.truncated {
                truncated = true;
                break;
            }
        }
        assert!(truncated);
    }

    // ---- GridWorld ----

    #[test]
    fn test_gridworld_reset() {
        let mut env = GridWorld::new(5, 3, 42);
        let obs = env.reset();
        assert_eq!(obs.shape(), vec![1, 25]);
        let d = obs.data();
        // Agent at (0,0) -> position 0 should be 0.33
        assert!((d[0] - 0.33).abs() < 0.01);
    }

    #[test]
    fn test_gridworld_step() {
        let mut env = GridWorld::new(5, 0, 42);
        env.reset();
        // Move right
        let action = Tensor::new(vec![3.0], vec![1], false);
        let result = env.step(&action);
        assert!(!result.done);
        assert!((result.reward - (-0.01)).abs() < 1e-6);
    }

    #[test]
    fn test_gridworld_reach_goal() {
        let mut env = GridWorld::new(3, 0, 42); // no obstacles
        env.reset();
        // Move to (2,2) from (0,0): right, right, down, down
        let moves = [3, 3, 1, 1]; // right, right, down, down
        let mut total_reward = 0.0;
        for &m in &moves {
            let action = Tensor::new(vec![m as f32], vec![1], false);
            let result = env.step(&action);
            total_reward += result.reward;
            if result.done {
                break;
            }
        }
        assert!(total_reward > 0.5); // should get +1 for goal
    }

    #[test]
    fn test_gridworld_obstacle_collision() {
        // Create grid where (0,1) is guaranteed obstacle
        let mut env = GridWorld::new(5, 0, 42);
        env.obstacles.push((0, 1));
        env.reset();
        // Move right into obstacle
        let action = Tensor::new(vec![3.0], vec![1], false);
        let result = env.step(&action);
        assert!(result.done);
        assert!((result.reward - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_gridworld_render() {
        let mut env = GridWorld::new(3, 0, 42);
        env.reset();
        let r = env.render();
        assert!(r.contains('A'));
        assert!(r.contains('G'));
    }

    // ---- FrozenLake ----

    #[test]
    fn test_frozenlake_reset() {
        let mut env = FrozenLake::new_4x4(42);
        let obs = env.reset();
        assert_eq!(obs.shape(), vec![1, 16]);
        let d = obs.data();
        assert!((d[0] - 1.0).abs() < 1e-6); // agent at position 0
    }

    #[test]
    fn test_frozenlake_step() {
        let mut env = FrozenLake::new_4x4(42);
        env.reset();
        let action = Tensor::new(vec![1.0], vec![1], false); // down
        let result = env.step(&action);
        assert_eq!(result.observation.shape(), vec![1, 16]);
    }

    #[test]
    fn test_frozenlake_8x8() {
        let mut env = FrozenLake::new_8x8(42);
        let obs = env.reset();
        assert_eq!(obs.shape(), vec![1, 64]);
    }

    #[test]
    fn test_frozenlake_render() {
        let mut env = FrozenLake::new_4x4(42);
        env.reset();
        let r = env.render();
        assert!(r.contains('A'));
    }

    // ---- BasicArithmetic ----

    #[test]
    fn test_basic_arithmetic_reset() {
        let mut env = BasicArithmetic::new(9, 42);
        let obs = env.reset();
        assert_eq!(obs.shape(), vec![1, 3]);
        let d = obs.data();
        assert!(d[0] >= 0.0 && d[0] <= 9.0); // a
        assert!(d[1] >= 0.0 && d[1] <= 2.0); // op
        assert!(d[2] >= 0.0 && d[2] <= 9.0); // b
    }

    #[test]
    fn test_basic_arithmetic_correct_answer() {
        let mut env = BasicArithmetic::new(9, 42);
        env.reset();
        let correct = env.correct;
        let action_idx = correct - env.answer_offset;
        let action = Tensor::new(vec![action_idx as f32], vec![1], false);
        let result = env.step(&action);
        assert!((result.reward - 1.0).abs() < 1e-6);
        assert!(result.done);
    }

    #[test]
    fn test_basic_arithmetic_wrong_answer() {
        let mut env = BasicArithmetic::new(9, 42);
        env.reset();
        let wrong = env.correct + 1;
        let action_idx = wrong - env.answer_offset;
        let action = Tensor::new(vec![action_idx as f32], vec![1], false);
        let result = env.step(&action);
        assert!((result.reward).abs() < 1e-6);
    }

    #[test]
    fn test_basic_arithmetic_question() {
        let mut env = BasicArithmetic::new(9, 42);
        env.reset();
        let q = env.question();
        assert!(q.starts_with("What is"));
    }

    #[test]
    fn test_basic_arithmetic_score() {
        let mut env = BasicArithmetic::new(9, 42);
        env.reset();
        let ans = env.answer();
        assert_eq!(env.score_answer(&ans), 1.0);
        assert_eq!(env.score_answer("999999"), 0.0);
    }

    // ---- ChainArithmetic ----

    #[test]
    fn test_chain_arithmetic_reset() {
        let mut env = ChainArithmetic::new(9, 3, 42);
        let obs = env.reset();
        // 3 operands + 2 ops = 5 values
        assert_eq!(obs.shape(), vec![1, 5]);
    }

    #[test]
    fn test_chain_arithmetic_question() {
        let mut env = ChainArithmetic::new(9, 2, 42);
        env.reset();
        let q = env.question();
        assert!(q.contains("What is"));
    }

    #[test]
    fn test_chain_arithmetic_score() {
        let mut env = ChainArithmetic::new(9, 2, 42);
        env.reset();
        let ans = env.answer();
        assert_eq!(env.score_answer(&ans), 1.0);
    }

    // ---- NumberSorting ----

    #[test]
    fn test_number_sorting_reset() {
        let mut env = NumberSorting::new(5, 100, 42);
        let obs = env.reset();
        assert_eq!(obs.shape(), vec![1, 5]);
    }

    #[test]
    fn test_number_sorting_score() {
        let mut env = NumberSorting::new(4, 100, 42);
        env.reset();
        let ans = env.answer();
        assert_eq!(env.score_answer(&ans), 1.0);
    }

    #[test]
    fn test_number_sorting_partial_score() {
        let mut env = NumberSorting::new(4, 100, 42);
        env.reset();
        // Completely wrong answer
        let score = env.score_answer("999, 998, 997, 996");
        assert!(score < 1.0);
    }

    // ---- SequenceCompletion ----

    #[test]
    fn test_sequence_completion_reset() {
        let mut env = SequenceCompletion::new(4, 100, 42);
        let obs = env.reset();
        assert_eq!(obs.shape(), vec![1, 4]);
    }

    #[test]
    fn test_sequence_completion_question() {
        let mut env = SequenceCompletion::new(4, 100, 42);
        env.reset();
        let q = env.question();
        assert!(q.contains("Complete the sequence"));
    }

    #[test]
    fn test_sequence_completion_score() {
        let mut env = SequenceCompletion::new(4, 100, 42);
        env.reset();
        let ans = env.answer();
        assert_eq!(env.score_answer(&ans), 1.0);
    }

    // ---- PropositionalLogic ----

    #[test]
    fn test_propositional_logic_reset() {
        let mut env = PropositionalLogic::new(2, 3, 42);
        let obs = env.reset();
        assert_eq!(obs.shape(), vec![1, 3]);
        let d = obs.data();
        for &v in &d {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn test_propositional_logic_correct_answer() {
        let mut env = PropositionalLogic::new(2, 3, 42);
        env.reset();
        let correct_action = if env.correct { 1.0 } else { 0.0 };
        let action = Tensor::new(vec![correct_action], vec![1], false);
        let result = env.step(&action);
        assert!((result.reward - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_propositional_logic_score() {
        let mut env = PropositionalLogic::new(2, 3, 42);
        env.reset();
        let ans = env.answer();
        assert_eq!(env.score_answer(&ans), 1.0);
    }

    // ---- TicTacToe ----

    #[test]
    fn test_tictactoe_reset() {
        let mut env = TicTacToe::new(TicTacToeOpponent::Random, 42);
        let obs = env.reset();
        assert_eq!(obs.shape(), vec![1, 9]);
        let d = obs.data();
        for &v in &d {
            assert!((v).abs() < 1e-6); // all empty
        }
    }

    #[test]
    fn test_tictactoe_valid_move() {
        let mut env = TicTacToe::new(TicTacToeOpponent::Random, 42);
        env.reset();
        let action = Tensor::new(vec![4.0], vec![1], false); // center
        let result = env.step(&action);
        let d = result.observation.data();
        assert!((d[4] - 1.0).abs() < 1e-6); // agent placed X at center
    }

    #[test]
    fn test_tictactoe_invalid_move() {
        let mut env = TicTacToe::new(TicTacToeOpponent::Random, 42);
        env.reset();
        // Play center
        let action = Tensor::new(vec![4.0], vec![1], false);
        env.step(&action);
        // If center still occupied by X, try it again -> should be caught
        // Actually opponent may have taken a cell. Let's just try an out-of-range move.
        let mut env2 = TicTacToe::new(TicTacToeOpponent::Random, 42);
        env2.reset();
        let action = Tensor::new(vec![9.0], vec![1], false); // out of range
        let result = env2.step(&action);
        assert!(result.done);
        assert!((result.reward - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_tictactoe_game_terminates() {
        let mut env = TicTacToe::new(TicTacToeOpponent::Random, 42);
        env.reset();
        let mut done = false;
        for i in 0..9 {
            if done {
                break;
            }
            let action = Tensor::new(vec![i as f32], vec![1], false);
            let result = env.step(&action);
            if result.done {
                done = true;
            }
        }
        assert!(done); // game must terminate within 9 moves
    }

    #[test]
    fn test_tictactoe_render() {
        let mut env = TicTacToe::new(TicTacToeOpponent::Random, 42);
        env.reset();
        let r = env.render();
        assert!(r.contains('.'));
    }

    #[test]
    fn test_tictactoe_minimax_opponent() {
        let mut env = TicTacToe::new(TicTacToeOpponent::Minimax, 42);
        env.reset();
        let action = Tensor::new(vec![0.0], vec![1], false);
        let result = env.step(&action);
        // Minimax opponent should respond; game continues or ends
        assert_eq!(result.observation.shape(), vec![1, 9]);
    }
}
