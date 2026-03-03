/// MobaEnv: wraps MobaGame as an RL Environment.
///
/// Each game tick requires 3 step() calls (one per Team A hero).
/// On the 3rd call, the opponent's actions are queried and game.tick() is executed.

use crate::entities::Team;
use crate::game::{GameEvent, MobaGame, NUM_ACTIONS, NUM_HEROES, OBS_DIM};
use peregrine::rl::{Environment, Space, StepResult};
use peregrine::tensor::Tensor;

/// Opponent policy: either scripted or a frozen neural network callback.
pub enum OpponentPolicy {
    Scripted,
    Callback(Box<dyn FnMut(&[f32], usize) -> usize>), // (obs, hero_id) -> action
}

pub struct MobaEnv {
    pub game: MobaGame,
    hero_cursor: usize, // 0, 1, 2 cycling through team A heroes
    opponent: OpponentPolicy,
    /// Per-hero accumulated reward within current tick
    pending_rewards: [f32; 3],
    pub episode_reward: f32,
    pub wins: usize,
    pub losses: usize,
    pub games_played: usize,
}

impl MobaEnv {
    pub fn new(seed: u64) -> Self {
        MobaEnv {
            game: MobaGame::new(seed),
            hero_cursor: 0,
            opponent: OpponentPolicy::Scripted,
            pending_rewards: [0.0; 3],
            episode_reward: 0.0,
            wins: 0,
            losses: 0,
            games_played: 0,
        }
    }

    pub fn set_opponent(&mut self, policy: OpponentPolicy) {
        self.opponent = policy;
    }

    /// Compute reward from game events for team A heroes.
    fn compute_rewards(&mut self) -> [f32; 3] {
        let mut rewards = [0.0f32; 3];
        for event in &self.game.events {
            match event {
                GameEvent::HeroKill { killer_id, .. } => {
                    if *killer_id < 3 {
                        // Team A hero got a kill
                        rewards[*killer_id] += 1.5; // 50% self
                        for r in rewards.iter_mut().take(3) {
                            *r += 0.5; // 50% team split
                        }
                    } else {
                        // Team B killed someone - penalty for team A
                        // (covered by HeroDeath)
                    }
                }
                GameEvent::HeroDeath { hero_id } => {
                    if *hero_id < 3 {
                        rewards[*hero_id] -= 1.0;
                    }
                }
                GameEvent::TowerDestroyed { team, by_hero } => {
                    if *team == Team::B {
                        // Team A destroyed enemy tower
                        if let Some(hi) = by_hero {
                            if *hi < 3 {
                                rewards[*hi] += 2.5; // 50% self
                            }
                        }
                        for r in rewards.iter_mut().take(3) {
                            *r += 0.833; // 50% split across 3
                        }
                    } else {
                        // Team A lost a tower
                        for r in rewards.iter_mut().take(3) {
                            *r -= 3.0;
                        }
                    }
                }
                GameEvent::CreepKill { hero_id } => {
                    if *hero_id < 3 {
                        rewards[*hero_id] += 0.2;
                    }
                }
                GameEvent::BaseDestroyed { team } => {
                    if *team == Team::B {
                        // Team A wins
                        for r in rewards.iter_mut().take(3) {
                            *r += 10.0;
                        }
                    } else {
                        // Team A loses
                        for r in rewards.iter_mut().take(3) {
                            *r -= 10.0;
                        }
                    }
                }
                GameEvent::DamageDealt { hero_id, amount } => {
                    if *hero_id < 3 {
                        rewards[*hero_id] += 0.01 * amount;
                    }
                }
            }
        }
        rewards
    }
}

impl Environment for MobaEnv {
    fn reset(&mut self) -> Tensor {
        self.game.reset();
        self.hero_cursor = 0;
        self.pending_rewards = [0.0; 3];
        self.episode_reward = 0.0;
        let obs = self.game.observe(0);
        Tensor::new(obs, vec![1, OBS_DIM], false)
    }

    fn step(&mut self, action: &Tensor) -> StepResult {
        let act = action.data()[0] as usize;
        let hero_id = self.hero_cursor;

        // Buffer team A hero action
        self.game.apply_hero_action(hero_id, act);
        self.hero_cursor += 1;

        if self.hero_cursor >= 3 {
            // All 3 team A heroes have acted — query opponent and tick
            for opp_id in 3..NUM_HEROES {
                let opp_action = match &mut self.opponent {
                    OpponentPolicy::Scripted => self.game.scripted_action(opp_id),
                    OpponentPolicy::Callback(f) => {
                        let obs = self.game.observe(opp_id);
                        f(&obs, opp_id)
                    }
                };
                self.game.apply_hero_action(opp_id, opp_action);
            }

            self.game.tick();
            self.hero_cursor = 0;

            // Compute rewards from events
            let tick_rewards = self.compute_rewards();
            for i in 0..3 {
                self.pending_rewards[i] += tick_rewards[i];
            }

            // Return reward for hero 2 (just acted), give obs for hero 0 (next)
            let reward = self.pending_rewards[hero_id];
            self.pending_rewards[hero_id] = 0.0;
            self.episode_reward += reward;

            let done = self.game.done;
            if done {
                self.games_played += 1;
                if self.game.winner == Some(Team::A) {
                    self.wins += 1;
                } else if self.game.winner == Some(Team::B) {
                    self.losses += 1;
                }
            }

            let next_obs = self.game.observe(0);
            StepResult {
                observation: Tensor::new(next_obs, vec![1, OBS_DIM], false),
                reward,
                done,
                truncated: done && self.game.winner.is_none(),
            }
        } else {
            // Not all heroes acted yet — return next hero's obs with 0 reward
            // Accumulate any pending reward for previous hero
            let next_hero = self.hero_cursor;
            let reward = self.pending_rewards[hero_id];
            self.pending_rewards[hero_id] = 0.0;
            self.episode_reward += reward;

            let next_obs = self.game.observe(next_hero);
            StepResult {
                observation: Tensor::new(next_obs, vec![1, OBS_DIM], false),
                reward,
                done: false,
                truncated: false,
            }
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-1.0; OBS_DIM],
            high: vec![1.0; OBS_DIM],
            shape: vec![OBS_DIM],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(NUM_ACTIONS)
    }

    fn render(&self) -> String {
        let mut grid = vec![vec!['.'; MAP_W]; MAP_H];

        // Draw lane
        for x in 0..MAP_W {
            grid[7][x] = '-';
            grid[8][x] = '-';
        }

        // Draw bases
        for b in &self.game.bases {
            if b.alive {
                let ch = match b.team { Team::A => 'A', Team::B => 'B' };
                for x in b.x_min..=b.x_max {
                    grid[b.y][x] = ch;
                }
            }
        }

        // Draw towers
        for t in &self.game.towers {
            if t.alive {
                let ch = match t.team { Team::A => 'a', Team::B => 'b' };
                grid[t.y][t.x] = ch;
            }
        }

        // Draw creeps
        for c in &self.game.creeps {
            if c.alive {
                let ch = match c.team { Team::A => 'o', Team::B => 'x' };
                grid[c.y][c.x] = ch;
            }
        }

        // Draw heroes
        for h in &self.game.heroes {
            if h.alive {
                let ch = match h.team {
                    Team::A => (b'1' + h.id as u8 % 3) as char,
                    Team::B => (b'4' + (h.id as u8 - 3) % 3) as char,
                };
                grid[h.y][h.x] = ch;
            }
        }

        let mut s = format!("Tick {}/{}\n", self.game.tick, crate::game::MAX_TICKS);
        for row in &grid {
            s.extend(row);
            s.push('\n');
        }
        s
    }
}

const MAP_W: usize = crate::game::MAP_W;
const MAP_H: usize = crate::game::MAP_H;
