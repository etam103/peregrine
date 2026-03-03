/// MOBA game engine: 3v3 single-lane on a 32x16 grid.
///
/// Map layout (lane along rows 7-8):
///   [Base A]  TowerA2  TowerA1  ----lane----  TowerB1  TowerB2  [Base B]
///    col 0-1   col 5    col 9                  col 22    col 26   col 30-31

use crate::entities::{Base, Creep, Hero, Team, Tower};

pub const MAP_W: usize = 32;
pub const MAP_H: usize = 16;
pub const LANE_Y: usize = 7; // lane occupies rows 7-8
pub const MAX_TICKS: usize = 500;
pub const CREEP_WAVE_INTERVAL: usize = 30;
pub const CREEPS_PER_WAVE: usize = 3;
pub const NUM_HEROES: usize = 6; // 3 per team
pub const NUM_ACTIONS: usize = 20; // 5 move * 4 attack
pub const OBS_DIM: usize = 167;

/// Events generated during a tick for reward shaping.
#[derive(Clone, Debug)]
pub enum GameEvent {
    HeroKill { killer_id: usize, #[allow(dead_code)] victim_id: usize },
    HeroDeath { hero_id: usize },
    TowerDestroyed { team: Team, by_hero: Option<usize> },
    CreepKill { hero_id: usize },
    BaseDestroyed { team: Team },
    DamageDealt { hero_id: usize, amount: f32 },
}

/// Snapshot of one tick for replay visualization.
#[derive(Clone)]
pub struct FrameSnapshot {
    pub tick: usize,
    pub heroes: Vec<(usize, usize, f32, f32, bool, u8)>, // x, y, hp, max_hp, alive, team(0/1)
    pub towers: Vec<(usize, usize, f32, f32, bool, u8)>,
    pub creeps: Vec<(usize, usize, f32, f32, bool, u8)>,
    pub bases: Vec<(f32, f32, bool, u8)>, // hp, max_hp, alive, team
}

pub struct MobaGame {
    pub heroes: Vec<Hero>,
    pub towers: Vec<Tower>,
    pub creeps: Vec<Creep>,
    pub bases: Vec<Base>,
    pub tick: usize,
    pub done: bool,
    pub winner: Option<Team>,
    pub events: Vec<GameEvent>,
    buffered_actions: [usize; NUM_HEROES],
}

impl MobaGame {
    pub fn new(_seed: u64) -> Self {
        let mut game = MobaGame {
            heroes: Vec::new(),
            towers: Vec::new(),
            creeps: Vec::new(),
            bases: Vec::new(),
            tick: 0,
            done: false,
            winner: None,
            events: Vec::new(),
            buffered_actions: [0; NUM_HEROES],
        };
        game.init_map();
        game
    }

    pub fn reset(&mut self) {
        self.heroes.clear();
        self.towers.clear();
        self.creeps.clear();
        self.bases.clear();
        self.tick = 0;
        self.done = false;
        self.winner = None;
        self.events.clear();
        self.buffered_actions = [0; NUM_HEROES];
        self.init_map();
    }

    fn init_map(&mut self) {
        // Team A heroes spawn near base (cols 2-3)
        self.heroes.push(Hero::new(0, Team::A, 2, 6));
        self.heroes.push(Hero::new(1, Team::A, 2, 7));
        self.heroes.push(Hero::new(2, Team::A, 2, 8));

        // Team B heroes spawn near base (cols 29-30)
        self.heroes.push(Hero::new(3, Team::B, 29, 6));
        self.heroes.push(Hero::new(4, Team::B, 29, 7));
        self.heroes.push(Hero::new(5, Team::B, 29, 8));

        // Towers: 2 per team
        self.towers.push(Tower::new(Team::A, 5, 7));   // A outer
        self.towers.push(Tower::new(Team::A, 9, 7));    // A inner
        self.towers.push(Tower::new(Team::B, 22, 7));   // B inner
        self.towers.push(Tower::new(Team::B, 26, 7));   // B outer

        // Bases
        self.bases.push(Base::new(Team::A, 0, 1, 7));
        self.bases.push(Base::new(Team::B, 30, 31, 7));
    }

    fn spawn_x(&self, team: Team) -> usize {
        match team {
            Team::A => 2,
            Team::B => 29,
        }
    }

    fn spawn_y(&self) -> usize {
        7
    }

    /// Buffer an action for a hero to be resolved at next tick().
    pub fn apply_hero_action(&mut self, hero_id: usize, action: usize) {
        assert!(hero_id < NUM_HEROES);
        assert!(action < NUM_ACTIONS);
        self.buffered_actions[hero_id] = action;
    }

    /// Pick a scripted action for an enemy hero (simple heuristic).
    pub fn scripted_action(&mut self, hero_id: usize) -> usize {
        let hero = &self.heroes[hero_id];
        if !hero.alive {
            return 0; // stay + no attack
        }
        let team = hero.team;
        let hy = hero.y;

        // Move towards lane, then march towards enemy base
        let move_dir = match team {
            Team::A => {
                if hy != LANE_Y && hy != LANE_Y + 1 {
                    if hy < LANE_Y { 2 } else { 1 }
                } else {
                    4 // move right towards enemy
                }
            }
            Team::B => {
                if hy != LANE_Y && hy != LANE_Y + 1 {
                    if hy < LANE_Y { 2 } else { 1 }
                } else {
                    3 // move left towards enemy
                }
            }
        };

        // Attack: prefer nearest enemy creep, then hero, then tower
        let mut attack_type = 0; // no attack

        // Check if any enemy creep in range
        let has_enemy_creep = self.creeps.iter().any(|cr| {
            cr.alive && cr.team != team && hero.dist_to(cr.x, cr.y) <= 2
        });
        if has_enemy_creep {
            attack_type = 2; // attack nearest creep
        }

        // Check if any enemy hero in range
        let has_enemy_hero = self.heroes.iter().any(|h| {
            h.alive && h.team != team && hero.dist_to(h.x, h.y) <= 2
        });
        if has_enemy_hero {
            attack_type = 1; // attack nearest hero (more aggressive)
        }

        move_dir * 4 + attack_type
    }

    /// Execute one game tick.
    pub fn tick(&mut self) {
        if self.done {
            return;
        }
        self.events.clear();
        self.tick += 1;

        // 1. Spawn creep waves
        if self.tick % CREEP_WAVE_INTERVAL == 0 {
            self.spawn_creep_wave();
        }

        // 2. Tower attacks (prioritize creeps, then nearest hero)
        self.resolve_tower_attacks();

        // 3. Creep AI and attacks
        self.resolve_creep_actions();

        // 4. Hero actions
        self.resolve_hero_actions();

        // 5. Hero respawn
        for i in 0..self.heroes.len() {
            let team = self.heroes[i].team;
            let sx = self.spawn_x(team);
            let sy = self.spawn_y();
            self.heroes[i].respawn_tick(sx, sy);
        }

        // 6. Remove dead creeps
        self.creeps.retain(|c| c.alive);

        // 7. Check win condition
        self.check_done();

        // Reset buffered actions
        self.buffered_actions = [0; NUM_HEROES];
    }

    fn spawn_creep_wave(&mut self) {
        for i in 0..CREEPS_PER_WAVE {
            self.creeps.push(Creep::new(Team::A, 3, LANE_Y + (i % 2)));
            self.creeps.push(Creep::new(Team::B, 28, LANE_Y + (i % 2)));
        }
    }

    fn resolve_tower_attacks(&mut self) {
        for ti in 0..self.towers.len() {
            if !self.towers[ti].alive {
                continue;
            }
            let tx = self.towers[ti].x;
            let ty = self.towers[ti].y;
            let team = self.towers[ti].team;
            let atk = self.towers[ti].attack;
            let range = self.towers[ti].attack_range;

            // Find nearest enemy creep in range
            let mut target_creep = None;
            let mut best_dist = usize::MAX;
            for (ci, creep) in self.creeps.iter().enumerate() {
                if !creep.alive || creep.team == team {
                    continue;
                }
                let d = chebyshev(tx, ty, creep.x, creep.y);
                if d <= range && d < best_dist {
                    best_dist = d;
                    target_creep = Some(ci);
                }
            }

            if let Some(ci) = target_creep {
                self.creeps[ci].take_damage(atk);
                continue;
            }

            // No creep target, find nearest enemy hero
            let mut target_hero = None;
            best_dist = usize::MAX;
            for (hi, hero) in self.heroes.iter().enumerate() {
                if !hero.alive || hero.team == team {
                    continue;
                }
                let d = chebyshev(tx, ty, hero.x, hero.y);
                if d <= range && d < best_dist {
                    best_dist = d;
                    target_hero = Some(hi);
                }
            }

            if let Some(hi) = target_hero {
                let died = self.heroes[hi].take_damage(atk);
                if died {
                    self.events.push(GameEvent::HeroDeath { hero_id: hi });
                }
            }
        }
    }

    fn resolve_creep_actions(&mut self) {
        let n = self.creeps.len();
        // Creep AI: march forward, attack nearest enemy in range 1
        for ci in 0..n {
            if !self.creeps[ci].alive {
                continue;
            }
            let team = self.creeps[ci].team;
            let cx = self.creeps[ci].x;
            let cy = self.creeps[ci].y;
            let atk = self.creeps[ci].attack;

            // Attack nearest enemy creep or hero in range 1
            let mut attacked = false;

            // Attack enemy creep
            let mut target_creep = None;
            let mut best_dist = usize::MAX;
            for (oci, other) in self.creeps.iter().enumerate() {
                if oci == ci || !other.alive || other.team == team {
                    continue;
                }
                let d = chebyshev(cx, cy, other.x, other.y);
                if d <= 1 && d < best_dist {
                    best_dist = d;
                    target_creep = Some(oci);
                }
            }
            if let Some(oci) = target_creep {
                // We can't mutably borrow twice, so just mark damage
                let hp_before = self.creeps[oci].hp;
                self.creeps[oci].hp -= atk;
                if self.creeps[oci].hp <= 0.0 {
                    self.creeps[oci].hp = 0.0;
                    self.creeps[oci].alive = false;
                }
                attacked = true;
                let _ = hp_before; // suppress warning
            }

            if !attacked {
                // Attack enemy hero in range 1
                for hi in 0..self.heroes.len() {
                    if !self.heroes[hi].alive || self.heroes[hi].team == team {
                        continue;
                    }
                    let d = chebyshev(cx, cy, self.heroes[hi].x, self.heroes[hi].y);
                    if d <= 1 {
                        let died = self.heroes[hi].take_damage(atk);
                        if died {
                            self.events.push(GameEvent::HeroDeath { hero_id: hi });
                        }
                        attacked = true;
                        break;
                    }
                }
            }

            // Attack base if adjacent
            if !attacked {
                for bi in 0..self.bases.len() {
                    if !self.bases[bi].alive || self.bases[bi].team == team {
                        continue;
                    }
                    let bx = (self.bases[bi].x_min + self.bases[bi].x_max) / 2;
                    let by = self.bases[bi].y;
                    if chebyshev(cx, cy, bx, by) <= 1 {
                        self.bases[bi].take_damage(atk);
                        attacked = true;
                        break;
                    }
                }
            }

            // If didn't attack, move forward
            if !attacked {
                match team {
                    Team::A => {
                        if cx < MAP_W - 1 {
                            self.creeps[ci].x += 1;
                        }
                    }
                    Team::B => {
                        if cx > 0 {
                            self.creeps[ci].x -= 1;
                        }
                    }
                }
            }
        }
    }

    fn resolve_hero_actions(&mut self) {
        // Decode and apply all hero actions simultaneously.
        // First pass: compute movements. Second pass: compute attacks.
        let mut new_positions = Vec::with_capacity(NUM_HEROES);
        for i in 0..NUM_HEROES {
            let hero = &self.heroes[i];
            if !hero.alive {
                new_positions.push((hero.x, hero.y));
                continue;
            }
            let action = self.buffered_actions[i];
            let move_dir = action / 4;
            let (mut nx, mut ny) = (hero.x, hero.y);
            match move_dir {
                0 => {} // stay
                1 => ny = ny.saturating_sub(1), // up
                2 => ny = (ny + 1).min(MAP_H - 1), // down
                3 => nx = nx.saturating_sub(1), // left
                4 => nx = (nx + 1).min(MAP_W - 1), // right
                _ => {}
            }
            new_positions.push((nx, ny));
        }

        // Apply movements
        for i in 0..NUM_HEROES {
            self.heroes[i].x = new_positions[i].0;
            self.heroes[i].y = new_positions[i].1;
        }

        // Resolve attacks
        for i in 0..NUM_HEROES {
            if !self.heroes[i].alive {
                continue;
            }
            let action = self.buffered_actions[i];
            let attack_type = action % 4;
            let hero_x = self.heroes[i].x;
            let hero_y = self.heroes[i].y;
            let hero_team = self.heroes[i].team;
            let hero_atk = self.heroes[i].attack;
            let hero_range = self.heroes[i].attack_range;

            match attack_type {
                0 => {} // no attack
                1 => {
                    // Attack nearest enemy hero
                    let mut target = None;
                    let mut best = usize::MAX;
                    for j in 0..NUM_HEROES {
                        if j == i || !self.heroes[j].alive || self.heroes[j].team == hero_team {
                            continue;
                        }
                        let d = chebyshev(hero_x, hero_y, self.heroes[j].x, self.heroes[j].y);
                        if d <= hero_range + 1 && d < best {
                            best = d;
                            target = Some(j);
                        }
                    }
                    if let Some(j) = target {
                        let died = self.heroes[j].take_damage(hero_atk);
                        self.events.push(GameEvent::DamageDealt {
                            hero_id: i,
                            amount: hero_atk,
                        });
                        if died {
                            self.events.push(GameEvent::HeroKill {
                                killer_id: i,
                                victim_id: j,
                            });
                            self.events.push(GameEvent::HeroDeath { hero_id: j });
                        }
                    }
                }
                2 => {
                    // Attack nearest enemy creep
                    let mut target = None;
                    let mut best = usize::MAX;
                    for (ci, creep) in self.creeps.iter().enumerate() {
                        if !creep.alive || creep.team == hero_team {
                            continue;
                        }
                        let d = chebyshev(hero_x, hero_y, creep.x, creep.y);
                        if d <= hero_range + 1 && d < best {
                            best = d;
                            target = Some(ci);
                        }
                    }
                    if let Some(ci) = target {
                        let killed = self.creeps[ci].take_damage(hero_atk);
                        self.events.push(GameEvent::DamageDealt {
                            hero_id: i,
                            amount: hero_atk,
                        });
                        if killed {
                            self.events.push(GameEvent::CreepKill { hero_id: i });
                        }
                    }
                }
                3 => {
                    // Attack nearest enemy tower
                    let mut target = None;
                    let mut best = usize::MAX;
                    for (ti, tower) in self.towers.iter().enumerate() {
                        if !tower.alive || tower.team == hero_team {
                            continue;
                        }
                        let d = chebyshev(hero_x, hero_y, tower.x, tower.y);
                        if d <= hero_range + 1 && d < best {
                            best = d;
                            target = Some(ti);
                        }
                    }
                    if let Some(ti) = target {
                        let destroyed = self.towers[ti].take_damage(hero_atk);
                        self.events.push(GameEvent::DamageDealt {
                            hero_id: i,
                            amount: hero_atk,
                        });
                        if destroyed {
                            self.events.push(GameEvent::TowerDestroyed {
                                team: self.towers[ti].team,
                                by_hero: Some(i),
                            });
                        }
                    }

                    // Also try to attack base if adjacent
                    if target.is_none() {
                        for bi in 0..self.bases.len() {
                            if !self.bases[bi].alive || self.bases[bi].team == hero_team {
                                continue;
                            }
                            let bx = (self.bases[bi].x_min + self.bases[bi].x_max) / 2;
                            let by = self.bases[bi].y;
                            if chebyshev(hero_x, hero_y, bx, by) <= hero_range + 1 {
                                let destroyed = self.bases[bi].take_damage(hero_atk);
                                self.events.push(GameEvent::DamageDealt {
                                    hero_id: i,
                                    amount: hero_atk,
                                });
                                if destroyed {
                                    self.events.push(GameEvent::BaseDestroyed {
                                        team: self.bases[bi].team,
                                    });
                                }
                                break;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn check_done(&mut self) {
        if self.tick >= MAX_TICKS {
            self.done = true;
            // Winner is team with more base HP
            let a_hp = self.bases.iter().find(|b| b.team == Team::A).map_or(0.0, |b| b.hp);
            let b_hp = self.bases.iter().find(|b| b.team == Team::B).map_or(0.0, |b| b.hp);
            self.winner = if a_hp > b_hp {
                Some(Team::A)
            } else if b_hp > a_hp {
                Some(Team::B)
            } else {
                None // draw
            };
            return;
        }

        for base in &self.bases {
            if !base.alive {
                self.done = true;
                self.winner = Some(base.team.opponent());
                return;
            }
        }
    }

    /// Build observation vector for a given hero (167 floats).
    pub fn observe(&self, hero_id: usize) -> Vec<f32> {
        let hero = &self.heroes[hero_id];
        let team = hero.team;
        let mut obs = Vec::with_capacity(OBS_DIM);

        // Global (7)
        obs.push(self.tick as f32 / MAX_TICKS as f32);
        let a_base = self.bases.iter().find(|b| b.team == Team::A).unwrap();
        let b_base = self.bases.iter().find(|b| b.team == Team::B).unwrap();
        obs.push(a_base.hp / a_base.max_hp);
        obs.push(b_base.hp / b_base.max_hp);
        // Tower alive flags: A0, A1, B0, B1
        for t in &self.towers {
            obs.push(if t.alive { 1.0 } else { 0.0 });
        }

        // Self (5)
        obs.push(hero.x as f32 / MAP_W as f32);
        obs.push(hero.y as f32 / MAP_H as f32);
        obs.push(hero.hp / hero.max_hp);
        obs.push(if hero.alive { 1.0 } else { 0.0 });
        obs.push(match team { Team::A => 0.0, Team::B => 1.0 });

        // Allies (10): 2 allies x (dx/32, dy/16, hp/250, alive, respawn/10)
        let allies: Vec<usize> = self.heroes.iter()
            .enumerate()
            .filter(|(i, h)| *i != hero_id && h.team == team)
            .map(|(i, _)| i)
            .collect();
        for &ai in allies.iter().take(2) {
            let a = &self.heroes[ai];
            obs.push((a.x as f32 - hero.x as f32) / MAP_W as f32);
            obs.push((a.y as f32 - hero.y as f32) / MAP_H as f32);
            obs.push(a.hp / a.max_hp);
            obs.push(if a.alive { 1.0 } else { 0.0 });
            obs.push(a.respawn_timer as f32 / 10.0);
        }
        // Pad if fewer than 2 allies
        for _ in allies.len()..2 {
            obs.extend_from_slice(&[0.0; 5]);
        }

        // Enemies (15): 3 enemies x (dx/32, dy/16, hp/250, alive, visible)
        let enemies: Vec<usize> = self.heroes.iter()
            .enumerate()
            .filter(|(_, h)| h.team != team)
            .map(|(i, _)| i)
            .collect();
        for &ei in enemies.iter().take(3) {
            let e = &self.heroes[ei];
            obs.push((e.x as f32 - hero.x as f32) / MAP_W as f32);
            obs.push((e.y as f32 - hero.y as f32) / MAP_H as f32);
            obs.push(e.hp / e.max_hp);
            obs.push(if e.alive { 1.0 } else { 0.0 });
            obs.push(1.0); // always visible in this version
        }
        for _ in enemies.len()..3 {
            obs.extend_from_slice(&[0.0; 5]);
        }

        // Local grid 11x11 (121): centered on hero
        let view_r = 5i32;
        for dy in -view_r..=view_r {
            for dx in -view_r..=view_r {
                let gx = hero.x as i32 + dx;
                let gy = hero.y as i32 + dy;
                if gx < 0 || gx >= MAP_W as i32 || gy < 0 || gy >= MAP_H as i32 {
                    obs.push(0.0);
                    continue;
                }
                let gx = gx as usize;
                let gy = gy as usize;
                let mut val = 0.0f32;
                // Check towers
                for t in &self.towers {
                    if t.alive && t.x == gx && t.y == gy {
                        val = if t.team == team { 1.0 } else { -1.0 };
                    }
                }
                // Check creeps (overrides empty)
                for c in &self.creeps {
                    if c.alive && c.x == gx && c.y == gy {
                        val = if c.team == team { 0.5 } else { -0.5 };
                    }
                }
                obs.push(val);
            }
        }

        // Nearby creeps (9): top-3 nearest enemy creeps (dx, dy, hp/40)
        let mut enemy_creeps: Vec<(usize, usize, f32)> = self.creeps.iter()
            .filter(|c| c.alive && c.team != team)
            .map(|c| (c.x, c.y, c.hp))
            .collect();
        enemy_creeps.sort_by_key(|(cx, cy, _)| chebyshev(hero.x, hero.y, *cx, *cy));
        for i in 0..3 {
            if i < enemy_creeps.len() {
                let (cx, cy, chp) = enemy_creeps[i];
                obs.push((cx as f32 - hero.x as f32) / MAP_W as f32);
                obs.push((cy as f32 - hero.y as f32) / MAP_H as f32);
                obs.push(chp / 40.0);
            } else {
                obs.extend_from_slice(&[0.0; 3]);
            }
        }

        debug_assert_eq!(obs.len(), OBS_DIM, "Observation dim mismatch: {}", obs.len());
        obs
    }

    /// Take a snapshot of the current state for replay.
    pub fn snapshot(&self) -> FrameSnapshot {
        FrameSnapshot {
            tick: self.tick,
            heroes: self.heroes.iter().map(|h| {
                (h.x, h.y, h.hp, h.max_hp, h.alive, match h.team { Team::A => 0, Team::B => 1 })
            }).collect(),
            towers: self.towers.iter().map(|t| {
                (t.x, t.y, t.hp, t.max_hp, t.alive, match t.team { Team::A => 0, Team::B => 1 })
            }).collect(),
            creeps: self.creeps.iter().map(|c| {
                (c.x, c.y, c.hp, c.max_hp, c.alive, match c.team { Team::A => 0, Team::B => 1 })
            }).collect(),
            bases: self.bases.iter().map(|b| {
                (b.hp, b.max_hp, b.alive, match b.team { Team::A => 0, Team::B => 1 })
            }).collect(),
        }
    }
}

fn chebyshev(x1: usize, y1: usize, x2: usize, y2: usize) -> usize {
    let dx = if x1 > x2 { x1 - x2 } else { x2 - x1 };
    let dy = if y1 > y2 { y1 - y2 } else { y2 - y1 };
    dx.max(dy)
}
