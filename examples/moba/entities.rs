/// Game entities for the 3v3 single-lane MOBA.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Team {
    A,
    B,
}

impl Team {
    pub fn opponent(self) -> Team {
        match self {
            Team::A => Team::B,
            Team::B => Team::A,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Hero {
    pub id: usize,
    pub team: Team,
    pub x: usize,
    pub y: usize,
    pub hp: f32,
    pub max_hp: f32,
    pub attack: f32,
    pub attack_range: usize,
    pub alive: bool,
    pub respawn_timer: usize,
}

impl Hero {
    pub fn new(id: usize, team: Team, x: usize, y: usize) -> Self {
        Hero {
            id,
            team,
            x,
            y,
            hp: 250.0,
            max_hp: 250.0,
            attack: 20.0,
            attack_range: 1,
            alive: true,
            respawn_timer: 0,
        }
    }

    pub fn take_damage(&mut self, dmg: f32) -> bool {
        if !self.alive {
            return false;
        }
        self.hp -= dmg;
        if self.hp <= 0.0 {
            self.hp = 0.0;
            self.alive = false;
            self.respawn_timer = 10;
            true // died
        } else {
            false
        }
    }

    pub fn respawn_tick(&mut self, spawn_x: usize, spawn_y: usize) {
        if !self.alive {
            self.respawn_timer = self.respawn_timer.saturating_sub(1);
            if self.respawn_timer == 0 {
                self.alive = true;
                self.hp = self.max_hp;
                self.x = spawn_x;
                self.y = spawn_y;
            }
        }
    }

    pub fn dist_to(&self, ox: usize, oy: usize) -> usize {
        let dx = if self.x > ox { self.x - ox } else { ox - self.x };
        let dy = if self.y > oy { self.y - oy } else { oy - self.y };
        dx.max(dy) // Chebyshev distance
    }
}

#[derive(Clone, Debug)]
pub struct Tower {
    pub team: Team,
    pub x: usize,
    pub y: usize,
    pub hp: f32,
    pub max_hp: f32,
    pub attack: f32,
    pub attack_range: usize,
    pub alive: bool,
}

impl Tower {
    pub fn new(team: Team, x: usize, y: usize) -> Self {
        Tower {
            team,
            x,
            y,
            hp: 200.0,
            max_hp: 200.0,
            attack: 15.0,
            attack_range: 3,
            alive: true,
        }
    }

    pub fn take_damage(&mut self, dmg: f32) -> bool {
        if !self.alive {
            return false;
        }
        self.hp -= dmg;
        if self.hp <= 0.0 {
            self.hp = 0.0;
            self.alive = false;
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Debug)]
pub struct Creep {
    pub team: Team,
    pub x: usize,
    pub y: usize,
    pub hp: f32,
    pub max_hp: f32,
    pub attack: f32,
    pub alive: bool,
}

impl Creep {
    pub fn new(team: Team, x: usize, y: usize) -> Self {
        Creep {
            team,
            x,
            y,
            hp: 40.0,
            max_hp: 40.0,
            attack: 5.0,
            alive: true,
        }
    }

    pub fn take_damage(&mut self, dmg: f32) -> bool {
        if !self.alive {
            return false;
        }
        self.hp -= dmg;
        if self.hp <= 0.0 {
            self.hp = 0.0;
            self.alive = false;
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Debug)]
pub struct Base {
    pub team: Team,
    pub x_min: usize,
    pub x_max: usize,
    pub y: usize,
    pub hp: f32,
    pub max_hp: f32,
    pub alive: bool,
}

impl Base {
    pub fn new(team: Team, x_min: usize, x_max: usize, y: usize) -> Self {
        Base {
            team,
            x_min,
            x_max,
            y,
            hp: 500.0,
            max_hp: 500.0,
            alive: true,
        }
    }

    pub fn take_damage(&mut self, dmg: f32) -> bool {
        if !self.alive {
            return false;
        }
        self.hp -= dmg;
        if self.hp <= 0.0 {
            self.hp = 0.0;
            self.alive = false;
            true
        } else {
            false
        }
    }
}
