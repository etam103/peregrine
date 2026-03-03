/// HTML visualization for MOBA training and replay, plus MP4 video export.

use crate::game::FrameSnapshot;
use std::io::Write;

// ── Software rasterizer for video export ──────────────────────────────

const VID_W: usize = 800;
const VID_H: usize = 420;
const PAD: f32 = 20.0;
const GRID_W: f32 = 32.0;
const GRID_H: f32 = 16.0;
const CELL_W: f32 = (VID_W as f32 - 2.0 * PAD) / GRID_W; // 23.75
const CELL_H: f32 = (VID_H as f32 - 2.0 * PAD) / GRID_H; // 23.75

const FONT_W: usize = 5;
const FONT_H: usize = 7;

fn char_bitmap(ch: char) -> [u8; 7] {
    match ch {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b10001, 0b01110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'i' => [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
        'c' => [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110],
        'k' => [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        ' ' => [0; 7],
        _ => [0; 7],
    }
}

struct Canvas {
    buf: Vec<u8>,
}

impl Canvas {
    fn new() -> Self {
        Canvas { buf: vec![0; VID_W * VID_H * 3] }
    }

    fn set_pixel(&mut self, x: i32, y: i32, r: u8, g: u8, b: u8) {
        if x >= 0 && y >= 0 && (x as usize) < VID_W && (y as usize) < VID_H {
            let idx = ((y as usize) * VID_W + x as usize) * 3;
            self.buf[idx] = r;
            self.buf[idx + 1] = g;
            self.buf[idx + 2] = b;
        }
    }

    fn clear(&mut self, r: u8, g: u8, b: u8) {
        for i in (0..self.buf.len()).step_by(3) {
            self.buf[i] = r;
            self.buf[i + 1] = g;
            self.buf[i + 2] = b;
        }
    }

    fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32, r: u8, g: u8, b: u8) {
        let x0 = x.round() as i32;
        let y0 = y.round() as i32;
        let x1 = (x + w).round() as i32;
        let y1 = (y + h).round() as i32;
        for py in y0..y1 {
            for px in x0..x1 {
                self.set_pixel(px, py, r, g, b);
            }
        }
    }

    fn stroke_rect(&mut self, x: f32, y: f32, w: f32, h: f32, r: u8, g: u8, b: u8) {
        let x0 = x.round() as i32;
        let y0 = y.round() as i32;
        let x1 = (x + w).round() as i32;
        let y1 = (y + h).round() as i32;
        for px in x0..=x1 {
            self.set_pixel(px, y0, r, g, b);
            self.set_pixel(px, y1, r, g, b);
        }
        for py in y0..=y1 {
            self.set_pixel(x0, py, r, g, b);
            self.set_pixel(x1, py, r, g, b);
        }
    }

    fn fill_circle(&mut self, cx: f32, cy: f32, radius: f32, r: u8, g: u8, b: u8) {
        let x0 = (cx - radius).floor() as i32;
        let y0 = (cy - radius).floor() as i32;
        let x1 = (cx + radius).ceil() as i32;
        let y1 = (cy + radius).ceil() as i32;
        let r2 = radius * radius;
        for py in y0..=y1 {
            for px in x0..=x1 {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                if dx * dx + dy * dy <= r2 {
                    self.set_pixel(px, py, r, g, b);
                }
            }
        }
    }

    fn stroke_circle(&mut self, cx: f32, cy: f32, radius: f32, r: u8, g: u8, b: u8) {
        let steps = (radius * std::f32::consts::TAU).ceil() as i32;
        for i in 0..steps {
            let angle = i as f32 / steps as f32 * std::f32::consts::TAU;
            let px = (cx + radius * angle.cos()).round() as i32;
            let py = (cy + radius * angle.sin()).round() as i32;
            self.set_pixel(px, py, r, g, b);
        }
    }

    fn vline(&mut self, x: f32, y0: f32, y1: f32, r: u8, g: u8, b: u8) {
        let px = x.round() as i32;
        for py in (y0.round() as i32)..=(y1.round() as i32) {
            self.set_pixel(px, py, r, g, b);
        }
    }

    fn hline(&mut self, y: f32, x0: f32, x1: f32, r: u8, g: u8, b: u8) {
        let py = y.round() as i32;
        for px in (x0.round() as i32)..=(x1.round() as i32) {
            self.set_pixel(px, py, r, g, b);
        }
    }

    fn draw_char(&mut self, ch: char, x: i32, y: i32, scale: i32, r: u8, g: u8, b: u8) {
        let bitmap = char_bitmap(ch);
        for row in 0..FONT_H {
            for col in 0..FONT_W {
                if (bitmap[row] >> (FONT_W - 1 - col)) & 1 == 1 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            self.set_pixel(
                                x + col as i32 * scale + sx,
                                y + row as i32 * scale + sy,
                                r, g, b,
                            );
                        }
                    }
                }
            }
        }
    }

    fn draw_string(&mut self, s: &str, x: i32, y: i32, scale: i32, r: u8, g: u8, b: u8) {
        let char_w = (FONT_W as i32 + 1) * scale;
        for (i, ch) in s.chars().enumerate() {
            self.draw_char(ch, x + i as i32 * char_w, y, scale, r, g, b);
        }
    }
}

fn grid_cx(x: usize) -> f32 { PAD + x as f32 * CELL_W + CELL_W / 2.0 }
fn grid_cy(y: usize) -> f32 { PAD + y as f32 * CELL_H + CELL_H / 2.0 }

fn render_frame(canvas: &mut Canvas, frame: &FrameSnapshot) {
    // Background (#16213e)
    canvas.clear(0x16, 0x21, 0x3e);

    // Grid lines (#2a2a4a)
    for x in 0..=32 {
        canvas.vline(PAD + x as f32 * CELL_W, PAD, PAD + 16.0 * CELL_H, 0x2a, 0x2a, 0x4a);
    }
    for y in 0..=16 {
        canvas.hline(PAD + y as f32 * CELL_H, PAD, PAD + 32.0 * CELL_W, 0x2a, 0x2a, 0x4a);
    }

    // Lane highlight (rows 7-8) — rgba(255,255,255,0.04) blended on #16213e ≈ (31,42,70)
    canvas.fill_rect(PAD, PAD + 7.0 * CELL_H, 32.0 * CELL_W, 2.0 * CELL_H, 31, 42, 70);

    // Bases
    for b in &frame.bases {
        let (hp, max_hp, alive, team) = *b;
        if !alive { continue; }
        let bx = if team == 0 { 0.0 } else { 30.0 };
        let bw = 2.0;
        // Fill: alpha-blend team color at 0.3 over background
        let (fr, fg, fb) = if team == 0 { (39, 82, 118) } else { (87, 48, 67) };
        canvas.fill_rect(PAD + bx * CELL_W, PAD + 6.0 * CELL_H, bw * CELL_W, 4.0 * CELL_H, fr, fg, fb);
        let (sr, sg, sb) = if team == 0 { (0x4f, 0xc3, 0xf7) } else { (0xef, 0x53, 0x50) };
        canvas.stroke_rect(PAD + bx * CELL_W, PAD + 6.0 * CELL_H, bw * CELL_W, 4.0 * CELL_H, sr, sg, sb);
        // HP bar
        let bcx = PAD + (bx + 1.0) * CELL_W;
        let bcy = PAD + 6.0 * CELL_H - 4.0;
        let bar_w = bw * CELL_W * 0.8;
        canvas.fill_rect(bcx - bar_w / 2.0, bcy - 4.0, bar_w, 4.0, 0x33, 0x33, 0x33);
        let filled = bar_w * hp / max_hp;
        canvas.fill_rect(bcx - bar_w / 2.0, bcy - 4.0, filled, 4.0, sr, sg, sb);
    }

    // Towers
    for t in &frame.towers {
        let (tx, ty, hp, max_hp, alive, team) = *t;
        if !alive { continue; }
        let s = CELL_W * 0.7;
        let (tr, tg, tb) = if team == 0 { (0x4f, 0xc3, 0xf7) } else { (0xef, 0x53, 0x50) };
        canvas.fill_rect(grid_cx(tx) - s / 2.0, grid_cy(ty) - s / 2.0, s, s, tr, tg, tb);
        canvas.stroke_rect(grid_cx(tx) - s / 2.0, grid_cy(ty) - s / 2.0, s, s, 0xff, 0xff, 0xff);
        // HP bar
        let bar_w = CELL_W * 0.8;
        canvas.fill_rect(grid_cx(tx) - bar_w / 2.0, grid_cy(ty) - CELL_H / 2.0 - 6.0, bar_w, 3.0, 0x33, 0x33, 0x33);
        let filled = bar_w * hp / max_hp;
        canvas.fill_rect(grid_cx(tx) - bar_w / 2.0, grid_cy(ty) - CELL_H / 2.0 - 6.0, filled, 3.0, tr, tg, tb);
    }

    // Creeps
    for c in &frame.creeps {
        let (cx, cy, _hp, _max_hp, alive, team) = *c;
        if !alive { continue; }
        let (cr, cg, cb) = if team == 0 { (0x81, 0xd4, 0xfa) } else { (0xef, 0x9a, 0x9a) };
        canvas.fill_circle(grid_cx(cx), grid_cy(cy), CELL_W * 0.2, cr, cg, cb);
    }

    // Heroes
    let hero_labels = ['1', '2', '3', '4', '5', '6'];
    for (i, h) in frame.heroes.iter().enumerate() {
        let (hx, hy, hp, max_hp, alive, team) = *h;
        if !alive { continue; }
        let r = CELL_W * 0.4;
        let (hr, hg, hb) = if team == 0 { (0x1e, 0x88, 0xe5) } else { (0xe5, 0x39, 0x35) };
        canvas.fill_circle(grid_cx(hx), grid_cy(hy), r, hr, hg, hb);
        canvas.stroke_circle(grid_cx(hx), grid_cy(hy), r, 0xff, 0xff, 0xff);
        // Number label
        canvas.draw_char(
            hero_labels[i],
            grid_cx(hx) as i32 - FONT_W as i32 / 2,
            grid_cy(hy) as i32 - FONT_H as i32 / 2,
            1, 0xff, 0xff, 0xff,
        );
        // HP bar
        let bar_w = CELL_W * 0.9;
        canvas.fill_rect(grid_cx(hx) - bar_w / 2.0, grid_cy(hy) - r - 6.0, bar_w, 3.0, 0x33, 0x33, 0x33);
        let (hpr, hpg, hpb) = if hp / max_hp > 0.3 { (0x4c, 0xaf, 0x50) } else { (0xff, 0x57, 0x22) };
        let filled = bar_w * hp / max_hp;
        canvas.fill_rect(grid_cx(hx) - bar_w / 2.0, grid_cy(hy) - r - 6.0, filled, 3.0, hpr, hpg, hpb);
    }

    // Tick counter (top-right)
    let tick_str = format!("Tick:{}/500", frame.tick);
    let scale = 2i32;
    let text_w = tick_str.len() as i32 * (FONT_W as i32 + 1) * scale;
    canvas.draw_string(&tick_str, VID_W as i32 - text_w - 10, 4, scale, 0xe0, 0xe0, 0xe0);
}

/// Render frames to an H.264 MP4 via FFmpeg.
pub fn write_video(path: &str, frames: &[FrameSnapshot]) {
    use std::io::BufWriter;
    use std::process::{Command, Stdio};

    let mut child = Command::new("/opt/homebrew/bin/ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", &format!("{}x{}", VID_W, VID_H),
            "-r", "5",
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-movflags", "+faststart",
            path,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn ffmpeg — is /opt/homebrew/bin/ffmpeg installed?");

    {
        let stdin = child.stdin.take().unwrap();
        let mut writer = BufWriter::new(stdin);
        let mut canvas = Canvas::new();
        let total = frames.len();

        for (i, frame) in frames.iter().enumerate() {
            render_frame(&mut canvas, frame);
            writer.write_all(&canvas.buf).expect("Failed to write frame to ffmpeg");
            if (i + 1) % 50 == 0 || i + 1 == total {
                eprint!("\rRendering frame {}/{}", i + 1, total);
            }
        }
    } // stdin dropped here, closing the pipe

    let status = child.wait().expect("Failed to wait for ffmpeg");
    eprintln!();
    if status.success() {
        println!("Video saved to {}", path);
    } else {
        eprintln!("ffmpeg exited with error (status {})", status);
    }
}

/// Training metrics for learning curves.
pub struct TrainingMetrics {
    pub rewards: Vec<f32>,
    pub win_rates: Vec<f32>,
    pub pi_losses: Vec<f32>,
    pub vf_losses: Vec<f32>,
    pub entropies: Vec<f32>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        TrainingMetrics {
            rewards: Vec::new(),
            win_rates: Vec::new(),
            pi_losses: Vec::new(),
            vf_losses: Vec::new(),
            entropies: Vec::new(),
        }
    }
}

/// Write learning curves HTML using Chart.js.
pub fn write_learning_curves(path: &str, metrics: &TrainingMetrics) {
    let mut html = String::from(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MOBA 3v3 — Training Curves</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; margin: 0; padding: 20px; }
  h1 { text-align: center; color: #e94560; margin-bottom: 30px; }
  .chart-container { max-width: 900px; margin: 0 auto 40px auto; background: #16213e; border-radius: 12px; padding: 20px; }
  canvas { width: 100% !important; }
</style>
</head>
<body>
<h1>MOBA 3v3 — PPO Training with LSTM Self-Play</h1>
"#);

    let charts: Vec<(&str, &str, Vec<(&str, &str, &Vec<f32>)>)> = vec![
        ("Mean Episode Reward", "Iteration", vec![("Reward", "#4fc3f7", &metrics.rewards)]),
        ("Win Rate", "Iteration", vec![("Win Rate", "#53d8a8", &metrics.win_rates)]),
        ("Policy Loss", "Iteration", vec![("Pi Loss", "#e94560", &metrics.pi_losses)]),
        ("Value Loss", "Iteration", vec![("VF Loss", "#ffa726", &metrics.vf_losses)]),
        ("Entropy", "Iteration", vec![("Entropy", "#ab47bc", &metrics.entropies)]),
    ];

    for (i, (title, x_label, series)) in charts.iter().enumerate() {
        let canvas_id = format!("chart{}", i);
        html.push_str(&format!(
            "<div class=\"chart-container\"><canvas id=\"{}\"></canvas></div>\n",
            canvas_id
        ));
        html.push_str("<script>\n");

        let mut datasets = String::new();
        for (j, (name, color, data)) in series.iter().enumerate() {
            let data_str: Vec<String> = data.iter().map(|v| format!("{:.6}", v)).collect();
            if j > 0 { datasets.push(','); }
            datasets.push_str(&format!(
                "{{label:'{}',data:[{}],borderColor:'{}',backgroundColor:'transparent',borderWidth:2,pointRadius:0,tension:0.3}}",
                name, data_str.join(","), color
            ));
        }

        let n = series.first().map(|(_, _, d)| d.len()).unwrap_or(0);
        let labels: Vec<String> = (1..=n).map(|x| x.to_string()).collect();

        html.push_str(&format!(
            r#"new Chart(document.getElementById('{}'),{{type:'line',data:{{labels:[{}],datasets:[{}]}},options:{{responsive:true,plugins:{{title:{{display:true,text:'{}',color:'#e0e0e0',font:{{size:16}}}},legend:{{labels:{{color:'#e0e0e0'}}}}}},scales:{{x:{{title:{{display:true,text:'{}',color:'#e0e0e0'}},ticks:{{color:'#aaa'}},grid:{{color:'#2a2a4a'}}}},y:{{ticks:{{color:'#aaa'}},grid:{{color:'#2a2a4a'}}}}}}}}}});"#,
            canvas_id, labels.join(","), datasets, title, x_label,
        ));
        html.push_str("\n</script>\n");
    }

    html.push_str("</body>\n</html>\n");

    let mut f = std::fs::File::create(path).expect("Failed to create HTML file");
    f.write_all(html.as_bytes()).expect("Failed to write HTML");
    println!("Learning curves saved to {}", path);
}

/// Write game replay animation HTML with Canvas2D.
pub fn write_replay_animation(path: &str, frames: &[FrameSnapshot]) {
    // Serialize frames to JSON
    let mut frames_json = String::from("[");
    for (fi, frame) in frames.iter().enumerate() {
        if fi > 0 { frames_json.push(','); }
        frames_json.push('{');

        // tick
        frames_json.push_str(&format!("\"tick\":{},", frame.tick));

        // heroes: [[x,y,hp,maxHp,alive,team],...]
        frames_json.push_str("\"heroes\":[");
        for (hi, h) in frame.heroes.iter().enumerate() {
            if hi > 0 { frames_json.push(','); }
            frames_json.push_str(&format!(
                "[{},{},{:.1},{:.1},{},{}]",
                h.0, h.1, h.2, h.3, if h.4 { 1 } else { 0 }, h.5
            ));
        }
        frames_json.push_str("],");

        // towers
        frames_json.push_str("\"towers\":[");
        for (ti, t) in frame.towers.iter().enumerate() {
            if ti > 0 { frames_json.push(','); }
            frames_json.push_str(&format!(
                "[{},{},{:.1},{:.1},{},{}]",
                t.0, t.1, t.2, t.3, if t.4 { 1 } else { 0 }, t.5
            ));
        }
        frames_json.push_str("],");

        // creeps
        frames_json.push_str("\"creeps\":[");
        for (ci, c) in frame.creeps.iter().enumerate() {
            if ci > 0 { frames_json.push(','); }
            frames_json.push_str(&format!(
                "[{},{},{:.1},{:.1},{},{}]",
                c.0, c.1, c.2, c.3, if c.4 { 1 } else { 0 }, c.5
            ));
        }
        frames_json.push_str("],");

        // bases
        frames_json.push_str("\"bases\":[");
        for (bi, b) in frame.bases.iter().enumerate() {
            if bi > 0 { frames_json.push(','); }
            frames_json.push_str(&format!(
                "[{:.1},{:.1},{},{}]",
                b.0, b.1, if b.2 { 1 } else { 0 }, b.3
            ));
        }
        frames_json.push_str("]");

        frames_json.push('}');
    }
    frames_json.push(']');

    let html = format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MOBA 3v3 — Game Replay</title>
<style>
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
  h1 {{ color: #e94560; margin-bottom: 10px; }}
  canvas {{ background: #16213e; border-radius: 12px; }}
  .controls {{ margin-top: 16px; display: flex; gap: 12px; align-items: center; }}
  button {{ background: #0f3460; color: #e0e0e0; border: none; border-radius: 6px; padding: 8px 18px; cursor: pointer; font-size: 14px; }}
  button:hover {{ background: #e94560; }}
  button.active {{ background: #e94560; }}
  .info {{ margin-top: 12px; font-size: 14px; font-variant-numeric: tabular-nums; display: flex; gap: 24px; }}
  .panel {{ background: #16213e; border-radius: 8px; padding: 12px 16px; min-width: 200px; }}
  .panel h3 {{ margin: 0 0 8px 0; color: #e94560; font-size: 14px; }}
  .team-a {{ color: #4fc3f7; }}
  .team-b {{ color: #ef5350; }}
</style>
</head>
<body>
<h1>MOBA 3v3 — Game Replay</h1>
<canvas id="cv" width="800" height="420"></canvas>
<div class="controls">
  <button id="playBtn">Pause</button>
  <button id="restartBtn">Restart</button>
  <button data-speed="0.5">0.5x</button>
  <button class="active" data-speed="1" id="speed1">1x</button>
  <button data-speed="2">2x</button>
  <button data-speed="4">4x</button>
</div>
<div class="info">
  <div class="panel" id="infoPanel">
    <h3>Game State</h3>
    <div id="tickInfo">Tick 0 / 500</div>
  </div>
  <div class="panel">
    <h3 class="team-a">Team Blue</h3>
    <div id="teamA">-</div>
  </div>
  <div class="panel">
    <h3 class="team-b">Team Red</h3>
    <div id="teamB">-</div>
  </div>
</div>
<script>
const frames = {frames_json};
const totalFrames = frames.length;
const W = 32, H = 16;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const pad = 20;
const cellW = (cv.width - 2 * pad) / W;
const cellH = (cv.height - 2 * pad) / H;

let frameIdx = 0;
let playing = true;
let speed = 1;
let timer = null;

function cx(x) {{ return pad + x * cellW + cellW / 2; }}
function cy(y) {{ return pad + y * cellH + cellH / 2; }}

function draw() {{
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (frameIdx >= totalFrames) return;
  const f = frames[frameIdx];

  // Draw grid lines
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= W; x++) {{
    ctx.beginPath();
    ctx.moveTo(pad + x * cellW, pad);
    ctx.lineTo(pad + x * cellW, pad + H * cellH);
    ctx.stroke();
  }}
  for (let y = 0; y <= H; y++) {{
    ctx.beginPath();
    ctx.moveTo(pad, pad + y * cellH);
    ctx.lineTo(pad + W * cellW, pad + y * cellH);
    ctx.stroke();
  }}

  // Draw lane highlight (rows 7-8)
  ctx.fillStyle = 'rgba(255,255,255,0.04)';
  ctx.fillRect(pad, pad + 7 * cellH, W * cellW, 2 * cellH);

  // Draw bases
  for (const b of f.bases) {{
    const [hp, maxHp, alive, team] = b;
    if (!alive) continue;
    const bx = team === 0 ? 0 : 30;
    const bw = 2;
    ctx.fillStyle = team === 0 ? 'rgba(79,195,247,0.3)' : 'rgba(239,83,80,0.3)';
    ctx.fillRect(pad + bx * cellW, pad + 6 * cellH, bw * cellW, 4 * cellH);
    ctx.strokeStyle = team === 0 ? '#4fc3f7' : '#ef5350';
    ctx.lineWidth = 2;
    ctx.strokeRect(pad + bx * cellW, pad + 6 * cellH, bw * cellW, 4 * cellH);
    // HP bar
    const bCx = pad + (bx + 1) * cellW;
    const bCy = pad + 6 * cellH - 4;
    const bW = bw * cellW * 0.8;
    ctx.fillStyle = '#333';
    ctx.fillRect(bCx - bW / 2, bCy - 4, bW, 4);
    ctx.fillStyle = team === 0 ? '#4fc3f7' : '#ef5350';
    ctx.fillRect(bCx - bW / 2, bCy - 4, bW * hp / maxHp, 4);
  }}

  // Draw towers
  for (const t of f.towers) {{
    const [tx, ty, hp, maxHp, alive, team] = t;
    if (!alive) continue;
    const s = cellW * 0.7;
    ctx.fillStyle = team === 0 ? '#4fc3f7' : '#ef5350';
    ctx.fillRect(cx(tx) - s / 2, cy(ty) - s / 2, s, s);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.strokeRect(cx(tx) - s / 2, cy(ty) - s / 2, s, s);
    // HP bar
    const bW = cellW * 0.8;
    ctx.fillStyle = '#333';
    ctx.fillRect(cx(tx) - bW / 2, cy(ty) - cellH / 2 - 6, bW, 3);
    ctx.fillStyle = team === 0 ? '#4fc3f7' : '#ef5350';
    ctx.fillRect(cx(tx) - bW / 2, cy(ty) - cellH / 2 - 6, bW * hp / maxHp, 3);
  }}

  // Draw creeps
  for (const c of f.creeps) {{
    const [cx_, cy_, hp, maxHp, alive, team] = c;
    if (!alive) continue;
    ctx.beginPath();
    ctx.arc(cx(cx_), cy(cy_), cellW * 0.2, 0, Math.PI * 2);
    ctx.fillStyle = team === 0 ? '#81d4fa' : '#ef9a9a';
    ctx.fill();
  }}

  // Draw heroes
  const heroLabels = ['1', '2', '3', '4', '5', '6'];
  for (let i = 0; i < f.heroes.length; i++) {{
    const [hx, hy, hp, maxHp, alive, team] = f.heroes[i];
    if (!alive) continue;
    const r = cellW * 0.4;
    ctx.beginPath();
    ctx.arc(cx(hx), cy(hy), r, 0, Math.PI * 2);
    ctx.fillStyle = team === 0 ? '#1e88e5' : '#e53935';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // Label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold ' + (cellW * 0.35) + 'px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(heroLabels[i], cx(hx), cy(hy));
    // HP bar
    const bW = cellW * 0.9;
    ctx.fillStyle = '#333';
    ctx.fillRect(cx(hx) - bW / 2, cy(hy) - r - 6, bW, 3);
    ctx.fillStyle = hp / maxHp > 0.3 ? '#4caf50' : '#ff5722';
    ctx.fillRect(cx(hx) - bW / 2, cy(hy) - r - 6, bW * hp / maxHp, 3);
  }}

  // Update info
  document.getElementById('tickInfo').textContent = 'Tick ' + f.tick + ' / 500';
  let aInfo = '';
  let bInfo = '';
  for (let i = 0; i < f.heroes.length; i++) {{
    const [hx, hy, hp, maxHp, alive, team] = f.heroes[i];
    const line = 'Hero ' + heroLabels[i] + ': ' + (alive ? Math.round(hp) + '/' + Math.round(maxHp) + ' HP' : 'DEAD') + '<br>';
    if (team === 0) aInfo += line; else bInfo += line;
  }}
  for (const b of f.bases) {{
    const [hp, maxHp, alive, team] = b;
    const line = 'Base: ' + (alive ? Math.round(hp) + '/' + Math.round(maxHp) + ' HP' : 'DESTROYED') + '<br>';
    if (team === 0) aInfo += line; else bInfo += line;
  }}
  document.getElementById('teamA').innerHTML = aInfo;
  document.getElementById('teamB').innerHTML = bInfo;
}}

function advance() {{
  if (frameIdx < totalFrames - 1) {{
    frameIdx++;
    draw();
  }} else {{
    playing = false;
    playBtn.textContent = 'Play';
    clearInterval(timer);
  }}
}}

function startTimer() {{
  clearInterval(timer);
  timer = setInterval(advance, 200 / speed);
}}

const playBtn = document.getElementById('playBtn');
playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  if (playing) startTimer(); else clearInterval(timer);
}});

document.getElementById('restartBtn').addEventListener('click', () => {{
  frameIdx = 0;
  draw();
  if (!playing) {{ playing = true; playBtn.textContent = 'Pause'; }}
  startTimer();
}});

document.querySelectorAll('[data-speed]').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('[data-speed]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    speed = parseFloat(btn.dataset.speed);
    if (playing) startTimer();
  }});
}});

draw();
startTimer();
</script>
</body>
</html>"##, frames_json = frames_json);

    let mut f = std::fs::File::create(path).expect("Failed to create animation HTML");
    f.write_all(html.as_bytes()).expect("Failed to write animation HTML");
    println!("Game replay saved to {}", path);
}
