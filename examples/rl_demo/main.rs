//! RL demonstration: PPO on CartPole, DQN on GridWorld, REINFORCE on BasicArithmetic.
//!
//! Usage:
//!   cargo run --example rl_demo --release               # PPO on CartPole
//!   cargo run --example rl_demo --release -- gridworld   # DQN on GridWorld
//!   cargo run --example rl_demo --release -- arithmetic  # REINFORCE on BasicArithmetic

use peregrine::envs::{BasicArithmetic, CartPole, GridWorld};
use peregrine::nn::Linear;
use peregrine::optim::Adam;
use peregrine::random;
use peregrine::rl::{
    copy_params, DqnConfig, DqnTrainer, Environment, PpoConfig, PpoTrainer, ReasoningEnv,
    Reinforce,
};
use peregrine::tensor::Tensor;
use std::io::Write;

// ============================================================================
// Chart HTML generation
// ============================================================================

struct Series {
    name: &'static str,
    color: &'static str,
    data: Vec<f32>,
}

struct ChartSpec {
    title: &'static str,
    x_label: &'static str,
    series: Vec<Series>,
}

fn write_html(path: &str, title: &str, charts: &[ChartSpec]) {
    let mut html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; margin: 0; padding: 20px; }}
  h1 {{ text-align: center; color: #e94560; margin-bottom: 30px; }}
  .chart-container {{ max-width: 900px; margin: 0 auto 40px auto; background: #16213e; border-radius: 12px; padding: 20px; }}
  canvas {{ width: 100% !important; }}
</style>
</head>
<body>
<h1>{title}</h1>
"#
    );

    for (i, chart) in charts.iter().enumerate() {
        let canvas_id = format!("chart{}", i);
        html.push_str(&format!(
            "<div class=\"chart-container\"><canvas id=\"{}\"></canvas></div>\n",
            canvas_id
        ));
        html.push_str("<script>\n");

        // Build datasets
        let mut datasets = String::new();
        for (j, s) in chart.series.iter().enumerate() {
            let data_str: Vec<String> = s.data.iter().map(|v| format!("{:.6}", v)).collect();
            if j > 0 {
                datasets.push(',');
            }
            datasets.push_str(&format!(
                "{{label:'{}',data:[{}],borderColor:'{}',backgroundColor:'transparent',borderWidth:2,pointRadius:0,tension:0.3}}",
                s.name,
                data_str.join(","),
                s.color
            ));
        }

        // X-axis labels (1-indexed)
        let n = chart.series.first().map(|s| s.data.len()).unwrap_or(0);
        let labels: Vec<String> = (1..=n).map(|x| x.to_string()).collect();

        html.push_str(&format!(
            r#"new Chart(document.getElementById('{}'),{{type:'line',data:{{labels:[{}],datasets:[{}]}},options:{{responsive:true,plugins:{{title:{{display:true,text:'{}',color:'#e0e0e0',font:{{size:16}}}},legend:{{labels:{{color:'#e0e0e0'}}}}}},scales:{{x:{{title:{{display:true,text:'{}',color:'#e0e0e0'}},ticks:{{color:'#aaa'}},grid:{{color:'#2a2a4a'}}}},y:{{ticks:{{color:'#aaa'}},grid:{{color:'#2a2a4a'}}}}}}}}}});"#,
            canvas_id,
            labels.join(","),
            datasets,
            chart.title,
            chart.x_label,
        ));
        html.push_str("\n</script>\n");
    }

    html.push_str("</body>\n</html>\n");

    let mut f = std::fs::File::create(path).expect("Failed to create HTML file");
    f.write_all(html.as_bytes())
        .expect("Failed to write HTML file");
    println!("Learning curves saved to {}", path);
}

// ============================================================================
// CartPole animation HTML generation
// ============================================================================

fn write_cartpole_animation(path: &str, frames: &[[f32; 2]]) {
    let frame_data: Vec<String> = frames
        .iter()
        .map(|f| format!("[{:.6},{:.6}]", f[0], f[1]))
        .collect();

    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CartPole Animation</title>
<style>
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
  h1 {{ color: #e94560; margin-bottom: 10px; }}
  canvas {{ background: #16213e; border-radius: 12px; }}
  .controls {{ margin-top: 16px; display: flex; gap: 12px; align-items: center; }}
  button {{ background: #0f3460; color: #e0e0e0; border: none; border-radius: 6px; padding: 8px 18px; cursor: pointer; font-size: 14px; }}
  button:hover {{ background: #e94560; }}
  button.active {{ background: #e94560; }}
  .info {{ margin-top: 12px; font-size: 16px; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<h1>CartPole PPO — Evaluation Episode</h1>
<canvas id="cv" width="700" height="400"></canvas>
<div class="controls">
  <button id="playBtn">Pause</button>
  <button id="restartBtn">Restart</button>
  <button class="active" data-speed="0.5">0.5x</button>
  <button class="active" data-speed="1" id="speed1">1x</button>
  <button class="active" data-speed="2">2x</button>
</div>
<div class="info" id="info">Step 0 / 0 &nbsp; Reward: 0</div>
<script>
const frames = [{}];
const totalFrames = frames.length;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const info = document.getElementById('info');
const playBtn = document.getElementById('playBtn');
const restartBtn = document.getElementById('restartBtn');

let frameIdx = 0;
let playing = true;
let speed = 1;
let timer = null;

// Physical constants (matching src/envs.rs)
const trackHalf = 2.4;
const poleLen = 1.0;      // 2 * half_length(0.5)
const cartW = 0.6;
const cartH = 0.3;

// Coordinate mapping: world units -> canvas pixels
const W = cv.width;
const H = cv.height;
const trackY = H * 0.72;
const scaleX = (W - 80) / (trackHalf * 2);  // pixels per world unit (x)
const scaleU = scaleX;                        // same scale for pole length
const cx = W / 2;                             // canvas centre x

function worldX(x) {{ return cx + x * scaleX; }}

function draw() {{
  const [x, theta] = frames[frameIdx];
  ctx.clearRect(0, 0, W, H);

  // Track
  ctx.strokeStyle = '#4a4a6a';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(worldX(-trackHalf), trackY);
  ctx.lineTo(worldX(trackHalf), trackY);
  ctx.stroke();

  // Track limits
  ctx.strokeStyle = '#e94560';
  ctx.lineWidth = 2;
  for (const lim of [-trackHalf, trackHalf]) {{
    const lx = worldX(lim);
    ctx.beginPath();
    ctx.moveTo(lx, trackY - 18);
    ctx.lineTo(lx, trackY + 18);
    ctx.stroke();
  }}

  // Cart
  const cartPxW = cartW * scaleX;
  const cartPxH = cartH * scaleX;
  const cartCx = worldX(x);
  const cartTop = trackY - cartPxH;
  ctx.fillStyle = '#0f3460';
  ctx.strokeStyle = '#537ec5';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(cartCx - cartPxW / 2, cartTop, cartPxW, cartPxH, 4);
  ctx.fill();
  ctx.stroke();

  // Wheels
  ctx.fillStyle = '#537ec5';
  const wheelR = 6;
  for (const dx of [-cartPxW * 0.3, cartPxW * 0.3]) {{
    ctx.beginPath();
    ctx.arc(cartCx + dx, trackY + 2, wheelR, 0, Math.PI * 2);
    ctx.fill();
  }}

  // Pole: theta=0 is upright, positive = clockwise
  const polePxLen = poleLen * scaleU;
  const poleBaseX = cartCx;
  const poleBaseY = cartTop;
  const poleTipX = poleBaseX + Math.sin(theta) * polePxLen;
  const poleTipY = poleBaseY - Math.cos(theta) * polePxLen;
  ctx.strokeStyle = '#e94560';
  ctx.lineWidth = 6;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(poleBaseX, poleBaseY);
  ctx.lineTo(poleTipX, poleTipY);
  ctx.stroke();

  // Pole tip dot
  ctx.fillStyle = '#e94560';
  ctx.beginPath();
  ctx.arc(poleTipX, poleTipY, 5, 0, Math.PI * 2);
  ctx.fill();

  // Pivot dot
  ctx.fillStyle = '#e0e0e0';
  ctx.beginPath();
  ctx.arc(poleBaseX, poleBaseY, 4, 0, Math.PI * 2);
  ctx.fill();

  info.textContent = 'Step ' + (frameIdx + 1) + ' / ' + totalFrames + '   Reward: ' + (frameIdx + 1);
}}

function tick() {{
  draw();
  frameIdx++;
  if (frameIdx >= totalFrames) {{
    frameIdx = totalFrames - 1;
    playing = false;
    playBtn.textContent = 'Play';
    clearInterval(timer);
    timer = null;
  }}
}}

function startTimer() {{
  if (timer) clearInterval(timer);
  timer = setInterval(tick, 20 / speed);
}}

playBtn.addEventListener('click', () => {{
  if (playing) {{
    playing = false;
    playBtn.textContent = 'Play';
    clearInterval(timer);
    timer = null;
  }} else {{
    if (frameIdx >= totalFrames - 1) frameIdx = 0;
    playing = true;
    playBtn.textContent = 'Pause';
    startTimer();
  }}
}});

restartBtn.addEventListener('click', () => {{
  frameIdx = 0;
  draw();
  if (!playing) {{
    playing = true;
    playBtn.textContent = 'Pause';
  }}
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

// Init: highlight 1x, start playing
document.querySelectorAll('[data-speed]').forEach(b => b.classList.remove('active'));
document.getElementById('speed1').classList.add('active');
draw();
startTimer();
</script>
</body>
</html>
"##,
        frame_data.join(",")
    );

    let mut f = std::fs::File::create(path).expect("Failed to create animation HTML file");
    f.write_all(html.as_bytes())
        .expect("Failed to write animation HTML file");
    println!("CartPole animation saved to {}", path);
}

// ============================================================================
// GridWorld animation HTML generation
// ============================================================================

struct GridWorldStep {
    row: usize,
    col: usize,
    action: usize,
    reward: f32,
}

fn write_gridworld_animation(
    path: &str,
    grid_size: usize,
    obstacles: &[(usize, usize)],
    goal: (usize, usize),
    steps: &[GridWorldStep],
) {
    // Build JSON arrays for JS
    let obs_json: Vec<String> = obstacles
        .iter()
        .map(|(r, c)| format!("[{},{}]", r, c))
        .collect();
    let steps_json: Vec<String> = steps
        .iter()
        .map(|s| format!("[{},{},{},{:.2}]", s.row, s.col, s.action, s.reward))
        .collect();

    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GridWorld DQN Animation</title>
<style>
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
  h1 {{ color: #e94560; margin-bottom: 10px; }}
  canvas {{ background: #16213e; border-radius: 12px; }}
  .controls {{ margin-top: 16px; display: flex; gap: 12px; align-items: center; }}
  button {{ background: #0f3460; color: #e0e0e0; border: none; border-radius: 6px; padding: 8px 18px; cursor: pointer; font-size: 14px; }}
  button:hover {{ background: #e94560; }}
  button.active {{ background: #e94560; }}
  .info {{ margin-top: 12px; font-size: 16px; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<h1>GridWorld DQN — Evaluation Episode</h1>
<canvas id="cv" width="500" height="500"></canvas>
<div class="controls">
  <button id="playBtn">Pause</button>
  <button id="restartBtn">Restart</button>
  <button data-speed="0.5">0.5x</button>
  <button class="active" data-speed="1" id="speed1">1x</button>
  <button data-speed="2">2x</button>
</div>
<div class="info" id="info">Step 0 / 0</div>
<script>
const gridSize = {grid_size};
const obstacles = [{obstacles_js}];
const goal = [{goal_r},{goal_c}];
const steps = [{steps_js}];
const totalSteps = steps.length;

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const info = document.getElementById('info');
const playBtn = document.getElementById('playBtn');
const restartBtn = document.getElementById('restartBtn');

let stepIdx = 0;
let playing = true;
let speed = 1;
let timer = null;

const pad = 40;
const cellSize = (cv.width - 2 * pad) / gridSize;
const dirLabels = ['\u2191 Up', '\u2193 Down', '\u2190 Left', '\u2192 Right'];
const visited = new Set();

function cellX(c) {{ return pad + c * cellSize; }}
function cellY(r) {{ return pad + r * cellSize; }}

function draw() {{
  ctx.clearRect(0, 0, cv.width, cv.height);

  // Draw grid cells
  for (let r = 0; r < gridSize; r++) {{
    for (let c = 0; c < gridSize; c++) {{
      const x = cellX(c);
      const y = cellY(r);
      ctx.fillStyle = '#1e2a4a';
      ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
    }}
  }}

  // Draw obstacles
  for (const [r, c] of obstacles) {{
    ctx.fillStyle = '#e94560';
    ctx.fillRect(cellX(c) + 1, cellY(r) + 1, cellSize - 2, cellSize - 2);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold ' + (cellSize * 0.4) + 'px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('#', cellX(c) + cellSize / 2, cellY(r) + cellSize / 2);
  }}

  // Draw goal
  ctx.fillStyle = '#53d8a8';
  ctx.fillRect(cellX(goal[1]) + 1, cellY(goal[0]) + 1, cellSize - 2, cellSize - 2);
  ctx.fillStyle = '#1a1a2e';
  ctx.font = 'bold ' + (cellSize * 0.4) + 'px system-ui';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('G', cellX(goal[1]) + cellSize / 2, cellY(goal[0]) + cellSize / 2);

  // Draw visited trail
  visited.forEach(key => {{
    const [r, c] = key.split(',').map(Number);
    // Skip if it's an obstacle or goal cell
    let isObs = obstacles.some(o => o[0] === r && o[1] === c);
    let isGoal = (r === goal[0] && c === goal[1]);
    if (!isObs && !isGoal) {{
      ctx.fillStyle = 'rgba(83, 216, 168, 0.15)';
      ctx.fillRect(cellX(c) + 1, cellY(r) + 1, cellSize - 2, cellSize - 2);
    }}
  }});

  // Draw agent
  if (stepIdx < totalSteps) {{
    const [ar, ac] = steps[stepIdx];
    visited.add(ar + ',' + ac);
    const ax = cellX(ac) + cellSize * 0.15;
    const ay = cellY(ar) + cellSize * 0.15;
    const as_ = cellSize * 0.7;
    ctx.fillStyle = '#537ec5';
    ctx.beginPath();
    ctx.roundRect(ax, ay, as_, as_, 6);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold ' + (cellSize * 0.35) + 'px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('A', cellX(ac) + cellSize / 2, cellY(ar) + cellSize / 2);
  }}

  // Draw grid lines
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 1;
  for (let i = 0; i <= gridSize; i++) {{
    ctx.beginPath();
    ctx.moveTo(pad + i * cellSize, pad);
    ctx.lineTo(pad + i * cellSize, pad + gridSize * cellSize);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad, pad + i * cellSize);
    ctx.lineTo(pad + gridSize * cellSize, pad + i * cellSize);
    ctx.stroke();
  }}

  // Info
  let cumReward = 0;
  for (let i = 0; i <= stepIdx && i < totalSteps; i++) cumReward += steps[i][3];
  const action = stepIdx < totalSteps ? dirLabels[steps[stepIdx][2]] : '-';
  const reward = stepIdx < totalSteps ? steps[stepIdx][3].toFixed(2) : '-';
  info.textContent = 'Step ' + (stepIdx + 1) + ' / ' + totalSteps +
    '   Action: ' + action +
    '   Reward: ' + reward +
    '   Total: ' + cumReward.toFixed(2);
}}

function tick() {{
  draw();
  stepIdx++;
  if (stepIdx >= totalSteps) {{
    stepIdx = totalSteps - 1;
    playing = false;
    playBtn.textContent = 'Play';
    clearInterval(timer);
    timer = null;
  }}
}}

function startTimer() {{
  if (timer) clearInterval(timer);
  timer = setInterval(tick, 500 / speed);
}}

playBtn.addEventListener('click', () => {{
  if (playing) {{
    playing = false;
    playBtn.textContent = 'Play';
    clearInterval(timer);
    timer = null;
  }} else {{
    if (stepIdx >= totalSteps - 1) {{ stepIdx = 0; visited.clear(); }}
    playing = true;
    playBtn.textContent = 'Pause';
    startTimer();
  }}
}});

restartBtn.addEventListener('click', () => {{
  stepIdx = 0;
  visited.clear();
  draw();
  if (!playing) {{
    playing = true;
    playBtn.textContent = 'Pause';
  }}
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

// Init
document.querySelectorAll('[data-speed]').forEach(b => b.classList.remove('active'));
document.getElementById('speed1').classList.add('active');
draw();
startTimer();
</script>
</body>
</html>
"##,
        grid_size = grid_size,
        obstacles_js = obs_json.join(","),
        goal_r = goal.0,
        goal_c = goal.1,
        steps_js = steps_json.join(","),
    );

    let mut f = std::fs::File::create(path).expect("Failed to create animation HTML file");
    f.write_all(html.as_bytes())
        .expect("Failed to write animation HTML file");
    println!("GridWorld animation saved to {}", path);
}

// ============================================================================
// Arithmetic animation HTML generation
// ============================================================================

struct ArithProblem {
    question: String,
    model_answer: i64,
    correct_answer: i64,
    is_correct: bool,
}

fn write_arithmetic_animation(path: &str, problems: &[ArithProblem]) {
    let problems_json: Vec<String> = problems
        .iter()
        .map(|p| {
            format!(
                r#"{{"q":"{}","model":{},"correct":{},"ok":{}}}"#,
                p.question, p.model_answer, p.correct_answer, p.is_correct
            )
        })
        .collect();

    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Arithmetic REINFORCE Animation</title>
<style>
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
  h1 {{ color: #e94560; margin-bottom: 10px; }}
  .card-area {{ width: 420px; height: 320px; display: flex; align-items: center; justify-content: center; margin: 20px 0; }}
  .card {{ background: #16213e; border-radius: 16px; padding: 40px 50px; text-align: center; min-width: 300px; position: relative; box-shadow: 0 4px 24px rgba(0,0,0,0.4); }}
  .question {{ font-size: 36px; font-weight: bold; color: #e0e0e0; margin-bottom: 24px; }}
  .answer-row {{ display: flex; justify-content: center; gap: 40px; margin-bottom: 18px; }}
  .answer-block {{ text-align: center; }}
  .answer-label {{ font-size: 13px; color: #888; margin-bottom: 4px; }}
  .answer-val {{ font-size: 28px; font-weight: bold; }}
  .model {{ color: #537ec5; }}
  .correct {{ color: #53d8a8; }}
  .result {{ font-size: 48px; margin-top: 10px; }}
  .result.ok {{ color: #53d8a8; }}
  .result.wrong {{ color: #e94560; }}
  .score {{ font-size: 20px; margin-top: 16px; font-variant-numeric: tabular-nums; }}
  .controls {{ margin-top: 16px; display: flex; gap: 12px; align-items: center; }}
  button {{ background: #0f3460; color: #e0e0e0; border: none; border-radius: 6px; padding: 8px 18px; cursor: pointer; font-size: 14px; }}
  button:hover {{ background: #e94560; }}
  button.active {{ background: #e94560; }}
  .info {{ margin-top: 12px; font-size: 16px; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<h1>Arithmetic REINFORCE — Evaluation</h1>
<div class="card-area">
  <div class="card" id="card">
    <div class="question" id="question"></div>
    <div class="answer-row">
      <div class="answer-block"><div class="answer-label">Model</div><div class="answer-val model" id="modelAns"></div></div>
      <div class="answer-block"><div class="answer-label">Correct</div><div class="answer-val correct" id="correctAns"></div></div>
    </div>
    <div class="result" id="result"></div>
  </div>
</div>
<div class="score" id="score"></div>
<div class="controls">
  <button id="playBtn">Pause</button>
  <button id="restartBtn">Restart</button>
  <button data-speed="0.5">0.5x</button>
  <button class="active" data-speed="1" id="speed1">1x</button>
  <button data-speed="2">2x</button>
</div>
<div class="info" id="info">Problem 0 / 0</div>
<script>
const problems = [{problems_js}];
const total = problems.length;

const question = document.getElementById('question');
const modelAns = document.getElementById('modelAns');
const correctAns = document.getElementById('correctAns');
const result = document.getElementById('result');
const scoreEl = document.getElementById('score');
const info = document.getElementById('info');
const playBtn = document.getElementById('playBtn');
const restartBtn = document.getElementById('restartBtn');

let idx = 0;
let playing = true;
let speed = 1;
let timer = null;
let phase = 0; // 0=show question, 1=reveal answer

function draw() {{
  if (idx >= total) return;
  const p = problems[idx];
  let numCorrect = 0;
  for (let i = 0; i < idx; i++) if (problems[i].ok) numCorrect++;

  if (phase === 0) {{
    question.textContent = p.q;
    modelAns.textContent = '?';
    correctAns.textContent = '?';
    result.textContent = '';
    result.className = 'result';
  }} else {{
    question.textContent = p.q;
    modelAns.textContent = p.model;
    correctAns.textContent = p.correct;
    if (p.ok) {{
      result.textContent = '\u2713';
      result.className = 'result ok';
      numCorrect++;
    }} else {{
      result.textContent = '\u2717';
      result.className = 'result wrong';
    }}
  }}
  scoreEl.textContent = numCorrect + ' / ' + (idx + (phase === 1 ? 1 : 0)) + ' correct';
  info.textContent = 'Problem ' + (idx + 1) + ' / ' + total;
}}

function tick() {{
  draw();
  if (phase === 0) {{
    phase = 1;
  }} else {{
    phase = 0;
    idx++;
    if (idx >= total) {{
      // Show final score
      let numCorrect = 0;
      for (let i = 0; i < total; i++) if (problems[i].ok) numCorrect++;
      scoreEl.textContent = 'Final: ' + numCorrect + ' / ' + total + ' correct';
      playing = false;
      playBtn.textContent = 'Play';
      clearInterval(timer);
      timer = null;
    }}
  }}
}}

function startTimer() {{
  if (timer) clearInterval(timer);
  timer = setInterval(tick, 2000 / speed);
}}

playBtn.addEventListener('click', () => {{
  if (playing) {{
    playing = false;
    playBtn.textContent = 'Play';
    clearInterval(timer);
    timer = null;
  }} else {{
    if (idx >= total) {{ idx = 0; phase = 0; }}
    playing = true;
    playBtn.textContent = 'Pause';
    startTimer();
  }}
}});

restartBtn.addEventListener('click', () => {{
  idx = 0;
  phase = 0;
  draw();
  if (!playing) {{
    playing = true;
    playBtn.textContent = 'Pause';
  }}
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

// Init
document.querySelectorAll('[data-speed]').forEach(b => b.classList.remove('active'));
document.getElementById('speed1').classList.add('active');
draw();
startTimer();
</script>
</body>
</html>
"##,
        problems_js = problems_json.join(","),
    );

    let mut f = std::fs::File::create(path).expect("Failed to create animation HTML file");
    f.write_all(html.as_bytes())
        .expect("Failed to write animation HTML file");
    println!("Arithmetic animation saved to {}", path);
}

// ============================================================================
// Network definitions
// ============================================================================

struct ActorCritic {
    actor_fc1: Linear,
    actor_fc2: Linear,
    actor_head: Linear,
    critic_fc1: Linear,
    critic_fc2: Linear,
    critic_head: Linear,
}

impl ActorCritic {
    fn new(obs_dim: usize, hidden: usize, num_actions: usize) -> Self {
        ActorCritic {
            actor_fc1: Linear::new(obs_dim, hidden),
            actor_fc2: Linear::new(hidden, hidden),
            actor_head: Linear::new(hidden, num_actions),
            critic_fc1: Linear::new(obs_dim, hidden),
            critic_fc2: Linear::new(hidden, hidden),
            critic_head: Linear::new(hidden, 1),
        }
    }

    fn policy(&self, x: &Tensor) -> Tensor {
        let h = self.actor_fc1.forward(x).relu();
        let h = self.actor_fc2.forward(&h).relu();
        self.actor_head.forward(&h)
    }

    fn value(&self, x: &Tensor) -> Tensor {
        let h = self.critic_fc1.forward(x).relu();
        let h = self.critic_fc2.forward(&h).relu();
        self.critic_head.forward(&h)
    }

    fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for t in self
            .actor_fc1
            .params()
            .into_iter()
            .chain(self.actor_fc2.params())
            .chain(self.actor_head.params())
            .chain(self.critic_fc1.params())
            .chain(self.critic_fc2.params())
            .chain(self.critic_head.params())
        {
            p.push(t.clone());
        }
        p
    }
}

struct QNetwork {
    fc1: Linear,
    fc2: Linear,
    head: Linear,
}

impl QNetwork {
    fn new(obs_dim: usize, hidden: usize, num_actions: usize) -> Self {
        QNetwork {
            fc1: Linear::new(obs_dim, hidden),
            fc2: Linear::new(hidden, hidden),
            head: Linear::new(hidden, num_actions),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.fc1.forward(x).relu();
        let h = self.fc2.forward(&h).relu();
        self.head.forward(&h)
    }

    fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for t in self
            .fc1
            .params()
            .into_iter()
            .chain(self.fc2.params())
            .chain(self.head.params())
        {
            p.push(t.clone());
        }
        p
    }
}

struct PolicyMlp {
    fc1: Linear,
    fc2: Linear,
    head: Linear,
}

impl PolicyMlp {
    fn new(obs_dim: usize, hidden: usize, num_actions: usize) -> Self {
        PolicyMlp {
            fc1: Linear::new(obs_dim, hidden),
            fc2: Linear::new(hidden, hidden),
            head: Linear::new(hidden, num_actions),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.fc1.forward(x).relu();
        let h = self.fc2.forward(&h).relu();
        self.head.forward(&h)
    }

    fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for t in self
            .fc1
            .params()
            .into_iter()
            .chain(self.fc2.params())
            .chain(self.head.params())
        {
            p.push(t.clone());
        }
        p
    }
}

// ============================================================================
// Demos
// ============================================================================

fn demo_cartpole_ppo() {
    println!("=== PPO on CartPole ===\n");
    random::seed(42);

    let mut env = CartPole::new(42);
    let net = ActorCritic::new(4, 64, 2);
    let params = net.params();
    let mut optimizer = Adam::new(params.clone(), 3e-4);

    let config = PpoConfig::default()
        .rollout_steps(256)
        .batch_size(64)
        .epochs(4)
        .clip_eps(0.2)
        .ent_coef(0.01)
        .vf_coef(0.5)
        .max_grad_norm(0.5);
    let mut trainer = PpoTrainer::new(config, 4);

    let policy_fn = |x: &Tensor| -> Tensor { net.policy(x) };
    let value_fn = |x: &Tensor| -> Tensor { net.value(x) };

    let mut rewards_hist = Vec::new();
    let mut pi_loss_hist = Vec::new();
    let mut vf_loss_hist = Vec::new();
    let mut entropy_hist = Vec::new();

    for iteration in 0..50 {
        let mean_reward = trainer.collect_rollouts(&mut env, &policy_fn, &value_fn, 2);
        let (pi_loss, vf_loss, entropy) =
            trainer.update(&policy_fn, &value_fn, &params, &mut optimizer, 2);

        rewards_hist.push(mean_reward);
        pi_loss_hist.push(pi_loss);
        vf_loss_hist.push(vf_loss);
        entropy_hist.push(entropy);

        if (iteration + 1) % 5 == 0 {
            println!(
                "Iter {:3}: reward={:6.1}  pi_loss={:7.4}  vf_loss={:7.4}  entropy={:.4}",
                iteration + 1,
                mean_reward,
                pi_loss,
                vf_loss,
                entropy
            );
        }
    }

    // Evaluation
    let mut total = 0.0;
    let num_eval = 10;
    let mut best_frames: Vec<[f32; 2]> = Vec::new();
    let mut best_reward = f32::NEG_INFINITY;
    for _ in 0..num_eval {
        let mut obs_data = env.reset().data();
        let mut ep_reward = 0.0;
        let mut ep_frames: Vec<[f32; 2]> = Vec::new();
        ep_frames.push([obs_data[0], obs_data[2]]); // x, theta
        loop {
            let obs_tensor = Tensor::new(obs_data.clone(), vec![1, 4], false);
            let logits = net.policy(&obs_tensor);
            let ld = logits.data();
            let action = if ld[0] > ld[1] { 0 } else { 1 };
            let action_t = Tensor::new(vec![action as f32], vec![1], false);
            let result = env.step(&action_t);
            ep_reward += result.reward;
            let next_data = result.observation.data();
            ep_frames.push([next_data[0], next_data[2]]);
            if result.done || result.truncated {
                break;
            }
            obs_data = next_data;
        }
        if ep_reward > best_reward {
            best_reward = ep_reward;
            best_frames = ep_frames;
        }
        total += ep_reward;
    }
    println!(
        "\nEval ({} episodes): mean_reward={:.1}\n",
        num_eval,
        total / num_eval as f32
    );

    write_cartpole_animation("rl_cartpole_anim.html", &best_frames);

    write_html(
        "rl_cartpole.html",
        "CartPole PPO Learning Curves",
        &[
            ChartSpec {
                title: "Mean Episode Reward",
                x_label: "Iteration",
                series: vec![Series {
                    name: "Reward",
                    color: "#e94560",
                    data: rewards_hist,
                }],
            },
            ChartSpec {
                title: "Policy Loss & Value Loss",
                x_label: "Iteration",
                series: vec![
                    Series {
                        name: "Policy Loss",
                        color: "#0f3460",
                        data: pi_loss_hist,
                    },
                    Series {
                        name: "Value Loss",
                        color: "#e94560",
                        data: vf_loss_hist,
                    },
                ],
            },
            ChartSpec {
                title: "Entropy",
                x_label: "Iteration",
                series: vec![Series {
                    name: "Entropy",
                    color: "#53d8a8",
                    data: entropy_hist,
                }],
            },
        ],
    );
}

fn demo_gridworld_dqn() {
    println!("=== DQN on GridWorld ===\n");
    random::seed(123);

    let grid_size = 5;
    let obs_dim = grid_size * grid_size;
    let num_actions = 4;
    let mut env = GridWorld::new(grid_size, 2, 123);

    let online = QNetwork::new(obs_dim, 128, num_actions);
    let target = QNetwork::new(obs_dim, 128, num_actions);
    let online_params = online.params();
    let target_params = target.params();
    copy_params(&online_params, &target_params);

    let mut optimizer = Adam::new(online_params.clone(), 5e-4);
    let config = DqnConfig::default()
        .batch_size(64)
        .buffer_size(10000)
        .gamma(0.99)
        .eps_start(1.0)
        .eps_end(0.02)
        .eps_decay(0.9995)
        .target_update(200);
    let mut trainer = DqnTrainer::new(config, obs_dim, 1);

    let online_fn = |x: &Tensor| -> Tensor { online.forward(x) };
    let target_fn = |x: &Tensor| -> Tensor { target.forward(x) };

    let num_episodes = 800;
    let mut recent_rewards: Vec<f32> = Vec::new();
    let mut rolling_mean_hist = Vec::new();
    let mut epsilon_hist = Vec::new();

    for episode in 0..num_episodes {
        let mut obs_data = env.reset().data();
        let mut ep_reward = 0.0;

        loop {
            let obs_tensor = Tensor::new(obs_data.clone(), vec![1, obs_dim], false);
            let action = trainer.select_action(&obs_tensor, &online_fn, num_actions);
            let action_t = Tensor::new(vec![action as f32], vec![1], false);
            let result = env.step(&action_t);

            let next_data = result.observation.data();
            trainer.store_transition(
                &obs_data,
                action,
                result.reward,
                &next_data,
                result.done,
            );
            ep_reward += result.reward;

            trainer.update(
                &online_fn,
                &target_fn,
                &online_params,
                &target_params,
                &mut optimizer,
                num_actions,
            );

            if result.done || result.truncated {
                break;
            }
            obs_data = next_data;
        }

        recent_rewards.push(ep_reward);
        if recent_rewards.len() > 50 {
            recent_rewards.remove(0);
        }
        let mean: f32 = recent_rewards.iter().sum::<f32>() / recent_rewards.len() as f32;
        rolling_mean_hist.push(mean);
        epsilon_hist.push(trainer.epsilon());

        if (episode + 1) % 50 == 0 {
            println!(
                "Episode {:3}: mean_reward(50)={:6.2}  eps={:.3}",
                episode + 1,
                mean,
                trainer.epsilon()
            );
        }
    }

    // Evaluation — run multiple episodes, pick best for animation
    // Parse grid layout once (obstacles/goal stay fixed across resets)
    let render_str = env.render();
    let mut obstacles: Vec<(usize, usize)> = Vec::new();
    let mut goal_pos = (grid_size - 1, grid_size - 1);
    for (r, line) in render_str.lines().enumerate() {
        for (c, ch) in line.chars().enumerate() {
            match ch {
                '#' => obstacles.push((r, c)),
                'G' => goal_pos = (r, c),
                _ => {}
            }
        }
    }

    // Extract agent position from obs tensor (cell with value ~0.33)
    let agent_pos_from_obs = |obs: &[f32]| -> (usize, usize) {
        for (i, &v) in obs.iter().enumerate() {
            if (v - 0.33).abs() < 0.1 {
                return (i / grid_size, i % grid_size);
            }
        }
        (0, 0)
    };

    let num_eval = 10;
    let mut best_steps: Vec<GridWorldStep> = Vec::new();
    let mut best_reward = f32::NEG_INFINITY;
    let mut total_reward = 0.0;
    for _ in 0..num_eval {
        let mut obs_data = env.reset().data();
        let mut ep_steps: Vec<GridWorldStep> = Vec::new();
        let mut ep_reward = 0.0;
        let start = agent_pos_from_obs(&obs_data);
        ep_steps.push(GridWorldStep {
            row: start.0,
            col: start.1,
            action: 0,
            reward: 0.0,
        });

        for _ in 0..grid_size * grid_size * 2 {
            let obs_tensor = Tensor::new(obs_data.clone(), vec![1, obs_dim], false);
            let q_vals = online.forward(&obs_tensor);
            let qd = q_vals.data();
            let action = qd
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let action_t = Tensor::new(vec![action as f32], vec![1], false);
            let result = env.step(&action_t);
            ep_reward += result.reward;

            let next_data = result.observation.data();
            let pos = agent_pos_from_obs(&next_data);
            ep_steps.push(GridWorldStep {
                row: pos.0,
                col: pos.1,
                action,
                reward: result.reward,
            });

            if result.done || result.truncated {
                break;
            }
            obs_data = next_data;
        }

        if ep_reward > best_reward {
            best_reward = ep_reward;
            best_steps = ep_steps;
        }
        total_reward += ep_reward;
    }
    println!(
        "\nEval ({} episodes): mean_reward={:.2}\n",
        num_eval,
        total_reward / num_eval as f32
    );

    // Print best path
    println!("Best eval path (reward={:.2}):", best_reward);
    for (i, s) in best_steps.iter().enumerate() {
        if i == 0 {
            println!("  start ({},{})", s.row, s.col);
        } else {
            let dir = ["up", "down", "left", "right"][s.action];
            println!(
                "  -> {} ({},{}) reward={:.2}",
                dir, s.row, s.col, s.reward
            );
        }
    }

    write_gridworld_animation(
        "rl_gridworld_anim.html",
        grid_size,
        &obstacles,
        goal_pos,
        &best_steps,
    );

    write_html(
        "rl_gridworld.html",
        "GridWorld DQN Learning Curves",
        &[
            ChartSpec {
                title: "Rolling Mean Reward (window=50)",
                x_label: "Episode",
                series: vec![Series {
                    name: "Mean Reward",
                    color: "#e94560",
                    data: rolling_mean_hist,
                }],
            },
            ChartSpec {
                title: "Epsilon Decay",
                x_label: "Episode",
                series: vec![Series {
                    name: "Epsilon",
                    color: "#53d8a8",
                    data: epsilon_hist,
                }],
            },
        ],
    );
}

fn demo_arithmetic_reinforce() {
    println!("=== REINFORCE on BasicArithmetic (addition only) ===\n");
    random::seed(7);

    // Addition only with max_val=3: answers in [0, 6], 7 action bins — tractable for REINFORCE
    let max_val = 3;
    let mut env = BasicArithmetic::addition_only(max_val, 7);
    let num_answer_bins = env.action_space().n();
    let obs_dim = 3;
    println!("Action space: {} bins (answers 0..{})\n", num_answer_bins, num_answer_bins - 1);

    let policy = PolicyMlp::new(obs_dim, 128, num_answer_bins);
    let params = policy.params();
    let mut optimizer = Adam::new(params, 1e-3);

    let reinforce = Reinforce::new(0.99).with_baseline();
    let policy_fn = |x: &Tensor| -> Tensor { policy.forward(x) };

    // Batch REINFORCE: aggregate 64 episodes per gradient update
    let batch_size = 64;
    let num_updates = 200;
    let mut recent_rewards: Vec<f32> = Vec::new();
    let mut accuracy_hist = Vec::new();

    for update in 0..num_updates {
        let mean_reward =
            reinforce.train_batch(&mut env, &policy_fn, &mut optimizer, batch_size);
        recent_rewards.push(mean_reward);
        if recent_rewards.len() > 20 {
            recent_rewards.remove(0);
        }
        let mean: f32 = recent_rewards.iter().sum::<f32>() / recent_rewards.len() as f32;
        accuracy_hist.push(mean * 100.0);

        if (update + 1) % 20 == 0 {
            println!(
                "Update {:3} (ep {:5}): accuracy(last 20 batches)={:.0}%",
                update + 1,
                (update + 1) * batch_size,
                mean * 100.0
            );
        }
    }

    // Evaluation — record problems for animation
    println!("\nEval (20 problems):");
    let mut correct = 0;
    let mut anim_problems: Vec<ArithProblem> = Vec::new();
    for _ in 0..20 {
        env.reset();
        let q = env.question();
        let ans = env.answer();
        let obs = Tensor::new(
            vec![env.a as f32, env.op as f32, env.b as f32],
            vec![1, obs_dim],
            false,
        );
        let logits = policy.forward(&obs);
        let ld = logits.data();
        let pred_idx = ld
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let pred_val = pred_idx as i64 + env.answer_offset;
        let ok = pred_val == env.correct;
        if ok {
            correct += 1;
        }
        println!(
            "  {} -> model={}, correct={} {}",
            q,
            pred_val,
            ans,
            if ok { "OK" } else { "WRONG" }
        );
        anim_problems.push(ArithProblem {
            question: q,
            model_answer: pred_val,
            correct_answer: env.correct,
            is_correct: ok,
        });
    }
    println!("\nAccuracy: {}/20\n", correct);

    write_arithmetic_animation("rl_arithmetic_anim.html", &anim_problems);

    write_html(
        "rl_arithmetic.html",
        "Arithmetic REINFORCE Learning Curve",
        &[ChartSpec {
            title: "Accuracy (%)",
            x_label: "Update",
            series: vec![Series {
                name: "Accuracy",
                color: "#e94560",
                data: accuracy_hist,
            }],
        }],
    );
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("cartpole");

    match mode {
        "cartpole" => demo_cartpole_ppo(),
        "gridworld" => demo_gridworld_dqn(),
        "arithmetic" => demo_arithmetic_reinforce(),
        _ => {
            println!("Unknown mode: {}", mode);
            println!("Usage: cargo run --example rl_demo --release [cartpole|gridworld|arithmetic]");
        }
    }
}
