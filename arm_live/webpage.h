// The web page served by the ESP32. Edit the HTML between the rawliteral markers.
#pragma once
#include <Arduino.h>

const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arm Live</title>
<style>
:root {
  --bg: #eef0ee; --panel: #f8f9f7; --fg: #1d2421; --muted: #5d6863; --line: #cfd6d2; --accent: #1f6f8b;
  --ok: #1e9e6a; --table: #d99a14; --none: #8b9692; --live: #e0502b; --view: #e3e7e4; --grid: #c3cbc6;
  --f-ui: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --f-mono: ui-monospace, Menlo, Consolas, "Roboto Mono", monospace;
}
@media (prefers-color-scheme: dark) { :root {
  --bg: #121614; --panel: #181d1b; --fg: #e3e8e5; --muted: #98a39e; --line: #2c3431; --accent: #5cb3d1;
  --ok: #34c98c; --table: #f0b42e; --none: #66716d; --live: #ff6b45; --view: #0d100f; --grid: #29312e; color-scheme: dark } }
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html, body { height: 100%; margin: 0; }
body { background: var(--bg); color: var(--fg); font: 14px/1.4 var(--f-ui); }
.app { display: grid; grid-template-columns: 330px 1fr; height: 100%; }
aside { background: var(--panel); border-right: 1px solid var(--line); overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
h1 { font-size: 21px; margin: 0; font-weight: 600; }
.pill { font: 600 11px var(--f-mono); padding: 2px 8px; border-radius: 999px; color: #fff; background: var(--none); }
.pill.on { background: var(--ok); }
.pill.moving { background: var(--live); }
.sec { display: flex; flex-direction: column; gap: 8px; }
.lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
.mono { font-family: var(--f-mono); font-size: 12px; font-variant-numeric: tabular-nums; }
.readout { background: var(--bg); border: 1px solid var(--line); border-radius: 4px; padding: 8px 10px; display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; }
.k { color: var(--muted); }
.row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
input, select, button { font: inherit; color: var(--fg); background: var(--bg); border: 1px solid var(--line); border-radius: 4px; padding: 6px 8px; }
input[type=text] { flex: 1; min-width: 0; font-family: var(--f-mono); }
.slider { display: grid; grid-template-columns: 70px 1fr 56px; gap: 8px; align-items: center; }
.testbar { background: var(--table); color: #1d2421; font-weight: 600; padding: 6px 10px; border-radius: 4px; font-size: 13px; }
.slider input { padding: 0; accent-color: var(--accent); min-width: 0; }
button { cursor: pointer; }
button.go { background: var(--accent); border-color: var(--accent); color: var(--panel); font-weight: 600; }
button:disabled { opacity: .5; cursor: default; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.legend label { display: grid; grid-template-columns: 16px 12px 1fr auto; gap: 8px; align-items: center; padding: 3px 0; cursor: pointer; }
.sw { width: 12px; height: 12px; border-radius: 50%; }
.cnt { color: var(--muted); }
#log { max-height: 180px; overflow-y: auto; background: var(--bg); border: 1px solid var(--line); border-radius: 4px; padding: 6px 8px; margin: 0; white-space: pre-wrap; word-break: break-word; }
.note { color: var(--muted); font-size: 12.5px; margin: 0; }
main { position: relative; min-width: 0; background: var(--view); }
canvas { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: grab; }
.views { position: absolute; top: 10px; right: 10px; display: flex; gap: 4px; }
.views button { padding: 4px 10px; font-size: 12px; background: var(--panel); }
.hud { position: absolute; left: 10px; bottom: 10px; font: 11px var(--f-mono); color: var(--muted); pointer-events: none; }
@media (max-width: 760px) {
  .app { grid-template-columns: 1fr; grid-template-rows: 58vh auto; height: auto; }
  main { order: -1; height: 58vh; }
  aside { border-right: 0; border-top: 1px solid var(--line); overflow: visible; }
}
</style>
</head>
<body>
<div class="app">
  <aside>
    <div class="head"><h1>Arm Live</h1><span class="pill" id="conn">offline</span></div>

    <div class="testbar" id="testbar" hidden>Test mode · slow speed. Check each joint below before sending points.</div>

    <div class="sec">
      <span class="lbl">Now</span>
      <div class="readout mono" id="live"><span class="k">waiting</span><span>for the arm…</span></div>
    </div>

    <div class="sec">
      <span class="lbl">Send a point (inches)</span>
      <form class="row" id="sendForm">
        <input type="text" id="xyz" value="6 0 4" aria-label="x y z">
        <button class="go" type="submit" id="sendBtn">Move</button>
      </form>
      <div class="slider"><label for="twist" class="note">Wrist twist</label><input type="range" id="twist" min="-90" max="90" step="1" value="0"><span class="mono" id="twistTxt">0°</span></div>
      <div class="slider"><label for="claw" class="note">Claw</label><input type="range" id="claw" min="0" max="100" step="1" value="0"><span class="mono" id="clawTxt">open</span></div>
      <div class="row">
        <button type="button" id="centerBtn">Center</button>
        <button type="button" id="stopBtn">Stop</button>
      </div>
    </div>

    <div class="sec">
      <span class="lbl">Test one joint</span>
      <div class="row">
        <select id="jogJoint" aria-label="Joint">
          <option value="0">Base</option><option value="1">Shoulder</option><option value="2">Elbow 1</option><option value="3">Elbow 2</option>
        </select>
        <input type="range" id="jogAngle" min="-90" max="90" step="5" value="0" style="flex:1;min-width:90px;padding:0;accent-color:var(--accent)" aria-label="Angle">
        <span class="mono" id="jogTxt" style="width:40px">0°</span>
        <button type="button" id="jogBtn">Move</button>
      </div>
      <p class="note" id="jogHint"></p>
    </div>

    <div class="sec" id="pickSec" hidden>
      <span class="lbl">Selected point</span>
      <div class="readout mono" id="pick"></div>
      <button class="go" type="button" id="pickSend">Send arm here</button>
    </div>

    <div class="sec">
      <span class="lbl">Map</span>
      <div class="legend" id="legend"></div>
      <div class="row">
        <span class="note">Grid step</span>
        <select id="step"><option value="1">1.0 in</option><option value="0.75" selected>0.75 in</option><option value="0.5">0.5 in</option></select>
        <label class="row note"><input type="checkbox" id="above" checked> z ≥ 0 only</label>
      </div>
      <p class="note">Click a dot to preview the pose, then send it. The map uses the limits reported by the arm.</p>
    </div>

    <div class="sec">
      <span class="lbl">Log</span>
      <pre class="mono" id="log"></pre>
    </div>
  </aside>
  <main>
    <canvas id="cv"></canvas>
    <div class="views"><button type="button" data-v="iso">3D</button><button type="button" data-v="top">Top</button><button type="button" data-v="side">Side</button></div>
    <div class="hud">drag to orbit · scroll or pinch to zoom · grid 2 in</div>
  </main>
</div>
<script>
const $ = id => document.getElementById(id);
const D2R = Math.PI / 180, R2D = 180 / Math.PI;

// ---------- config (replaced by the arm's own values on connect) ----------
let C = { L: [0.5, 4, 3.5, 7], min: [60, 85, 60, 90], max: [512, 485, 530, 490], dir: [-1, 1, 1, -1],
          center: 285, ppd: 200 / 90, pitch0: -90, pstep: 5, flip: 1, table: 1, tableZ: 0 };

// ---------- IK (mirrors the firmware) ----------
function a2p(a, i) {
  let v = C.center + C.dir[i] * a * C.ppd, ok = true;
  if (v < C.min[i]) { v = C.min[i]; ok = false; } else if (v > C.max[i]) { v = C.max[i]; ok = false; }
  return { p: Math.round(v), ok };
}
const p2a = (p, i) => (p - C.center) / (C.dir[i] * C.ppd);
function wrap(a) { while (a > 180) a -= 360; while (a < -180) a += 360; return a; }
function planar(r, sz, pitch, sign) {
  const [H, L1, L2, L3] = C.L, p = pitch * D2R;
  const wr = r - L3 * Math.cos(p), wz = sz - L3 * Math.sin(p), d = Math.hypot(wr, wz);
  if (d < Math.abs(L1 - L2) || d > L1 + L2) return null;
  const c = Math.max(-1, Math.min(1, (wr * wr + wz * wz - L1 * L1 - L2 * L2) / (2 * L1 * L2)));
  const e = sign * Math.acos(c);
  const s = Math.atan2(wz, wr) - Math.atan2(L2 * Math.sin(e), L1 + L2 * Math.cos(e));
  return { sh: 90 - s * R2D, el: e * R2D, wr: wrap((p - s - e) * R2D), ez: H + L1 * Math.sin(s), wz: H + wz };
}
function solve(x, y, z) {
  const r = Math.hypot(x, y), base = Math.atan2(y, x) * R2D, bp = a2p(base, 0);
  if (!bp.ok) return { status: 'base', base };
  const sz = z - C.L[0]; let hit = false;
  for (let off = 0; off <= 180; off += C.pstep) for (let side = 0; side < 2; side++) {
    if (side === 1 && (off === 0 || off === 180)) continue;
    const pitch = wrap(C.pitch0 + (side === 0 ? off : -off));
    for (let b = 0; b < (C.flip ? 2 : 1); b++) {
      const k = planar(r, sz, pitch, b === 0 ? 1 : -1);
      if (!k) continue;
      const s = a2p(k.sh, 1), e = a2p(k.el, 2), w = a2p(k.wr, 3);
      if (!(s.ok && e.ok && w.ok)) continue;
      if (C.table && (k.ez < C.tableZ || k.wz < C.tableZ)) { hit = true; continue; }
      return { status: 'ok', pitch, base, sh: k.sh, el: k.el, wr: k.wr, pwm: [bp.p, s.p, e.p, w.p] };
    }
  }
  return { status: hit ? 'table' : 'none', base };
}
function fk(pwm) {
  const [H, L1, L2, L3] = C.L, b = p2a(pwm[0], 0) * D2R, s = (90 - p2a(pwm[1], 1)) * D2R, e = p2a(pwm[2], 2) * D2R, w = p2a(pwm[3], 3) * D2R;
  const pts = [[0, 0], [0, H]]; let r = 0, z = H;
  r += L1 * Math.cos(s); z += L1 * Math.sin(s); pts.push([r, z]);
  r += L2 * Math.cos(s + e); z += L2 * Math.sin(s + e); pts.push([r, z]);
  r += L3 * Math.cos(s + e + w); z += L3 * Math.sin(s + e + w); pts.push([r, z]);
  return pts.map(([rr, zz]) => [rr * Math.cos(b), rr * Math.sin(b), zz]);
}

// ---------- workspace map ----------
const CATS = [
  { id: 'ok', name: 'Reachable', on: true },
  { id: 'table', name: 'Blocked: arm would hit the table', on: false },
  { id: 'none', name: 'Unreachable', on: false },
];
let map = { x: new Float32Array(0), y: new Float32Array(0), z: new Float32Array(0), cat: new Uint8Array(0), n: 0 };
let mapJob = 0;
function buildMap() {
  const job = ++mapJob, step = +$('step').value, xs = [], zs = [];
  for (let v = -15; v <= 15 + 1e-9; v += step) xs.push(+v.toFixed(3));
  for (let v = -10; v <= 16 + 1e-9; v += step) zs.push(+v.toFixed(3));
  const X = [], Y = [], Z = [], K = [], counts = [0, 0, 0]; let i = 0;
  (function chunk() {
    if (job !== mapJob) return;
    const t0 = performance.now();
    while (i < xs.length && performance.now() - t0 < 30) {
      const x = xs[i++];
      for (const y of xs) for (const z of zs) {
        const st = solve(x, y, z).status, c = CATS.findIndex(k => k.id === st);
        if (c < 0) continue;              // outside base range: not drawn
        X.push(x); Y.push(y); Z.push(z); K.push(c);
        if (z >= 0) counts[c]++;
      }
    }
    if (i < xs.length) { setTimeout(chunk, 0); return; }
    map = { x: Float32Array.from(X), y: Float32Array.from(Y), z: Float32Array.from(Z), cat: Uint8Array.from(K), n: K.length };
    CATS.forEach((c, j) => $('cnt-' + c.id).textContent = counts[j].toLocaleString());
    dirty = true;
  })();
}

// ---------- live state ----------
let live = { pwm: null, busy: 0, target: null, ik: null }, trail = [], pick = null;
function updateLive() {
  if (!live.pwm) return;
  const a = live.pwm.map((p, i) => p2a(p, i)), tip = fk(live.pwm)[4], f = v => (v >= 0 ? ' ' : '') + v.toFixed(1);
  const names = ['base', 'shoulder', 'elbow 1', 'elbow 2'];
  let h = `<span class="k">target</span><span>${live.target ? live.target.map(v => v.toFixed(2)).join(', ') : 'center'}</span>
    <span class="k">tip now</span><span>${tip.map(v => v.toFixed(2)).join(', ')}</span>`;
  if (live.ik && live.ik.ok) h += `<span class="k">tip pitch</span><span>${f(live.ik.pitch)}°</span>`;
  names.forEach((n, i) => h += `<span class="k">${n}</span><span>${f(a[i])}° · pwm ${live.pwm[i]}</span>`);
  if (live.pwm.length >= 6 && C.twist) {
    h += `<span class="k">wrist twist</span><span>${f(twistAngle(live.pwm[4]))}° · pwm ${live.pwm[4]}</span>`;
    h += `<span class="k">claw</span><span>${clawPct(live.pwm[5]).toFixed(0)}% closed · pwm ${live.pwm[5]}</span>`;
  }
  $('live').innerHTML = h;
  const c = $('conn'); c.textContent = live.busy ? 'moving' : 'idle'; c.className = 'pill ' + (live.busy ? 'moving' : 'on');
  const last = trail[trail.length - 1];
  if (!last || Math.hypot(tip[0] - last[0], tip[1] - last[1], tip[2] - last[2]) > 0.05) { trail.push(tip); if (trail.length > 400) trail.shift(); }
  dirty = true;
}
function log(m) {
  const el = $('log'), t = new Date().toLocaleTimeString([], { hour12: false });
  el.textContent += `${t}  ${m}\n`;
  const lines = el.textContent.split('\n'); if (lines.length > 120) el.textContent = lines.slice(-120).join('\n');
  el.scrollTop = el.scrollHeight;
}

const twistAngle = p => (p - C.center) / (C.twist[2] * C.ppd);
const clawPct = p => 100 * (p - C.claw[0]) / ((C.claw[1] - C.claw[0]) || 1);
function syncSliders() {
  if (!C.twist) return;
  const a = [twistAngle(C.twist[0]), twistAngle(C.twist[1])];
  $('twist').min = Math.ceil(Math.min(...a)); $('twist').max = Math.floor(Math.max(...a));
  if (live.pwm && live.pwm.length >= 6 && document.activeElement !== $('twist') && document.activeElement !== $('claw')) {
    $('twist').value = Math.round(twistAngle(live.pwm[4])); $('claw').value = Math.round(clawPct(live.pwm[5]));
  }
  $('twistTxt').textContent = $('twist').value + '°';
  $('clawTxt').textContent = +$('claw').value === 0 ? 'open' : $('claw').value + '%';
}
$('twist').addEventListener('input', () => $('twistTxt').textContent = $('twist').value + '°');
$('twist').addEventListener('change', () => send(`r ${$('twist').value}`));
$('claw').addEventListener('input', () => $('clawTxt').textContent = +$('claw').value === 0 ? 'open' : $('claw').value + '%');
$('claw').addEventListener('change', () => send(`g ${$('claw').value}`));

const JOG_HINTS = [
  'Base: + should turn it counterclockwise, seen from above.',
  'Shoulder: 0° is straight up. + should lean the arm forward.',
  'Elbow 1: + should bend the next link upward when the arm reaches forward.',
  'Elbow 2: + should tilt the claw upward, the same way as elbow 1.',
];
function syncJog() { $('jogTxt').textContent = $('jogAngle').value + '°'; $('jogHint').textContent = JOG_HINTS[+$('jogJoint').value] + ' If it goes the other way, flip that joint\'s DIRECTION in the code. Start with small angles like ±15°.'; }
$('jogAngle').addEventListener('input', syncJog);
$('jogJoint').addEventListener('change', () => { $('jogAngle').value = 0; syncJog(); });
$('jogBtn').onclick = () => send(`j ${$('jogJoint').value} ${$('jogAngle').value}`);
syncJog();

// ---------- websocket ----------
let ws = null;
function connect() {
  ws = new WebSocket(`ws://${location.hostname || '192.168.4.1'}:81/`);
  ws.onopen = () => { $('conn').textContent = 'idle'; $('conn').className = 'pill on'; log('Connected'); };
  ws.onclose = () => { $('conn').textContent = 'offline'; $('conn').className = 'pill'; setTimeout(connect, 2000); };
  ws.onmessage = ev => {
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    if (m.t === 'cfg') { const { t, ...rest } = m; C = rest; $('testbar').hidden = !C.test; buildMap(); syncSliders(); }
    else if (m.t === 's') { const was = live.busy; live.pwm = m.p; live.busy = m.b; live.target = m.g; updateLive(); if (was && !m.b) syncSliders(); }
    else if (m.t === 'k') { live.ik = m; }
    else if (m.t === 'l') log(m.m);
  };
}
function send(cmd) {
  if (!ws || ws.readyState !== 1) { log('Not connected to the arm'); return; }
  ws.send(cmd);
}
$('sendForm').addEventListener('submit', e => {
  e.preventDefault();
  const v = $('xyz').value.split(/[\s,]+/).filter(Boolean).map(Number);
  if (v.length !== 3 || v.some(isNaN)) { log('Enter three numbers: x y z'); return; }
  send(`m ${v.join(' ')}`);
});
$('centerBtn').onclick = () => send('c');
$('stopBtn').onclick = () => send('s');
$('pickSend').onclick = () => { if (pick) { $('xyz').value = pick.xyz.join(' '); send(`m ${pick.xyz.join(' ')}`); } };

// ---------- legend ----------
$('legend').innerHTML = CATS.map(c => `<label><input type="checkbox" id="cat-${c.id}" ${c.on ? 'checked' : ''}><span class="sw" style="background:var(--${c.id})"></span><span>${c.name}</span><span class="cnt mono" id="cnt-${c.id}">–</span></label>`).join('');
CATS.forEach(c => $('cat-' + c.id).addEventListener('change', e => { c.on = e.target.checked; dirty = true; }));
$('step').addEventListener('change', buildMap);
$('above').addEventListener('change', () => dirty = true);

// ---------- renderer (plain canvas, no libraries, works offline) ----------
const cv = $('cv'), g = cv.getContext('2d');
let W = 0, Hh = 0, dpr = 1, dirty = true;
const cam = { yaw: -0.75, el: 0.5, dist: 46, tx: 3, ty: 0, tz: 3 };
const VIEWS = { iso: { yaw: -0.75, el: 0.5 }, top: { yaw: -Math.PI / 2 + 0.0001, el: 1.5 }, side: { yaw: 0, el: 0.02 } };
let col = {};
function readColors() { const s = getComputedStyle(document.documentElement); ['ok', 'table', 'none', 'live', 'fg', 'muted', 'accent', 'grid', 'view'].forEach(k => col[k] = s.getPropertyValue('--' + k).trim()); dirty = true; }
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readColors);
function resize() { const r = cv.getBoundingClientRect(); dpr = Math.min(devicePixelRatio || 1, 2); W = r.width; Hh = r.height; cv.width = W * dpr; cv.height = Hh * dpr; dirty = true; }
new ResizeObserver(resize).observe(cv);

function proj(x, y, z) {
  x -= cam.tx; y -= cam.ty; z -= cam.tz;
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), ce = Math.cos(cam.el), se = Math.sin(cam.el);
  const u = x * cy - y * sy, Y = x * sy + y * cy;
  const v = z * ce + Y * se, d = Y * ce - z * se;
  const s = Math.min(W, Hh) * 1.15 / (cam.dist + d);
  return [W / 2 + u * s, Hh / 2 - v * s, d, s];
}
function line(pts, color, width, dash) {
  g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash || []); g.beginPath();
  pts.forEach((p, i) => { const q = proj(...p); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); }); g.stroke(); g.setLineDash([]);
}
function drawArm(joints, color, width, dash) {
  line(joints, color, width, dash);
  joints.slice(1).forEach((j, i) => { const q = proj(...j); g.fillStyle = i === 3 ? col.live : color; g.beginPath(); g.arc(q[0], q[1], i === 3 ? width * 0.9 : width * 0.75, 0, 7); g.fill(); });
}
let screen = { sx: new Float32Array(0), sy: new Float32Array(0), d: new Float32Array(0), idx: new Int32Array(0), n: 0 };
function render() {
  dirty = false;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = col.view; g.fillRect(0, 0, W, Hh);

  // floor grid and base range
  for (let v = -14; v <= 14; v += 2) { line([[v, -14, 0], [v, 14, 0]], col.grid, 1); line([[-14, v, 0], [14, v, 0]], col.grid, 1); }
  const lo = p2a(C.max[0], 0), hi = p2a(C.min[0], 0), wedge = [[0, 0, 0]];
  for (let a = Math.min(lo, hi); a <= Math.max(lo, hi) + 0.01; a += 3) wedge.push([15 * Math.cos(a * D2R), 15 * Math.sin(a * D2R), 0]);
  wedge.push([0, 0, 0]); line(wedge, col.accent, 1.2);
  const xa = proj(16, 0, 0); g.fillStyle = col.muted; g.font = '11px ' + getComputedStyle(document.body).fontFamily; g.fillText('+X', xa[0] + 4, xa[1]);
  const ya = proj(0, 16, 0); g.fillText('+Y', ya[0] + 4, ya[1]);

  // points, depth-binned far to near
  const n = map.n, above = $('above').checked, NB = 14;
  if (screen.sx.length < n) screen = { sx: new Float32Array(n), sy: new Float32Array(n), d: new Float32Array(n), idx: new Int32Array(n), n: 0 };
  const bins = Array.from({ length: NB }, () => []); let m = 0;
  for (let i = 0; i < n; i++) {
    const c = map.cat[i]; if (!CATS[c].on || (above && map.z[i] < 0)) continue;
    const q = proj(map.x[i], map.y[i], map.z[i]);
    screen.sx[m] = q[0]; screen.sy[m] = q[1]; screen.d[m] = q[2]; screen.idx[m] = i;
    const b = Math.max(0, Math.min(NB - 1, Math.floor((q[2] + 20) / 40 * NB)));
    bins[NB - 1 - b].push(m); m++;
  }
  screen.n = m;
  const step = +$('step').value, catColors = CATS.map(c => col[c.id]);
  for (let b = 0; b < NB; b++) {
    g.globalAlpha = 0.25 + 0.75 * (b / (NB - 1));
    for (const k of bins[b]) {
      const i = screen.idx[k], c = map.cat[i];
      const sz = Math.max(1.5, step * 0.38 * Math.min(W, Hh) * 1.15 / (cam.dist + screen.d[k]));
      g.fillStyle = catColors[c];
      g.fillRect(screen.sx[k] - sz / 2, screen.sy[k] - sz / 2, sz, sz);
    }
  }
  g.globalAlpha = 1;

  // trail, preview, live arm, target
  if (trail.length > 1) { g.globalAlpha = 0.7; line(trail, col.live, 1.5); g.globalAlpha = 1; }
  if (pick && pick.res.status === 'ok') drawArm(fk(pick.res.pwm), col.muted, 3, [6, 4]);
  if (pick) { const q = proj(...pick.xyz); g.strokeStyle = col.fg; g.lineWidth = 1.5; g.beginPath(); g.arc(q[0], q[1], 7, 0, 7); g.stroke(); }
  if (live.target) { const q = proj(...live.target); g.strokeStyle = col.live; g.lineWidth = 2; g.beginPath(); g.moveTo(q[0] - 7, q[1]); g.lineTo(q[0] + 7, q[1]); g.moveTo(q[0], q[1] - 7); g.lineTo(q[0], q[1] + 7); g.stroke(); }
  const q0 = proj(0, 0, 0); g.fillStyle = col.muted; g.beginPath(); g.ellipse(q0[0], q0[1], 1.3 * q0[3], 0.6 * q0[3], 0, 0, 7); g.fill();
  if (live.pwm) drawArm(fk(live.pwm), col.fg, 6);
}
(function frame() { if (dirty && W) render(); requestAnimationFrame(frame); })();

// ---------- interaction ----------
const ptrs = new Map(); let pinch0 = 0, dist0 = 0, moved = 0;
cv.addEventListener('pointerdown', e => { cv.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]); moved = 0;
  if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch0 = Math.hypot(a[0] - b[0], a[1] - b[1]); dist0 = cam.dist; } });
cv.addEventListener('pointermove', e => {
  if (!ptrs.has(e.pointerId)) return;
  const prev = ptrs.get(e.pointerId), dx = e.clientX - prev[0], dy = e.clientY - prev[1];
  ptrs.set(e.pointerId, [e.clientX, e.clientY]); moved += Math.abs(dx) + Math.abs(dy);
  if (ptrs.size === 1) { cam.yaw -= dx * 0.008; cam.el = Math.max(-0.3, Math.min(1.55, cam.el + dy * 0.008)); }
  else if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; cam.dist = Math.max(18, Math.min(120, dist0 * pinch0 / Math.max(1, Math.hypot(a[0] - b[0], a[1] - b[1])))); }
  dirty = true;
});
const up = e => { const wasTap = ptrs.size === 1 && moved < 5; ptrs.delete(e.pointerId); if (wasTap) pickAt(e); };
cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', e => ptrs.delete(e.pointerId));
cv.addEventListener('wheel', e => { e.preventDefault(); cam.dist = Math.max(18, Math.min(120, cam.dist * Math.exp(e.deltaY * 0.001))); dirty = true; }, { passive: false });
document.querySelectorAll('.views button').forEach(b => b.onclick = () => { Object.assign(cam, VIEWS[b.dataset.v]); dirty = true; });

function pickAt(e) {
  const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  let best = -1, bestD = Infinity;
  for (let k = 0; k < screen.n; k++) {
    if (Math.hypot(screen.sx[k] - mx, screen.sy[k] - my) > 9) continue;
    if (screen.d[k] < bestD) { bestD = screen.d[k]; best = k; }
  }
  if (best < 0) { pick = null; $('pickSec').hidden = true; dirty = true; return; }
  const i = screen.idx[best], xyz = [map.x[i], map.y[i], map.z[i]], res = solve(...xyz);
  pick = { xyz, res };
  const f = v => (v >= 0 ? ' ' : '') + v.toFixed(1);
  let h = `<span class="k">point</span><span>${xyz.join(', ')}</span>`;
  if (res.status === 'ok') {
    h += `<span class="k">tip pitch</span><span>${f(res.pitch)}°</span>`;
    ['base', 'shoulder', 'elbow 1', 'elbow 2'].forEach((n, j) => h += `<span class="k">${n}</span><span>${f([res.base, res.sh, res.el, res.wr][j])}° · pwm ${res.pwm[j]}</span>`);
  } else h += `<span class="k">result</span><span>${res.status === 'table' ? 'would hit the table' : 'unreachable'}</span>`;
  $('pick').innerHTML = h; $('pickSend').disabled = res.status !== 'ok'; $('pickSec').hidden = false; dirty = true;
}

readColors(); resize(); buildMap(); connect();
</script>
</body>
</html>
)rawliteral";
