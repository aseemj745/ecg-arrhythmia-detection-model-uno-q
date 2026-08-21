// ECG Arrhythmia Classifier - dashboard client.
//
// Talks to python/main.py over the App Lab WebUI Brick (socket.io underneath,
// wrapped by libs/arduino.js). Python pushes a full state snapshot on the
// "state" event about every 400ms. This file only draws what it's handed, it
// doesn't classify or interpret anything.
const COLOUR = {NOR:"#1a9850",LBBB:"#d73027",RBBB:"#7b3294",PVC:"#e08214",AFIB:"#c51b7d"};
const canvas = document.getElementById('wave');
const ctx = canvas.getContext('2d');

function resize(){ canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight; }
window.addEventListener('resize', resize); resize();

function draw(d){
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  if(!d.wave.length) return;
  const t0 = d.t0, span = Math.max(d.now - d.t0, 0.001);
  const vmin = Math.min(...d.wave), vmax = Math.max(...d.wave);
  const pad = (vmax - vmin) * 0.15 || 1;
  const lo = vmin - pad, hi = vmax + pad;
  // Leave room at the top for the class labels so a tall R-peak never
  // collides with the text naming it.
  const TOP = 42;
  const x = i => (i/(d.wave.length-1)) * W;
  const y = v => TOP + (H - TOP) - ((v - lo)/(hi - lo)) * (H - TOP);

  ctx.strokeStyle = '#2b6cb0'; ctx.lineWidth = 1.4; ctx.beginPath();
  d.wave.forEach((v,i)=>{ const px=x(i), py=y(v); i===0?ctx.moveTo(px,py):ctx.lineTo(px,py); });
  ctx.stroke();

  // Every beat gets a name, not just a dot, NOR included, so you can read off
  // the trace what the model called each beat instead of decoding a colour.
  // Beats below the reporting gate never reach the episode table, so they're
  // drawn faded with a "?".
  //
  // The gate is per class and comes from the board, since it differs by class
  // and by mode. Reading it from the payload keeps "faded" here meaning
  // exactly "not in the table below".
  const gates = d.gates || {};

  // Live only, demo is untouched: a beat below the gate is drawn as NOR rather
  // than as a faded guess at an arrhythmia. Below the gate the table already
  // reports nothing, because on live input those calls sit near the 0.20
  // chance level, so drawing "AFIB?" told two different stories about the same
  // beat and the alarming one had no evidence behind it.
  //
  // This doesn't claim the model checked the beat and found it normal. It
  // declined to call it anything. Live mode reports an arrhythmia only when
  // confident, and this is what that looks like on the trace.
  const asNor = !!d.unconfident_as_nor;
  ctx.textAlign = 'center';
  let lastX = -1e9;
  d.beats.forEach(b=>{
    const frac = (b.t - t0)/span; if(frac<0||frac>1) return;
    const px = frac*W;
    let reported = b.conf >= (gates[b.label] || 0);
    let label = b.label;
    if(!reported && asNor){ label = 'NOR'; reported = true; }
    const col = COLOUR[label] || '#888';

    ctx.strokeStyle = col; ctx.globalAlpha = reported?0.55:0.2;
    ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(px, TOP-8); ctx.lineTo(px, TOP+4); ctx.stroke();
    ctx.globalAlpha = 1;

    // Skip the text if it would land on top of the previous label; the
    // tick above still marks the beat, so nothing is silently dropped.
    if(px - lastX < 34) return;
    lastX = px;
    const txt = reported ? label : label + '?';
    ctx.font = '600 11px system-ui,sans-serif';
    const w = ctx.measureText(txt).width + 10;
    ctx.globalAlpha = reported?1:0.45;
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.roundRect(px-w/2, TOP-24, w, 16, 4);
    ctx.fill();
    ctx.fillStyle = '#0b0d12';
    ctx.fillText(txt, px, TOP-12);
    ctx.globalAlpha = 1;
  });

  // Motion markers, one per completed pause visible in the window. A single
  // vertical line rather than a shaded span, because a pause has no width on
  // this axis: no samples are missing from the buffer, the ones either side of
  // a pause are just adjacent. Shading would imply a stretch of waveform we
  // ignored, when really it was never captured. The label carries the duration.
  const MOTION_COLOUR = '#94a3b8';
  (d.motion_windows || []).forEach(w => {
    if (w.t == null) return;
    const frac = (w.t - t0)/span; if(frac<0||frac>1) return;
    const px = frac*W;

    ctx.strokeStyle = MOTION_COLOUR; ctx.globalAlpha = 0.8;
    ctx.setLineDash([3,3]); ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.moveTo(px, TOP-8); ctx.lineTo(px, H); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;

    const txt = 'MOTION ' + w.duration.toFixed(1) + 's';
    ctx.font = '600 10.5px system-ui,sans-serif';
    const tw = ctx.measureText(txt).width + 10;
    ctx.fillStyle = MOTION_COLOUR;
    ctx.beginPath();
    ctx.roundRect(px-tw/2, TOP-24, tw, 16, 4);
    ctx.fill();
    ctx.fillStyle = '#0b0d12';
    ctx.fillText(txt, px, TOP-12);
  });
}

function pill(label){
  const c = COLOUR[label] || '#888';
  return `<span class="pill" style="background:${c}">${label}</span>`;
}

function render(d){
  document.getElementById('modebadge').textContent = d.mode.toUpperCase();
  document.getElementById('modebadge').className = 'badge ' +
    (d.mode==='live' ? 'b-live' : 'b-selftest');
  document.getElementById('status').innerHTML =
    `<b>${d.status}</b> &middot; ${d.source} &middot; `+
    `${d.samples_seen.toLocaleString()} samples processed`;
  // Explains why early beats carry no label - without this the warmup
  // period is indistinguishable from the model missing obvious beats.
  const warm = document.getElementById('warmup');
  warm.textContent = d.warmup || '';
  warm.style.display = d.warmup ? 'block' : 'none';

  // While the MPU6050 reports movement the MCU stops sending ECG, so the
  // trace below simply stops advancing. Say that out loud - a frozen
  // waveform with no explanation reads as a crashed app, and this is the
  // motion-rejection feature working exactly as intended.
  const mo = document.getElementById('motion');
  mo.textContent = d.motion
    ? 'MOVEMENT DETECTED - ECG paused by the MCU, not analysing. '
      + 'Hold still and it resumes automatically.'
    : '';
  mo.style.display = d.motion ? 'block' : 'none';

  // Motion Log table - live sensor only. A demo replay is a file, so the
  // whole panel is hidden rather than shown-and-empty; the backend already
  // sends an empty motion_windows list outside live mode, this just keeps
  // the table itself off screen too, matching the episodes/demo split.
  const mpanel = document.getElementById('motionpanel');
  const isLive = d.run_mode === 'live';
  mpanel.style.display = isLive ? 'block' : 'none';
  if(isLive){
    const mw = d.motion_windows || [];
    document.getElementById('motionempty').style.display =
      mw.length ? 'none' : 'block';
    document.getElementById('motionrows').innerHTML = mw.map(w =>
      `<tr><td>${w.at}</td><td>${w.duration.toFixed(1)}s</td></tr>`
    ).join('');
  }

  draw(d);
  const tbody = document.getElementById('episodes');
  document.getElementById('empty').style.display = d.episodes.length?'none':'block';
  tbody.innerHTML = d.episodes.map(e => `<tr class="${e.conf<0.5?'lowconf':''}">
    <td>${pill(e.label)}</td>
    <td>${e.start.toFixed(1)}s - ${e.end.toFixed(1)}s</td>
    <td>${e.n_beats}</td>
    <td>${e.hr.toFixed(0)} bpm</td>
    <td>${(e.conf*100).toFixed(0)}%</td>
    <td>${e.truth || '-'}</td>
  </tr>`).join('');
}

const demoBtn = document.getElementById('demoBtn');
const liveBtn = document.getElementById('liveBtn');
const stopBtn = document.getElementById('stopBtn');
const recordSel = document.getElementById('recordSel');
const recordNote = document.getElementById('recordNote');
let runMode = 'idle';
let recordsBuilt = false;

// Built once from the first state payload. Rebuilding on every update would
// fight the user for control of the dropdown while they are using it.
function buildRecords(records, selected){
  if(recordsBuilt || !records || !records.length) return;
  recordsBuilt = true;
  const groups = {};
  records.forEach(r => { (groups[r.condition] = groups[r.condition] || []).push(r); });
  const order = ['NOR','PVC','LBBB','RBBB','AFIB'];
  const title = {NOR:'Healthy - mostly normal sinus', PVC:'PVC - ventricular ectopy',
                 LBBB:'LBBB - left bundle branch block', RBBB:'RBBB - right bundle branch block',
                 AFIB:'AFib - atrial fibrillation'};
  order.filter(c => groups[c]).forEach(c => {
    const og = document.createElement('optgroup');
    og.label = title[c] || c;
    groups[c].forEach(r => {
      const o = document.createElement('option');
      o.value = r.record;
      o.textContent = `Record ${r.record} - ${r.note}`;
      og.appendChild(o);
    });
    recordSel.appendChild(og);
  });
  if(selected) recordSel.value = selected;
  showNote();
}

function showNote(){
  const o = recordSel.selectedOptions[0];
  if(!o) return;
  recordNote.textContent =
    `${o.textContent} - held-out test patient, never seen during training. ` +
    `Detections below are the model's own, computed on the board.`;
}

// Three states, same shape as the desktop GUI: idle / live / demo.
// Nothing runs until a Start is pressed, and Stop is the only way out of a
// running mode - so switching demo records is Stop, choose, Demo, and
// never has to detour through live capture.
function setRunMode(mode){
  runMode = mode;
  const running = mode !== 'idle';
  liveBtn.classList.toggle('active', mode === 'live');
  demoBtn.classList.toggle('active', mode === 'demo');
  // While something runs, both Starts are inert; Stop is the live control.
  liveBtn.disabled = running;
  demoBtn.disabled = running;
  stopBtn.disabled = !running;
  // Changing the record mid-replay would silently mismatch the label and
  // the trace, so the dropdown is only live while nothing is playing.
  recordSel.disabled = running;
}

const ui = new WebUI();
ui.on_connect(() => {
  document.getElementById('status').textContent = 'connected, waiting for first update ...';
  // The first state payload calls setDemoRunning(), which decides which of
  // the two buttons is the live control; until then leave both inert.
});
ui.on_disconnect(() => {
  document.getElementById('status').textContent = 'disconnected from the board';
  demoBtn.disabled = true; liveBtn.disabled = true;
  stopBtn.disabled = true; recordSel.disabled = true;
});
ui.on_message('state', d => {
  buildRecords(d.demo_records, d.demo_record);
  // The board is the authority on what is actually running - mirror its
  // run_mode rather than guessing from the status text, so a demo that
  // ends on its own, or a second browser driving the same board, leaves
  // the buttons showing the truth.
  setRunMode(d.run_mode || 'idle');
  render(d);
});

recordSel.addEventListener('change', showNote);

demoBtn.addEventListener('click', () => {
  if(runMode !== 'idle') return;
  ui.send_message('start_demo', {record: recordSel.value});
});

liveBtn.addEventListener('click', () => {
  if(runMode !== 'idle') return;
  ui.send_message('start_live');
});

stopBtn.addEventListener('click', () => {
  if(runMode === 'idle') return;
  ui.send_message('stop_all');
});
