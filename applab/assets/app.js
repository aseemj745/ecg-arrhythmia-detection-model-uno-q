// ECG Arrhythmia Classifier - dashboard client.
//
// Talks to python/main.py over the App Lab WebUI Brick (socket.io under the
// hood, wrapped by libs/arduino.js). The Python side pushes a full state
// snapshot on the "state" event roughly every 400ms; this file only draws
// whatever it is handed - it does not classify or interpret anything, all
// of that already happened on the board before this message was sent.
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
  const x = i => (i/(d.wave.length-1)) * W;
  const y = v => H - ((v - lo)/(hi - lo)) * H;

  ctx.strokeStyle = '#2b6cb0'; ctx.lineWidth = 1.4; ctx.beginPath();
  d.wave.forEach((v,i)=>{ const px=x(i), py=y(v); i===0?ctx.moveTo(px,py):ctx.lineTo(px,py); });
  ctx.stroke();

  d.beats.forEach(b=>{
    const frac = (b.t - t0)/span; if(frac<0||frac>1) return;
    const px = frac*W;
    const col = COLOUR[b.label] || '#888';
    ctx.fillStyle = col; ctx.globalAlpha = b.conf>=0.5?1:0.4;
    ctx.beginPath(); ctx.arc(px, 14, 4, 0, 7); ctx.fill();
    ctx.globalAlpha = 1;
    if(b.label !== 'NOR'){
      ctx.fillStyle = col; ctx.font='11px sans-serif'; ctx.textAlign='center';
      ctx.fillText(b.label, px, 30);
    }
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
let demoRunning = false;

function setDemoRunning(running){
  demoRunning = running;
  demoBtn.textContent = running ? 'Stop Demo Replay' : 'Start Demo Replay';
  demoBtn.classList.toggle('stop', running);
}

const ui = new WebUI();
ui.on_connect(() => {
  document.getElementById('status').textContent = 'connected, waiting for first update ...';
  demoBtn.disabled = false;
});
ui.on_disconnect(() => {
  document.getElementById('status').textContent = 'disconnected from the board';
  demoBtn.disabled = true;
});
ui.on_message('state', d => {
  // The status text itself says DEMO REPLAY while it's active (set on the
  // Python side) - mirror that into the button so a stray click on "Stop"
  // after someone else already stopped it, or after the demo runs to a
  // natural point, does not look out of sync with what is really running.
  setDemoRunning(d.status.startsWith('DEMO'));
  render(d);
});

demoBtn.addEventListener('click', () => {
  ui.send_message(demoRunning ? 'stop_demo' : 'start_demo');
});
