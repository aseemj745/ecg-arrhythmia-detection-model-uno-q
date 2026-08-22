"""
ECG Web Dashboard
===================
A no-display alternative to ecg_gui_live.py. Runs entirely as a small local
web server on tally -- you view the live chart and analysis results from a
browser on ANY device on the same network (laptop, phone, etc). No X11, no
VNC, no monitor needed on tally itself.

Reuses the same LiveReader (live_stream.py) and the same analyze(signal, fs)
module contract as the desktop GUIs, so pqrst_adapter.py works here
unchanged.

By default the "Data source command" is the SIMULATOR
(adc_reader_template.py). To view/record real ECG data, switch it to:
    python3 adc_reader_uno_q_bridge.py
which receives real samples pushed from sketch/sketch.ino
(the MCU sketch that reads the BioAmp EXG Pill). See adc_reader_uno_q_bridge.py
for setup details.

Motion / "please be still" popup:
sketch.ino also reads an MPU6050 IMU. When it detects a big displacement,
the MCU pauses ECG collection and notifies the state change over the
Bridge; adc_reader_uno_q_bridge.py relays that as a "MOTION:0"/"MOTION:1"
line, live_stream.py's LiveReader exposes it as status()["motion_detected"],
and this page shows a "Please stay still" overlay on the chart while it's
true. To try the popup without any hardware, set the "Data source command"
to:
    python3 adc_reader_template.py --fs 250 --simulate-motion

Every raw sample shown on the live chart is also appended, as it arrives,
to a timestamped CSV under recordings/ (e.g. recordings/ecg_20260813_101500.csv)
-- click "Download CSV" on the page to grab the current session's file.

RUN ON TALLY:
    python3 ecg_web_dashboard.py
    (starts a web server on port 5000)

VIEW FROM YOUR LAPTOP'S BROWSER:
    http://<tally's IP address>:5000
    Find tally's IP by running on tally: hostname -I
"""

import os
import sys
import time
import threading
import importlib.util
import traceback
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_file, abort

from live_stream import LiveReader

app = Flask(__name__)

# ---------------------------------------------------------------- state

state = {
    "reader": None,
    "module_path": "",
    "last_result": None,
    "last_result_time": None,
    "last_error": None,
    "lock": threading.Lock(),
}

DEFAULT_COMMAND = "python3 adc_reader_template.py --fs 250"
DEFAULT_FS = 250.0
DEFAULT_WINDOW_S = 10.0

# Every raw sample shown on the dashboard is also appended to a CSV under
# this folder as it streams in, so the live chart and the saved file are
# always the same data -- whether the source is the simulator or the real
# BioAmp EXG Pill via adc_reader_uno_q_bridge.py.
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")


# ---------------------------------------------------------------- page

PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ECG Live Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: #f5f6f8; color: #222; }
  h1 { font-size: 20px; margin-bottom: 12px; }
  .row { display: flex; gap: 20px; flex-wrap: wrap; }
  .card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .controls { flex: 1 1 100%; }
  .controls label { display: inline-block; margin-right: 8px; font-size: 13px; color: #555; }
  .controls input[type=text] { width: 320px; padding: 5px; margin-right: 12px; }
  .controls input[type=number] { width: 70px; padding: 5px; margin-right: 12px; }
  button { padding: 6px 14px; margin-right: 8px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; }
  #startBtn { background: #2e7d32; color: white; }
  #stopBtn { background: #c62828; color: white; }
  #runBtn { background: #1565c0; color: white; }
  #status { font-size: 13px; color: #555; margin-top: 8px; }
  .chart-card { flex: 2 1 500px; position: relative; }
  .results-card { flex: 1 1 280px; max-height: 520px; overflow-y: auto; }
  pre { white-space: pre-wrap; font-size: 13px; }
  canvas { max-height: 420px; }

  /* "Please be still" motion popup, shown over the chart while the
     MPU6050 (see sketch.ino) reports a big displacement. */
  #motionOverlay {
    display: none;
    position: absolute;
    inset: 0;
    background: rgba(198, 40, 40, 0.94);
    border-radius: 10px;
    color: white;
    text-align: center;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    padding: 24px;
    z-index: 10;
  }
  #motionOverlay .icon { font-size: 40px; margin-bottom: 8px; }
  #motionOverlay .title { font-size: 20px; font-weight: 700; margin-bottom: 6px; }
  #motionOverlay .subtitle { font-size: 14px; opacity: 0.95; max-width: 380px; }
</style>
</head>
<body>
<h1>ECG Live Dashboard</h1>
<div class="row">
  <div class="card controls">
    <div>
      <label>Data source command</label>
      <input type="text" id="cmd" value="python3 adc_reader_template.py --fs 250">
      <span style="font-size:11px; color:#888;">(simulated; use "python3 adc_reader_uno_q_bridge.py" for the real BioAmp EXG Pill + MPU6050, or add "--simulate-motion" here to try the "please be still" popup with no hardware)</span>
      <label>fs (Hz)</label>
      <input type="number" id="fs" value="250">
      <label>window (s)</label>
      <input type="number" id="win" value="10">
    </div>
    <div style="margin-top:10px;">
      <label>Analysis module (.py path on tally)</label>
      <input type="text" id="modpath" value="pqrst_adapter.py" style="width:420px;">
    </div>
    <div style="margin-top:10px;">
      <button id="startBtn" onclick="startLive()">Start Live</button>
      <button id="stopBtn" onclick="stopLive()">Stop</button>
      <button id="runBtn" onclick="runModule()">Run Module Now</button>
      <button id="downloadBtn" onclick="downloadCsv()" style="background:#6a1b9a; color:white;">Download CSV</button>
      <label><input type="checkbox" id="autoRun" onchange="toggleAuto()"> Auto-run every</label>
      <input type="number" id="autoInterval" value="5" style="width:50px;"> sec
    </div>
    <div id="status">Idle.</div>
    <div id="recording" style="font-size:12px; color:#777; margin-top:4px;"></div>
  </div>
</div>
<div class="row" style="margin-top:16px;">
  <div class="card chart-card">
    <canvas id="ecgChart"></canvas>
    <div id="motionOverlay">
      <div class="icon">✋</div>
      <div class="title">Please stay still</div>
      <div class="subtitle">Motion detected — data collection is paused until you're still again.</div>
    </div>
  </div>
  <div class="card results-card">
    <b>Module output</b>
    <pre id="results">(no results yet)</pre>
  </div>
</div>

<script>
let autoTimer = null;
let plotTimer = null;
let lastAnnotations = null;
let lastPeaks = null;

const ctx = document.getElementById('ecgChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: { datasets: [
    { label: 'ECG', data: [], borderColor: '#1565c0', borderWidth: 1, pointRadius: 0, showLine: true },
    { label: 'peaks', data: [], borderColor: 'red', backgroundColor: 'red', pointRadius: 4, showLine: false },
  ]},
  options: {
    animation: false,
    parsing: false,
    scales: {
      x: { type: 'linear', title: { display: true, text: 'Time (s)' } },
      y: { title: { display: true, text: 'Amplitude' } },
    },
  },
});

function setStatus(msg) { document.getElementById('status').innerText = msg; }

function startLive() {
  const cmd = document.getElementById('cmd').value;
  const fs = parseFloat(document.getElementById('fs').value);
  const win = parseFloat(document.getElementById('win').value);
  fetch('/api/start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({command: cmd, fs: fs, window: win})
  }).then(r => r.json()).then(d => {
    setStatus(d.message);
    if (d.ok) {
      if (plotTimer) clearInterval(plotTimer);
      plotTimer = setInterval(updatePlot, 250);
    }
  });
}

function stopLive() {
  fetch('/api/stop', { method: 'POST' }).then(r => r.json()).then(d => {
    setStatus(d.message);
    if (plotTimer) { clearInterval(plotTimer); plotTimer = null; }
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; document.getElementById('autoRun').checked = false; }
    document.getElementById('motionOverlay').style.display = 'none';
  });
}

function updatePlot() {
  fetch('/api/data').then(r => r.json()).then(d => {
    if (!d.ok) { setStatus(d.message); return; }
    const signal = d.signal;
    const fs = d.fs;
    const points = signal.map((v, i) => ({x: i / fs, y: v}));
    chart.data.datasets[0].data = points;

    if (lastPeaks) {
      chart.data.datasets[1].data = lastPeaks
        .filter(i => i >= 0 && i < signal.length)
        .map(i => ({x: i / fs, y: signal[i]}));
    }
    chart.update('none');

    // d.motion_detected is true/false once the MPU6050-aware source has
    // reported at least once, or null for sources with no motion sensor
    // (e.g. the plain simulator without --simulate-motion).
    const overlay = document.getElementById('motionOverlay');
    overlay.style.display = (d.motion_detected === true) ? 'flex' : 'none';

    let statusMsg = `[${d.running ? 'running' : 'STOPPED'}] ${d.samples_received} samples received | buffer ${d.buffer_fill}/${d.buffer_capacity}`;
    if (d.motion_detected === true) {
      statusMsg += ' | MOTION DETECTED — collection paused';
    }
    setStatus(statusMsg);

    if (d.log_file) {
      document.getElementById('recording').innerText = `Recording to: ${d.log_file} (every displayed sample is being saved)`;
    }
  });
}

function downloadCsv() {
  window.location.href = '/api/download';
}

function runModule() {
  const modpath = document.getElementById('modpath').value;
  setStatus('Running module...');
  fetch('/api/analyze', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({module_path: modpath})
  }).then(r => r.json()).then(d => {
    if (!d.ok) {
      document.getElementById('results').innerText = 'ERROR:\\n' + d.message;
      setStatus('Module error.');
      return;
    }
    let text = '';
    for (const [k, v] of Object.entries(d.result)) {
      if (k === 'peaks' || k === 'annotations') continue;
      text += `${k}: ${v}\\n`;
    }
    if (d.result.peaks) text += `\\npeaks: ${d.result.peaks.length} indices\\n`;
    if (d.result.annotations) {
      text += '\\nannotations:\\n';
      for (const [k, v] of Object.entries(d.result.annotations)) text += `  ${k}: ${v.length} indices\\n`;
    }
    document.getElementById('results').innerText = text;
    lastPeaks = d.result.peaks || null;
    setStatus('Module ran successfully.');
  });
}

function toggleAuto() {
  const on = document.getElementById('autoRun').checked;
  if (on) {
    const interval = Math.max(1, parseFloat(document.getElementById('autoInterval').value)) * 1000;
    runModule();
    autoTimer = setInterval(runModule, interval);
  } else if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True)
    command = data.get("command", DEFAULT_COMMAND)
    fs = float(data.get("fs", DEFAULT_FS))
    window_s = float(data.get("window", DEFAULT_WINDOW_S))

    with state["lock"]:
        if state["reader"] is not None and state["reader"].is_running():
            return jsonify(ok=False, message="Already running. Stop it first.")

        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(RECORDINGS_DIR, f"ecg_{stamp}.csv")

        reader = LiveReader(command, fs=fs, window_seconds=window_s, log_path=log_path)
        try:
            reader.start()
        except Exception as e:
            return jsonify(ok=False, message=f"Failed to start: {e}")
        state["reader"] = reader

    return jsonify(ok=True, message=f"Streaming from: {command}", log_file=os.path.basename(log_path))


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state["lock"]:
        if state["reader"] is not None:
            state["reader"].stop()
    return jsonify(ok=True, message="Stopped.")


@app.route("/api/data")
def api_data():
    with state["lock"]:
        reader = state["reader"]
    if reader is None:
        return jsonify(ok=False, message="Not started yet. Click Start Live.")
    st = reader.status()
    signal = reader.snapshot()
    return jsonify(
        ok=True,
        signal=signal,
        fs=reader.fs,
        running=st["running"],
        samples_received=st["samples_received"],
        buffer_fill=st["buffer_fill"],
        buffer_capacity=st["buffer_capacity"],
        log_file=os.path.basename(st["log_path"]) if st["log_path"] else None,
        motion_detected=st["motion_detected"],
    )


@app.route("/api/download")
def api_download():
    with state["lock"]:
        reader = state["reader"]
    if reader is None or not reader.log_path or not os.path.isfile(reader.log_path):
        abort(404, "No recording available yet. Click Start Live first.")
    return send_file(reader.log_path, as_attachment=True, download_name=os.path.basename(reader.log_path))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True)
    module_path = data.get("module_path", "").strip()

    with state["lock"]:
        reader = state["reader"]
    if reader is None or not reader.snapshot():
        return jsonify(ok=False, message="No live data collected yet.")
    if not module_path or not os.path.isfile(module_path):
        return jsonify(ok=False, message=f"Module file not found: {module_path}")

    signal = np.array(reader.snapshot())
    fs = reader.fs

    try:
        module_name = "web_analysis_module_" + os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "analyze"):
            return jsonify(ok=False, message="Module has no analyze(signal, fs) function.")

        result = module.analyze(signal, fs)
        if not isinstance(result, dict):
            return jsonify(ok=False, message=f"analyze() must return a dict, got {type(result).__name__}.")

        # make JSON-serializable (numpy types -> native)
        clean = {}
        for k, v in result.items():
            if k == "peaks":
                clean[k] = [int(x) for x in v]
            elif k == "annotations":
                clean[k] = {label: [int(x) for x in idxs] for label, idxs in v.items()}
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            else:
                clean[k] = v

        return jsonify(ok=True, result=clean)
    except Exception:
        return jsonify(ok=False, message=traceback.format_exc())


def main():
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting ECG web dashboard on http://0.0.0.0:{port}")
    print("Open this from a browser on your laptop using tally's IP address, e.g. http://<tally-ip>:5000")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
