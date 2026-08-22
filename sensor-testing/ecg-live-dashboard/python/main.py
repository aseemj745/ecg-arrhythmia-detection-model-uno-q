# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "matplotlib",
# ]
# ///
"""
main.py  --  App Lab entry point for the ECG UNO Q project
=============================================================
This IS the app that arduino-app-cli / App Lab launches (python/main.py).
It is the ONLY process that can hold the Bridge connection to the MCU,
so it does both jobs in one process instead of two:

  1. Bridge receiver
     Registers Bridge.provide("ecg_batch", ...) to receive the 25-sample
     batches pushed by the STM32 sketch (Bridge.notify("ecg_batch", ...))
     and feeds them into an in-memory rolling buffer.

  2. Web dashboard
     A small HTTP server (Python's stdlib http.server -- NOT Flask) that
     serves the same live chart + "Run Module Now" UI as before, reading
     straight from the in-memory buffer.

VIEW:
    http://<tally-ip>:5000
"""

import os
import sys
import csv
import json
import threading
import importlib.util
import traceback
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from arduino.app_utils import App, Bridge


# ============================================================
# CONFIGURATION
# ============================================================

FS_HZ = 250.0
WINDOW_S = 10.0
BUFFER_SIZE = int(FS_HZ * WINDOW_S)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODULE_PATH = os.path.join(THIS_DIR, "pqrst_adapter.py")
RECORDINGS_DIR = os.path.join(THIS_DIR, "recordings")


# ============================================================
# IN-PROCESS ROLLING BUFFER (fed directly by the Bridge callback)
# ============================================================

class SensorBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.samples_received = 0
        self.log_path = None
        self._log_file = None
        self._log_writer = None

    def start_new_recording(self):
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with self.lock:
            self.log_path = os.path.join(RECORDINGS_DIR, f"ecg_{stamp}.csv")
            self._log_file = open(self.log_path, "w", newline="")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow(["sample_index", "timestamp_utc", "value"])
            self._log_file.flush()

    def stop_recording(self):
        with self.lock:
            if self._log_file is not None:
                try:
                    self._log_file.close()
                except Exception:
                    pass
            self._log_file = None
            self._log_writer = None
            self.log_path = None

    def ingest_batch(self, samples):
        """Called directly from the Bridge callback -- one batch (~25 values) at a time."""
        with self.lock:
            for value in samples:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                self.buffer.append(value)
                idx = self.samples_received
                self.samples_received += 1
                if self._log_writer is not None:
                    ts = datetime.now(timezone.utc).isoformat()
                    self._log_writer.writerow([idx, ts, value])
                    if idx % 25 == 0:
                        self._log_file.flush()

    def snapshot(self):
        with self.lock:
            return list(self.buffer)

    def status(self):
        with self.lock:
            return {
                "samples_received": self.samples_received,
                "buffer_fill": len(self.buffer),
                "buffer_capacity": self.buffer.maxlen,
                "log_path": self.log_path,
            }


sensor = SensorBuffer(maxlen=BUFFER_SIZE)


# ============================================================
# BRIDGE CALLBACK  (registered once, runs for the app's whole life)
# ============================================================

def on_ecg_batch(samples):
    """
    Called automatically by the Bridge whenever the STM32 sketch executes:
        Bridge.notify("ecg_batch", ecg_batch)
    `samples` is a list of ~25 raw ADC ints (100 ms at 250 Hz).
    """
    if not isinstance(samples, (list, tuple)):
        print(f"[ECG ERROR] Expected list/tuple, got {type(samples)}", flush=True)
        return
    sensor.ingest_batch(samples)


Bridge.provide("ecg_batch", on_ecg_batch)


# ============================================================
# ANALYSIS  (numpy/pandas/scipy/matplotlib are imported lazily,
# only when this runs -- their absence never affects app startup)
# ============================================================

def run_analysis(module_path: str) -> dict:
    module_path = (module_path or "").strip() or DEFAULT_MODULE_PATH

    signal_list = sensor.snapshot()
    if not signal_list:
        return {"ok": False, "message": "No live data collected yet -- is the MCU sketch running?"}
    if not os.path.isfile(module_path):
        return {"ok": False, "message": f"Module file not found: {module_path}"}

    try:
        import numpy as np  # lazy: only needed here, not at app startup
    except ImportError:
        return {
            "ok": False,
            "message": (
                "numpy is not installed in this environment, so analysis modules "
                "can't run. The live chart still works without it -- this only "
                "affects 'Run Module Now'. See python/requirements.txt / the PEP 723 "
                "block at the top of main.py for what needs to be installed."
            ),
        }

    signal = np.array(signal_list)
    fs = FS_HZ

    try:
        module_name = "web_analysis_module_" + os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "analyze"):
            return {"ok": False, "message": "Module has no analyze(signal, fs) function."}

        result = module.analyze(signal, fs)
        if not isinstance(result, dict):
            return {"ok": False, "message": f"analyze() must return a dict, got {type(result).__name__}."}

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

        return {"ok": True, "result": clean}
    except Exception:
        return {"ok": False, "message": traceback.format_exc()}


# ============================================================
# DASHBOARD PAGE  (plain string, no Jinja/Flask needed)
# ============================================================

PAGE_TEMPLATE = """
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
  .chart-card { flex: 2 1 500px; }
  .results-card { flex: 1 1 280px; max-height: 520px; overflow-y: auto; }
  pre { white-space: pre-wrap; font-size: 13px; }
  canvas { max-height: 420px; }
</style>
</head>
<body>
<h1>ECG Live Dashboard (real sensor -- Bridge-fed)</h1>
<div class="row">
  <div class="card controls">
    <div>
      <label>Analysis module (.py path on tally)</label>
      <input type="text" id="modpath" value="__DEFAULT_MODULE__" style="width:420px;">
    </div>
    <div style="margin-top:10px;">
      <button id="startBtn" onclick="startRecording()">Start Recording</button>
      <button id="stopBtn" onclick="stopRecording()">Stop Recording</button>
      <button id="runBtn" onclick="runModule()">Run Module Now</button>
      <button id="downloadBtn" onclick="downloadCsv()" style="background:#6a1b9a; color:white;">Download CSV</button>
      <label><input type="checkbox" id="autoRun" onchange="toggleAuto()"> Auto-run every</label>
      <input type="number" id="autoInterval" value="5" style="width:50px;"> sec
    </div>
    <div id="status">The Bridge connection is always live -- the chart below is streaming as soon as the sketch is running on the MCU. "Start Recording" only controls whether samples are also saved to a CSV.</div>
    <div id="recording" style="font-size:12px; color:#777; margin-top:4px;"></div>
  </div>
</div>
<div class="row" style="margin-top:16px;">
  <div class="card chart-card">
    <canvas id="ecgChart"></canvas>
  </div>
  <div class="card results-card">
    <b>Module output</b>
    <pre id="results">(no results yet)</pre>
  </div>
</div>

<script>
let autoTimer = null;
let plotTimer = setInterval(updatePlot, 250);
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

function startRecording() {
  fetch('/api/start', { method: 'POST' }).then(r => r.json()).then(d => setStatus(d.message));
}
function stopRecording() {
  fetch('/api/stop', { method: 'POST' }).then(r => r.json()).then(d => setStatus(d.message));
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
    setStatus(`${d.samples_received} samples received | buffer ${d.buffer_fill}/${d.buffer_capacity}`);
    if (d.log_file) {
      document.getElementById('recording').innerText = `Recording to: ${d.log_file}`;
    } else {
      document.getElementById('recording').innerText = '';
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


def render_page():
    return PAGE_TEMPLATE.replace("__DEFAULT_MODULE__", DEFAULT_MODULE_PATH)


# ============================================================
# HTTP SERVER  (stdlib http.server -- replaces Flask entirely)
# ============================================================

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[http] " + (fmt % args), flush=True)

    def _send_json(self, payload, code=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            body = render_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/data":
            st = sensor.status()
            self._send_json({
                "ok": True,
                "signal": sensor.snapshot(),
                "fs": FS_HZ,
                "samples_received": st["samples_received"],
                "buffer_fill": st["buffer_fill"],
                "buffer_capacity": st["buffer_capacity"],
                "log_file": os.path.basename(st["log_path"]) if st["log_path"] else None,
            })

        elif path == "/api/download":
            if not sensor.log_path or not os.path.isfile(sensor.log_path):
                self._send_json({"ok": False, "message": "No recording available yet."}, code=404)
                return
            with open(sensor.log_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(sensor.log_path)}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        else:
            self._send_json({"ok": False, "message": "Not found"}, code=404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/start":
            sensor.start_new_recording()
            self._send_json({"ok": True, "message": f"Recording to {os.path.basename(sensor.log_path)}"})

        elif path == "/api/stop":
            sensor.stop_recording()
            self._send_json({"ok": True, "message": "Recording stopped."})

        elif path == "/api/analyze":
            data = self._read_json_body()
            result = run_analysis(data.get("module_path", ""))
            self._send_json(result)

        else:
            self._send_json({"ok": False, "message": "Not found"}, code=404)


def run_http_server():
    port = int(os.environ.get("PORT", 5000))
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"[ECG] Dashboard listening on http://0.0.0.0:{port}", flush=True)
    server.serve_forever()


# ============================================================
# APP LAB ENTRY POINT
# ============================================================

def main():
    print("==============================================", flush=True)
    print("        ECG UNO Q -- LIVE DASHBOARD APP", flush=True)
    print("==============================================", flush=True)
    print(f"Expected sampling rate : {FS_HZ} Hz", flush=True)
    print(f"Rolling buffer         : {BUFFER_SIZE} samples ({WINDOW_S:.0f}s)", flush=True)
    print("Bridge method           : ecg_batch", flush=True)
    print("Waiting for STM32 sketch to start sending batches...", flush=True)
    print("==============================================", flush=True)

    # http.server blocks, so run it in a background thread; the main
    # thread stays free to run the Bridge's event loop via App.run().
    server_thread = threading.Thread(target=run_http_server, daemon=True)
    server_thread.start()

    # This call is what keeps the Bridge (and this whole app) alive --
    # it must run in the process arduino-app-cli launched, i.e. right here.
    App.run()


if __name__ == "__main__":
    main()
