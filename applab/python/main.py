"""
ECG Arrhythmia Classifier - App Lab entry point for the Arduino UNO Q.

Runs the INT8 ONNX model on the QRB2210 Linux side and serves a dashboard you
can open in a browser while the board does the work.

    python3 python/main.py --mode selftest   # no hardware needed, run this first
    python3 python/main.py                   # live, and the default

selftest replays a bundled 90s clip of MIT-BIH record 119, which the model
never trained on. Checks the model runs on ARM without involving any wiring.

live reads what the MCU pushes over the Bridge. The Bridge is push, not pull,
so the callback just queues each batch and returns. read_bridge_batch() drains
the queue, so run_loop() gets the same shape either way and doesn't need to
know which mode it's in.

Default is live because App Lab's Start button can't pass arguments, so
whatever the default is is what a demo actually runs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from ecg import config as C                                        # noqa: E402
from ecg.pipeline import (ECGAnalyzer, StreamAnalyzer,             # noqa: E402
                          group_episodes)

MODEL = APP_ROOT / "models" / "model_int8.onnx"
SAMPLE = APP_ROOT / "data" / "sample_ecg_250hz.csv"

# Must match FS_HZ in sketch/sketch.ino. Was 250, which was wrong - timing a
# real capture gave 124.96 Hz. A wrong value doesn't crash, it just scales
# every heart rate, so update this if you change the sketch.
MCU_SAMPLE_RATE = 125

# The bundled selftest clip is genuinely 250 Hz. Separate constant since the
# two rates differ.
SELFTEST_SAMPLE_RATE = 250

# Samples per Bridge.notify(), so 200 ms at 125 Hz. Only useful for reading logs.
MCU_BATCH_SIZE = 25

DASHBOARD_PORT = 8080
DASHBOARD_WINDOW_S = 8.0     # seconds of waveform shown on the page


# --------------------------------------------------------------------------
# Live: Bridge push -> thread-safe queue -> pull-shaped read()
# --------------------------------------------------------------------------
_bridge_queue: "queue.Queue" = queue.Queue()

# Set while a demo replay is running. Up here because _on_ecg_batch reads it.
_demo_stop: "threading.Event | None" = None

# Who owns the analysis stream: "idle", "live" or "demo". One at a time, which
# keeps the two sources out of each other's buffer. Starts idle because the
# sketch pushes as soon as the app boots, so defaulting to "live" meant a
# session was always running before anyone picked anything.
_run_mode = "idle"


# Motion state, pushed by the sketch as Bridge.notify("motion_state", 0/1).
# The MCU decides and stops sending ECG while moving, so the bad samples never
# get here. This side just reports what it did.
#
# No StreamAnalyzer reset when motion ends - RR_CLIP drops the odd interval
# and the 10-beat window absorbs the rest. Resetting would cost ~10s of warmup
# after every movement for nothing.
MIN_MOTION_LOG_S = 0.30    # pauses shorter than this aren't logged

_motion_active = False
_motion_started_at = 0.0
_motion_windows = collections.deque(maxlen=200)   # completed, for the UI
_motion_log = queue.Queue()                       # completed, for the CSV


def _on_motion_state(value, stream=None):
    """Edge-triggered by the sketch. Runs on the framework's dispatch thread,
    so it does the minimum: flip a flag and queue a completed window for
    run_loop to write.

    Duration uses wall-clock because no samples arrive during motion, so
    there's no sample counter to measure it with. `stream` is only read to
    find where the pause sits in sample time, so the browser can put the
    marker at the right x position on the waveform.
    """
    global _motion_active, _motion_started_at
    # Live only. The IMU keeps reporting during a demo or in idle, but what's
    # being analysed then is a file, so logging "ECG paused by movement"
    # against it would be wrong. A bump during a demo used to leave a phantom
    # window that showed up in the next live session.
    if _run_mode != "live":
        return
    try:
        moving = bool(int(value))
    except (TypeError, ValueError):
        print("[motion] unexpected payload %r, ignoring" % (value,), flush=True)
        return
    if moving == _motion_active:
        return
    now = time.time()
    if moving:
        _motion_active = True
        _motion_started_at = now
        print("  [MOTION] movement detected - MCU paused ECG", flush=True)
    else:
        _motion_active = False
        dur = now - _motion_started_at
        sample_t = None
        if stream is not None and stream.fs:
            sample_t = (stream.offset + len(stream.buf)) / stream.fs
        window = {"start": _motion_started_at, "end": now,
                  "duration": round(dur, 2), "sample_t": sample_t}
        if dur < MIN_MOTION_LOG_S:
            # Too short to be worth a row. The gate still fired and the banner
            # still showed, this only decides what gets recorded. Keeps the
            # log readable on a board running older firmware with the
            # symmetric debounce, which produced a wall of sub-0.2s rows.
            print("  [MOTION] settled after %.2fs - too brief to log"
                  % dur, flush=True)
        else:
            _motion_windows.append(window)
            _motion_log.put(window)
            print("  [MOTION] settled after %.1fs - ECG resumed" % dur,
                  flush=True)


def _demo_active():
    return _demo_stop is not None and not _demo_stop.is_set()


def _enqueue(samples):
    """The demo's way into the analysis queue. Not _on_ecg_batch(), because
    that drops everything while a demo is active and the demo would silence
    itself."""
    _bridge_queue.put(samples)


def _on_ecg_batch(samples):
    """
    Registered as Bridge.provide("ecg_batch", _on_ecg_batch).

    Called by the framework on its own dispatch thread, so it must never block
    or raise. It only checks the shape and queues the batch; the real work
    happens in run_loop.

    Drops the batch unless live mode owns the stream. The sketch streams
    continuously from startup, so without this the sensor's samples got
    interleaved with the demo's MIT-BIH samples in one buffer, which made the
    waveform look corrupted and dropped the model to near-chance confidence.
    """
    if _run_mode != "live":
        return
    if not isinstance(samples, (list, tuple)):
        print(f"[ECG] Bridge sent unexpected type {type(samples)} for "
              f"'ecg_batch', ignoring", flush=True)
        return
    _bridge_queue.put(samples)


def drain_bridge_queue():
    """
    Throw away anything queued but not yet analysed.

    Needed when switching between demo and live, since both feed the same
    queue. Without it, a demo that was just stopped leaves batches behind that
    run_loop then reports as sensor data. That's what made the dashboard flash
    "LIVE - receiving ECG data" with no MCU connected at all.
    """
    try:
        while True:
            _bridge_queue.get_nowait()
    except queue.Empty:
        pass


# Bumped on every demo start/stop. run_loop watches it and clears its own
# per-run locals, since clearing StreamAnalyzer and DashboardState alone still
# left run_loop grouping demo beats and live beats into one episode.
_stream_generation = 0


def request_stream_reset():
    global _stream_generation
    _stream_generation += 1


def read_bridge_batch():
    """
    Drain everything the Bridge has pushed since the last call.

    run_loop expects a pull-shaped read(): whatever is new, an empty array if
    nothing is, or None for end of stream. A live sensor has no end, so this
    never returns None; only selftest's file replay does.

    Values are raw ADC counts, no scaling to volts anywhere. Fine here, since
    every beat is z-normalised before the model sees it.
    """
    chunks = []
    try:
        while True:
            chunks.append(_bridge_queue.get_nowait())
    except queue.Empty:
        pass
    if not chunks:
        return np.zeros(0)
    flat = [v for batch in chunks for v in batch]
    return np.asarray(flat, dtype=np.float64)


# --------------------------------------------------------------------------
# Demo replay - show it working with no hardware attached
# --------------------------------------------------------------------------
# Feeds a bundled MIT-BIH clip into the same queue a real Bridge push lands in,
# so run_loop can't tell the difference. Needed because Start always launches
# live mode, and with no sketch flashed there's nothing to show.
#
# Records come from data/demo_records.json (written by
# scripts/export_demo_records.py) and are all test fold.


def load_demo_records():
    """Read the demo manifest. Falls back to the single bundled capture if
    it's missing or corrupt, so an older data/ still gives a working button."""
    path = APP_ROOT / "data" / "demo_records.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if records:
            return records
    except Exception as exc:
        print(f"[ECG] could not read {path.name} ({exc}); "
              f"falling back to the bundled single capture", flush=True)
    return [{"record": "119", "condition": "PVC", "file": SAMPLE.name,
             "fs": SELFTEST_SAMPLE_RATE, "seconds": 90,
             "note": "442 PVC in bigeminy"}]


DEMO_RECORDS = load_demo_records()
DEFAULT_DEMO_RECORD = "119"


def _demo_meta(record_id):
    for r in DEMO_RECORDS:
        if r["record"] == str(record_id):
            return r
    return DEMO_RECORDS[0]


def _demo_replay_worker(stop_event, meta):
    # Stream runs at MCU_SAMPLE_RATE but the clips are stored at 250 Hz, so
    # resample. resample_poly rather than data[::2] - plain slicing aliases QRS
    # energy down into the band the model reads.
    src_fs = int(meta.get("fs", SELFTEST_SAMPLE_RATE))
    data = np.loadtxt(APP_ROOT / "data" / meta["file"], delimiter=",",
                      dtype=np.float64)
    if src_fs != MCU_SAMPLE_RATE:
        from math import gcd
        g = gcd(src_fs, MCU_SAMPLE_RATE)
        data = resample_poly(data, MCU_SAMPLE_RATE // g, src_fs // g)

    chunk = max(1, MCU_SAMPLE_RATE // 20)
    i = 0
    while not stop_event.is_set():
        batch = data[i:i + chunk]
        if len(batch) == 0:
            i = 0                      # loop the capture - a demo running
            continue                   # unattended should not go quiet
        _enqueue(batch.tolist())
        i += chunk
        time.sleep(chunk / MCU_SAMPLE_RATE)


def _reset_stream(dashboard, stream):
    """Teardown for every mode switch. Whatever was running must leave nothing
    behind the next mode could pick up and report as its own: queued batches,
    the StreamAnalyzer buffer, run_loop's locals, dashboard history and motion
    windows."""
    global _motion_active
    _motion_active = False
    _motion_windows.clear()
    try:
        while True:
            _motion_log.get_nowait()
    except queue.Empty:
        pass
    drain_bridge_queue()
    request_stream_reset()
    if stream is not None:
        stream.reset()
    if dashboard is not None:
        dashboard.clear_history()


def start_live_capture(dashboard, stream=None):
    """Begin analysing the BioAmp sensor. Idempotent."""
    global _run_mode, _demo_stop
    if _run_mode == "live":
        return
    if _demo_stop is not None:
        _demo_stop.set()
    _reset_stream(dashboard, stream)
    _run_mode = "live"
    if dashboard is not None:
        dashboard.demo_record = None
        dashboard.gates = {c: C.LIVE_MIN_EPISODE_CONFIDENCE for c in C.CLASSES}
        dashboard.status = "waiting for the MCU sketch..."
        dashboard.source_desc = "LIVE via BioAmp EXG Pill / MCU Bridge"
    print("  [LIVE] capturing from BioAmp EXG Pill", flush=True)


def stop_capture(dashboard, stream=None):
    """The Stop button: stop whatever is running and analyse nothing. Exists so
    a demo record can be swapped without going through live mode, since
    stopping releases the dropdown."""
    global _run_mode, _demo_stop
    if _demo_stop is not None:
        _demo_stop.set()
    # Drain after setting the flag, since the worker may be mid-sleep and push
    # one more batch. run_loop's generation check catches anything that slips
    # through that window.
    _reset_stream(dashboard, stream)
    _run_mode = "idle"
    if dashboard is not None:
        dashboard.gates = {c: 0.0 for c in C.CLASSES}
        dashboard.status = "IDLE - choose Live Sensor or Demo Replay to start"
        dashboard.source_desc = "nothing running"
        dashboard.warmup_note = ""
    print("  [IDLE] stopped", flush=True)


def start_demo_replay(dashboard, stream=None, record=None):
    """Idempotent, a second click while running does nothing. Clears history
    first so a demo starts on a clean dashboard instead of appending onto
    whatever was already showing."""
    global _demo_stop, _run_mode
    if _demo_active():
        return
    meta = _demo_meta(record or DEFAULT_DEMO_RECORD)
    _demo_stop = threading.Event()
    _reset_stream(dashboard, stream)
    _run_mode = "demo"
    if dashboard is not None:
        dashboard.demo_record = meta["record"]
        # Demo replays database signal, so it's graded on the per-class MIT-BIH
        # map. Tell the page the same thing run_loop will use.
        dashboard.gates = {c: C.MIN_EPISODE_CONFIDENCE.get(c, 0.0)
                           for c in C.CLASSES}
        dashboard.status = (
            f"DEMO REPLAY - MIT-BIH record {meta['record']} "
            f"({meta['condition']}), NOT a live sensor")
        dashboard.source_desc = (
            f"DEMO replay of held-out MIT-BIH patient {meta['record']} - "
            f"{meta['note']} - not connected to a real sensor")
    print(f"  [DEMO] replaying record {meta['record']} "
          f"({meta['condition']}): {meta['note']}", flush=True)
    threading.Thread(target=_demo_replay_worker, args=(_demo_stop, meta),
                     daemon=True).start()


def stop_demo_replay(dashboard, stream=None):
    """Kept because callers and tests use the name. Stopping the demo now lands
    in idle rather than quietly starting live capture."""
    stop_capture(dashboard, stream)


def selftest_source():
    """Replay the bundled capture paced to real time, at its own 250 Hz rather
    than the MCU's 125."""
    data = np.loadtxt(SAMPLE, delimiter=",", dtype=np.float64)
    chunk = max(1, SELFTEST_SAMPLE_RATE // 20)     # ~50 ms batches
    state = {"i": 0}

    def read():
        if state["i"] >= len(data):
            return None
        time.sleep(chunk / SELFTEST_SAMPLE_RATE)
        out = data[state["i"]:state["i"] + chunk]
        state["i"] += chunk
        return out
    return read, f"bundled capture ({len(data) / SELFTEST_SAMPLE_RATE:.0f}s)"


# --------------------------------------------------------------------------
# Dashboard state - written by run_loop, read by the HTTP handler
# --------------------------------------------------------------------------
class DashboardState:
    """
    Thread-safe snapshot of what the dashboard shows.

    run_loop pushes into this as beats and episodes happen, and the HTTP
    handler reads a snapshot out of it on a different thread. One lock around
    list mutation and snapshotting is enough, since this is a display feed and
    not a hot path.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.mode = "selftest"
        self.source_desc = ""
        self.status = "starting"
        self.samples_seen = 0
        self.fs = MCU_SAMPLE_RATE
        self.started_at = time.time()
        self.beats = collections.deque(maxlen=300)
        self.episodes = collections.deque(maxlen=100)
        # Shown while nothing has been classified yet. Not an error state.
        self.warmup_note = ""
        # Sent to the browser so the waveform labels the same beats the table
        # shows. Needs the whole per-class map - a single number made the page
        # fade beats as "not reported" that the table then reported.
        self.gates = {c: 0.0 for c in C.CLASSES}
        # So the dropdown shows the right record after a reconnect or in a
        # second tab.
        self.demo_record = DEFAULT_DEMO_RECORD

    def clear_history(self):
        """Wipe beats, episodes and the sample count. Called on both demo start
        and stop, so demo results can't linger and be mistaken for sensor
        results, or the other way round."""
        with self._lock:
            self.beats.clear()
            self.episodes.clear()
            self.samples_seen = 0

    def push_beat(self, b):
        with self._lock:
            self.beats.append({"t": round(b.time_s, 3), "label": b.label,
                               "conf": round(b.confidence, 3)})

    def push_episode(self, e, annotated_truth=""):
        with self._lock:
            self.episodes.append({
                "label": e.label, "start": round(e.start_time, 2),
                "end": round(e.end_time, 2), "n_beats": e.n_beats,
                "hr": round(e.mean_hr, 1), "conf": round(e.mean_confidence, 3),
                "truth": annotated_truth,
            })

    def snapshot(self, stream):
        """Build the dict for /state.json. Reads the sample buffer off `stream`
        instead of copying it, since StreamAnalyzer already is the rolling
        buffer. No lock needed: push() swaps in a new array rather than
        mutating, so we always see a whole one."""
        buf = stream.buf
        fs = stream.fs
        n_show = int(DASHBOARD_WINDOW_S * fs)
        tail = buf[-n_show:] if len(buf) > n_show else buf
        t0 = (stream.offset + max(0, len(buf) - len(tail))) / fs
        now = (stream.offset + len(buf)) / fs

        with self._lock:
            beats = [b for b in self.beats if b["t"] >= now - DASHBOARD_WINDOW_S]
            episodes = list(self.episodes)[-25:]
            status, mode, source_desc = self.status, self.mode, self.source_desc
            samples_seen = self.samples_seen
            warmup_note = self.warmup_note
            gates = dict(self.gates)
            demo_record = self.demo_record

        return {
            "mode": mode, "source": source_desc, "status": status,
            "now": round(now, 3), "fs": fs, "t0": round(t0, 3),
            "samples_seen": samples_seen, "warmup": warmup_note,
            "gates": gates, "demo_record": demo_record,
            "run_mode": _run_mode,
            # Motion only means something for the live sensor. A demo replay is
            # a file, nothing is attached to anybody.
            "motion": bool(_motion_active) and _run_mode == "live",
            # Feeds the waveform marker (t, same sample-time axis as the wave)
            # and the Motion Log table (at, duration). Newest first.
            "motion_windows": ([
                {"t": w["sample_t"], "duration": w["duration"],
                 "at": datetime.fromtimestamp(w["end"]).strftime("%H:%M:%S")}
                for w in reversed(list(_motion_windows)[-20:])
            ] if _run_mode == "live" else []),
            # Live only: a beat below the gate shows as NOR instead of a faded
            # guess. Demo replay keeps the faded "?" labels as they were.
            "unconfident_as_nor": _run_mode == "live",
            "demo_records": DEMO_RECORDS,
            "wave": [round(float(v), 2) for v in tail.tolist()],
            "beats": beats[-60:],
            "episodes": list(reversed(episodes)),
        }


# --------------------------------------------------------------------------
# The analysis loop, same code for selftest and live
# --------------------------------------------------------------------------
def run_loop(read, stream, writer, fs, stop_event=None, dashboard=None,
            truth_lookup=None, episode_gate=None):
    """
    Pull from read() until it returns None (selftest hits end of file) or
    stop_event is set (live shutdown). How a beat becomes a logged episode
    lives here and only here, so selftest and live can't drift apart.

    dashboard, if given, is fed every new beat and finalised episode. It never
    changes what gets classified or logged, only what gets shown.

    episode_gate is the minimum mean confidence an episode needs. None keeps
    the per-class MIT-BIH map, which suits database signal; live mode passes
    C.LIVE_MIN_EPISODE_CONFIDENCE instead. Same analysis either way, only the
    reporting threshold changes, because the input is different.

    truth_lookup is selftest-only: the MIT-BIH annotation for a beat, shown
    next to the model's own call so a viewer can see it's right. Live has no
    ground truth, so it's None there.

    Returns (n_samples_seen, reported_episode_keys, compute_seconds).
    compute_seconds leaves out time spent waiting for data, so it answers
    "can the board keep up" rather than "how fast does the source deliver".
    """
    beats, reported = [], set()
    n, last_status = 0, time.time()
    compute_s = 0.0
    generation = _stream_generation

    while stop_event is None or not stop_event.is_set():
        # A demo start/stop invalidates everything so far. Caller clears the
        # stream and dashboard, these locals are ours. Without it the sample
        # count snapped back and beats from two sources landed in one episode.
        if _stream_generation != generation:
            generation = _stream_generation
            beats, reported = [], set()
            n = 0

        chunk = read()
        if chunk is None:
            break
        if len(chunk) == 0:
            time.sleep(0.005)
            continue

        # Demo feeds the same queue as a real push, so decide once here which
        # gate applies (the live gate would suppress real LBBB/AFIB on database
        # signal) and how the CSV row gets tagged.
        demo_active = _demo_active()
        demo_record_id = (dashboard.demo_record
                          if demo_active and dashboard is not None else "")
        tag = "[DEMO] " if demo_active else ""
        gate_now = None if demo_active else episode_gate
        # Shorter chunks during a demo so the table fills while someone watches.
        chunk_now = C.DEMO_MAX_EPISODE_S if demo_active else None

        n += len(chunk)
        _t = time.perf_counter()
        new_beats = stream.push(chunk)
        beats.extend(new_beats)
        if len(beats) > 400:
            beats = beats[-400:]
        episodes_now = group_episodes(beats, min_confidence=gate_now,
                                      max_episode_s=chunk_now)
        compute_s += time.perf_counter() - _t

        if dashboard is not None:
            # The placeholder is set once before the loop, so without this it
            # stays on "waiting" even after batches start arriving.
            if (_run_mode == "live"
                    and dashboard.status == "waiting for the MCU sketch..."):
                dashboard.status = "LIVE - receiving ECG data"
            # Nothing comes back until 12 R-peaks are buffered, so the first
            # ~10s is a waveform with no markers. Looks like the model is
            # ignoring obvious beats, so say what's happening instead.
            if not stream.emitted:
                dashboard.warmup_note = (
                    f"warming up - the model needs {C.RR_LOCAL_WINDOW + 2} "
                    f"beats of rhythm history before it can classify, so "
                    f"beats in the first few seconds stay unlabelled")
            else:
                dashboard.warmup_note = ""
            dashboard.samples_seen = n
            for b in new_beats:
                dashboard.push_beat(b)

        # Written from here and not the Bridge callback, so only one thread
        # ever touches `writer`. source=MOTION and condition=MOTION, so a
        # movement can't be read back later as a cardiac finding.
        while True:
            try:
                w = _motion_log.get_nowait()
            except queue.Empty:
                break
            writer.writerow([
                datetime.fromtimestamp(w["start"]).strftime(
                    "%Y-%m-%d %H:%M:%S"),
                "MOTION", "MOTION", "", "", w["duration"], "", "", "",
                "ECG paused by MCU - movement above threshold"])
            print("  [MOTION] logged %.1fs pause" % w["duration"], flush=True)

        for e in episodes_now:
            key = (e.label, int(e.start_sample))
            if key in reported:
                continue
            if (beats and e.end_sample >= beats[-1].sample
                    and e.duration < C.MAX_EPISODE_S):
                continue
            reported.add(key)
            flag = "low confidence" if e.low_confidence else ""
            truth = truth_lookup(e.start_sample) if truth_lookup else ""
            # Demo episodes do get logged, but the source column says DEMO so a
            # replay can't be read back as a sensor detection. Dropping them
            # made the CSV look broken during a demo, which was worse.
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (f"DEMO:{demo_record_id}" if demo_active else "LIVE"),
                e.label,
                round(e.start_time, 2), round(e.end_time, 2),
                round(e.duration, 2), e.n_beats, round(e.mean_hr, 1),
                round(e.mean_confidence, 3), flag])
            print(f"  {tag}{e.label:<5} {e.start_time:7.2f}s  "
                  f"{e.n_beats:>2} beat(s)  HR {e.mean_hr:5.1f}  "
                  f"conf {e.mean_confidence:.2f} {flag}", flush=True)
            if dashboard is not None:
                dashboard.push_episode(e, truth or "")

        if time.time() - last_status > 10:
            last_status = time.time()
            secs = n / fs
            load = compute_s / max(secs, 1e-6)
            print(f"  ... {secs:6.1f}s processed, {len(stream.emitted):>4} "
                  f"beats, {len(reported):>3} episodes, CPU load "
                  f"{load * 100:.1f}%", flush=True)

    return n, reported, compute_s


# --------------------------------------------------------------------------
# Web dashboard
# --------------------------------------------------------------------------
CLASS_COLOUR = {
    "NOR": "#1a9850", "LBBB": "#d73027", "RBBB": "#7b3294",
    "PVC": "#e08214", "AFIB": "#c51b7d",
}

DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<title>ECG Arrhythmia Classifier - Arduino UNO Q</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0f1115; color:#e6e6e6;
         font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:14px 20px; background:#161922; border-bottom:1px solid #262b38; }
  h1 { font-size:16px; margin:0 0 4px; }
  #status { font-size:12.5px; color:#9aa4b2; }
  #status b { color:#e6e6e6; }
  .badge { display:inline-block; padding:1px 8px; border-radius:10px;
           font-size:11px; font-weight:600; margin-left:6px; }
  .b-live { background:#7f1d1d; color:#fecaca; }
  .b-selftest { background:#1e3a5f; color:#bfdbfe; }
  main { padding:16px 20px; max-width:1100px; margin:0 auto; }
  canvas { width:100%; height:260px; background:#0b0d12; border:1px solid #262b38;
           border-radius:8px; display:block; }
  table { width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; }
  th,td { text-align:left; padding:6px 10px; border-bottom:1px solid #20242e; }
  th { color:#9aa4b2; font-weight:600; font-size:11.5px; text-transform:uppercase;
       letter-spacing:.04em; }
  .pill { padding:2px 8px; border-radius:10px; font-weight:700; font-size:12px;
          color:#0b0d12; }
  .lowconf { opacity:.55; }
  #empty { color:#6b7280; padding:14px 0; }
  footer { padding:16px 20px; color:#5b6472; font-size:11.5px; }
</style></head>
<body>
<header>
  <h1>ECG Arrhythmia Classifier<span id="modebadge" class="badge"></span></h1>
  <div id="status">connecting ...</div>
</header>
<main>
  <canvas id="wave" width="1000" height="260"></canvas>
  <table>
    <thead><tr><th>Class</th><th>Time</th><th>Beats</th><th>HR</th>
      <th>Confidence</th><th>Annotated</th></tr></thead>
    <tbody id="episodes"></tbody>
  </table>
  <div id="empty" style="display:none">No arrhythmia episodes detected yet.</div>
</main>
<footer>Rendering happens in this browser; every number above is computed on
  the Arduino UNO Q that served this page.</footer>
<script>
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

async function tick(){
  try{
    const r = await fetch('/state.json', {cache:'no-store'});
    const d = await r.json();
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
  }catch(err){
    document.getElementById('status').textContent = 'connection lost: ' + err;
  }
  setTimeout(tick, 400);
}
tick();
</script>
</body></html>
"""


def make_dashboard_handler(stream, dashboard):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass          # silence per-request logging; polls every 400ms

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = DASHBOARD_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/state.json":
                body = json.dumps(dashboard.snapshot(stream)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()
    return Handler


def start_local_preview_server(stream, dashboard, port):
    """
    Fallback for anywhere the App Lab framework doesn't exist, mainly a desktop
    check. This is how DashboardState and snapshot() were tested before ever
    touching the board, so it's kept working rather than replaced.

    A raw http.server is not reachable from outside on the board itself: the
    container gets its own private Docker network with no port published to the
    host (docker inspect shows Ports=map[]). See start_dashboard() for what
    does work there.
    """
    handler = make_dashboard_handler(stream, dashboard)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def start_dashboard(stream, dashboard, port):
    """
    Publish the dashboard so an outside browser can reach it.

    The `arduino:web_ui` Brick is the only thing on this platform that gets a
    port published from the container to the host. Searched all 156 app.yaml
    files in arduino/app-bricks-examples for a real `ports:` value and got zero
    hits; every example that serves a page uses this Brick instead.

    Client API, from the 08-web-ui-basics/02-data-transmission example (its
    files are copied into assets/libs/): ui.send_message(event, data) for
    server to browser, plus ui.on_connect / ui.on_disconnect. assets/app.js
    listens for "state" and uses the same draw and table code as the local
    preview page, copied rather than rewritten so a fix lands in both.

    Falls back to the local http.server preview when the Brick can't be
    imported, which is any off-board run.
    """
    try:
        from arduino.app_bricks.web_ui import WebUI
    except ImportError:
        return start_local_preview_server(stream, dashboard, port)

    ui = WebUI()

    def push_loop():
        while True:
            try:
                ui.send_message("state", dashboard.snapshot(stream))
            except Exception as e:
                print(f"[dashboard] send_message failed: {e}", flush=True)
            time.sleep(0.4)

    def on_connect(connection):
        # Send on connect too, so a browser opening mid-run doesn't wait 400ms
        # for its first picture.
        try:
            ui.send_message("state", dashboard.snapshot(stream))
        except Exception:
            pass

    ui.on_connect(on_connect)
    threading.Thread(target=push_loop, daemon=True).start()
    return ui


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # live and not selftest, because the Start button can't pass CLI arguments
    # to the container, so whatever this default is is what a demo runs.
    ap.add_argument("--mode", choices=["selftest", "live"], default="live")
    # No fixed default: selftest is 250 Hz and live is 125, so pick below
    # based on the mode unless the caller overrides it.
    ap.add_argument("--fs", type=int, default=None)
    ap.add_argument("--out", default=str(APP_ROOT / "detections.csv"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--port", type=int, default=DASHBOARD_PORT)
    ap.add_argument("--no-dashboard", action="store_true")
    args = ap.parse_args()
    if args.fs is None:
        args.fs = (SELFTEST_SAMPLE_RATE if args.mode == "selftest"
                   else MCU_SAMPLE_RATE)

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    print("=" * 66, flush=True)
    print("ECG ARRHYTHMIA CLASSIFIER - Arduino UNO Q", flush=True)
    print("=" * 66, flush=True)

    t = time.perf_counter()
    analyzer = ECGAnalyzer(str(MODEL))
    print(f"model loaded      {MODEL.name} "
          f"({MODEL.stat().st_size / 1024:.0f} KB) in "
          f"{(time.perf_counter() - t) * 1000:.0f} ms", flush=True)
    print(f"classes           {', '.join(C.CLASSES)}", flush=True)
    print(f"input rate        {args.fs} Hz -> resampled to {C.FS} Hz",
          flush=True)

    stream = StreamAnalyzer(analyzer, fs=args.fs)
    out_path = Path(args.out)
    new_file = not out_path.exists()
    fh = open(out_path, "a", newline="", buffering=1)
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["logged_at", "source", "condition", "start_s",
                         "end_s", "duration_s", "n_beats", "mean_hr_bpm",
                         "confidence", "flag"])

    print(f"detections ->     {out_path}", flush=True)
    print(f"warmup            {C.RR_LOCAL_WINDOW + 2} beats of rhythm history "
          f"needed before the first classification", flush=True)

    dashboard = None
    if not args.no_dashboard:
        dashboard = DashboardState()
        dashboard.mode = args.mode
        dashboard.fs = args.fs
        via = start_dashboard(stream, dashboard, args.port)
        via_desc = ("App Lab WebUI Brick - open this app in App Lab, or its "
                   "published URL"
                   if via.__class__.__name__ != "ThreadingHTTPServer" else
                   f"http://0.0.0.0:{args.port}/  (local preview only - "
                   f"confirmed NOT reachable from outside this container "
                   f"when run via apps_start; fine for a desktop check)")
        print(f"dashboard         {via_desc}", flush=True)

    result = {}
    stop_event = None

    try:
        if args.mode == "selftest":
            read, src = selftest_source()
            source_desc = f"SELF-TEST replaying {src} of MIT-BIH record 119"
            print(f"mode              SELF-TEST, no sensor required",
                  flush=True)
            print(f"source            {src} of MIT-BIH record 119",
                  flush=True)
            print(f"expectation       record 119 is a HELD-OUT patient with "
                  f"frequent PVCs;", flush=True)
            print(f"                  expect roughly 15-20 PVC episodes and "
                  f"no AFIB/LBBB/RBBB", flush=True)
            print("-" * 66, flush=True)
            if dashboard is not None:
                dashboard.status = "replaying MIT-BIH record 119"
                dashboard.source_desc = source_desc

            n, reported, compute_s = run_loop(read, stream, writer, args.fs,
                                              dashboard=dashboard)
            result.update(n=n, reported=reported, compute_s=compute_s)

        else:
            # Lazy import: only live mode needs the App Lab framework, so
            # selftest keeps working even off-board where this module does
            # not exist (e.g. a desktop sanity check).
            from arduino.app_utils import App, Bridge

            Bridge.provide("ecg_batch", _on_ecg_batch)
            # The sketch has been sending this since it was copied in, but
            # nothing was listening, so every movement was being discarded.
            Bridge.provide("motion_state",
                           lambda v: _on_motion_state(v, stream))
            print(f"mode              LIVE via MCU Bridge  "
                  f"(key 'ecg_batch', {MCU_SAMPLE_RATE} Hz, raw ADC)",
                  flush=True)
            print(f"                  waiting for the MCU sketch to start "
                  f"pushing batches ...", flush=True)
            print("-" * 66, flush=True)
            if dashboard is not None:
                # Start idle. Nothing is analysed until someone presses Live
                # Sensor or Demo Replay.
                dashboard.status = ("IDLE - choose Live Sensor or "
                                    "Demo Replay to start")
                dashboard.source_desc = "nothing running"
                dashboard.gates = {c: 0.0 for c in C.CLASSES}

            # Dashboard buttons. Only available with the real WebUI Brick, the
            # local preview fallback has no on_message. Registered here rather
            # than in start_dashboard() because they need the live-mode queue.
            if hasattr(via, "on_message"):
                def _record_of(data):
                    # The browser sends {"record": "214"}. Tolerate a bare
                    # string and fall back to the default on anything else, so
                    # a malformed message can't kill the demo.
                    if isinstance(data, dict):
                        return data.get("record")
                    if isinstance(data, str):
                        return data
                    return None

                via.on_message("start_demo",
                               lambda client, data=None:
                               start_demo_replay(dashboard, stream,
                                                 _record_of(data)))
                via.on_message("start_live",
                               lambda client, data=None:
                               start_live_capture(dashboard, stream))
                via.on_message("stop_all",
                               lambda client, data=None:
                               stop_capture(dashboard, stream))
                # Older page builds only knew "stop_demo". It now lands in idle
                # like the Stop button rather than starting live.
                via.on_message("stop_demo",
                               lambda client, data=None:
                               stop_capture(dashboard, stream))

            stop_event = threading.Event()

            def _worker():
                n, reported, compute_s = run_loop(
                    read_bridge_batch, stream, writer, args.fs,
                    stop_event=stop_event, dashboard=dashboard,
                    episode_gate=C.LIVE_MIN_EPISODE_CONFIDENCE)
                result.update(n=n, reported=reported, compute_s=compute_s)

            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()

            # App.run() blocks and is what dispatches Bridge pushes to our
            # callbacks, going by every working app on this board. Whether it
            # returns on Ctrl+C or swallows it isn't confirmed, but either way
            # is safe: the worker is a daemon thread and the CSV is
            # line-buffered, so at most the last row is at risk.
            App.run()
            stop_event.set()
            worker.join(timeout=5)

    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)
        if stop_event is not None:
            stop_event.set()
    finally:
        fh.close()

    n = result.get("n", 0)
    reported = result.get("reported", set())
    compute_s = result.get("compute_s", 0.0)
    secs = n / args.fs
    # Paced sources sleep between reads, so a real-time factor would sit just
    # under 1.0 however fast the board is and look like a false overload
    # warning. CPU load (compute time over signal time, waits excluded) is the
    # number that actually answers whether the board keeps up.
    load = compute_s / max(secs, 1e-6)

    print("-" * 66, flush=True)
    print(f"RESULT  {len(reported)} episodes from {len(stream.emitted)} beats "
          f"over {secs:.0f}s of signal", flush=True)
    print(f"        CPU load {load * 100:.1f}%  "
          f"({compute_s:.1f}s of compute per {secs:.0f}s of ECG)", flush=True)

    if args.mode == "selftest":
        labels = {}
        for lbl, _ in reported:
            labels[lbl] = labels.get(lbl, 0) + 1
        print(f"        breakdown: {labels or 'none'}", flush=True)
        ok = 10 <= len(reported) <= 30 and set(labels) <= {"PVC"}
        print(f"        SELF-TEST {'PASSED' if ok else 'NEEDS A LOOK'} - "
              f"expected 15-20 PVC-only episodes", flush=True)
        if load > 0.8:
            print(f"        WARNING: CPU load {load * 100:.0f}% leaves "
                  f"little headroom. Try --threads 3.", flush=True)
    print(f"        wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
