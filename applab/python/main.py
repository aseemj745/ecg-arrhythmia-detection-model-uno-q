"""
ECG Arrhythmia Classifier - Arduino App Lab entry point (Arduino UNO Q).

Runs the INT8 ONNX arrhythmia model on the QRB2210 Linux side, and serves a
live web dashboard on the board's own HTTP port so the computation can be
watched from any browser (PC, phone) while it is provably happening on the
board, not the viewing device.

TWO MODES
---------
selftest (default via --mode selftest)  Replays a bundled 90 s recording of
                    MIT-BIH record 119 - a patient the model was NEVER
                    trained on - resampled to 250 Hz during dataset prep.
                    This is unrelated to the real MCU's sample rate (see
                    below); the replay paces and resamples itself
                    independently via SELFTEST_SAMPLE_RATE. Needs NO sensor
                    and NO electrodes. Run this first: it proves the model
                    executes correctly on ARM, separately from any question
                    about wiring.

live                Reads samples pushed by the MCU over the App Lab Bridge.

                    The Bridge is PUSH, not pull: the MCU sketch calls
                    Bridge.notify("ecg_batch", batch) roughly every 100 ms
                    (~12-13 raw ADC ints @ 125 Hz - observed batch sizes on
                    real hardware run 14-22, since the push interval is not
                    exactly 100ms), which the App Lab framework
                    delivers to whatever function was registered with
                    Bridge.provide("ecg_batch", ...). Confirmed against the
                    working ecg-monitor and ecg_uno_q_project apps on this
                    board - both use that exact key.

                    Because it is push, not pull, the callback (_on_ecg_batch)
                    just drops each batch into a thread-safe queue and
                    returns immediately; read_bridge_batch() drains that
                    queue and hands the analysis loop exactly the same shape
                    of data selftest mode does, so run_loop() itself does not
                    know or care which mode it is running under.

                    App.run() is what every working Bridge app on this board
                    ends with, so it is presumably what actually dispatches
                    Bridge pushes to registered callbacks, and it blocks.
                    Because of that, the analysis loop runs on a background
                    thread in live mode, and App.run() owns the main thread -
                    matching the pattern every known-good app on this board
                    uses, rather than guessing at a different one.

    python3 python/main.py --mode selftest # no hardware needed, run this first
    python3 python/main.py                 # LIVE is the default (see below)

Default is --mode live, not selftest, because App Lab's Start button
(apps_start) cannot pass arguments to the container - confirmed by reading
its own tool schema, which accepts only an app id, nothing else. Whatever
mode is the default is what a demo actually runs. Always verify with
--mode selftest explicitly from a terminal first.

WEB DASHBOARD
-------------
A plain stdlib http.server (no new dependency) runs alongside the analysis
loop in EITHER mode, so it can be tested today with selftest replay data,
before any sketch or electrodes exist. Open http://<host>:<port>/ (default
port 8080) in a browser: scrolling waveform, live beat labels, episode
table. All computation stays on whichever machine runs this script; the
browser is just a display - this is the point, it is what makes "open a
browser and watch it happen" real evidence of on-device execution rather
than a claim.
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

# The MCU sketch samples A0 at 125 Hz. Originally assumed 250 Hz from a
# stale comment in the .ino; corrected 2026-08-20 after two independent
# confirmations: (1) the literal FS_HZ constant in ecg-monitor's sketch.ino
# is 125, and (2) a real BioAmp EXG Pill capture's own timestamps measured
# 124.96 Hz empirically (2950 samples over 23.60 real seconds). Running our
# Pan-Tompkins detector on that same capture at 125 Hz gave a physiologically
# plausible ~103 bpm; at the old (wrong) 250 Hz assumption it gave ~200 bpm -
# every downstream heart rate and duration was being computed exactly 2x too
# fast. If this ever changes again, change it here too - everything
# downstream is derived from this number, and getting it wrong does not
# crash anything, it just silently scales every heart rate and duration.
MCU_SAMPLE_RATE = 125

# Native rate of the bundled MIT-BIH selftest/demo capture (sample_ecg_250hz.csv),
# resampled to 250 Hz during dataset prep. Deliberately a SEPARATE constant
# from MCU_SAMPLE_RATE now that the two are known to differ - selftest_source()
# paces/interprets the file at its own true rate; _demo_replay_worker()
# decimates it down to MCU_SAMPLE_RATE before feeding it into the live-mode
# stream, since that shared stream is created with fs=MCU_SAMPLE_RATE.
SELFTEST_SAMPLE_RATE = 250

# Batch size the MCU sends per Bridge.notify() call: BATCH_SIZE in
# sketch/sketch.ino is 25, which at 125 Hz is 200 ms of ECG per push.
# (The "Batch size: 14-22" numbers in the sketch's serial debug output are
# the PARTIAL batch still being filled at the moment of the 1 Hz debug
# print, not the size that gets sent.) Not load-bearing anywhere below -
# the queue-based drain in read_bridge_batch() works for any batch size -
# but documented here since it is what "one push" means when reading logs.
MCU_BATCH_SIZE = 25

DASHBOARD_PORT = 8080
DASHBOARD_WINDOW_S = 8.0     # seconds of waveform shown on the page


# --------------------------------------------------------------------------
# Live: Bridge push -> thread-safe queue -> pull-shaped read()
# --------------------------------------------------------------------------
_bridge_queue: "queue.Queue" = queue.Queue()

# Set while "Start Demo Replay" is running. Declared here (above
# _on_ecg_batch, which reads it) rather than next to the demo code below,
# because the real Bridge callback has to know to stand aside.
_demo_stop: "threading.Event | None" = None

# Which source, if any, currently owns the analysis stream.
#
#   "idle"  nothing is being analysed. THIS IS THE START STATE. The MCU
#           sketch pushes from the moment the app boots, so starting in
#           "live" meant a session was always already running against
#           whatever the electrodes happened to be doing before anyone had
#           chosen anything - the first thing on screen was a half-warmed-up
#           run nobody asked for, and stopping it was not even possible.
#   "live"  BioAmp EXG Pill via the MCU Bridge.
#   "demo"  MIT-BIH replay from the bundled CSVs.
#
# Exactly one owner at a time. That single-owner rule is what keeps the two
# signal sources out of each other's buffer - see _on_ecg_batch.
_run_mode = "idle"


# Motion state, pushed by the sketch as Bridge.notify("motion_state", 0/1).
#
# The MCU already owns the decision: it samples the MPU6050 at 50 Hz,
# compares |accel| against a slowly-adapting baseline (MOTION_THRESHOLD_G,
# debounced MOTION_DEBOUNCE_COUNT samples) and, while motion is present,
# stops pushing ecg_batch at all. So by the time anything reaches this
# process the motion-corrupted samples are already gone - Linux is not
# filtering here, it is REPORTING what the MCU decided, which is the only
# honest way to describe it on screen.
#
# Deliberately NOT resetting StreamAnalyzer when motion ends. A pause makes
# the samples either side of it adjacent, and the reflex is to treat that
# splice as dangerous - but measured on record 234 with five pauses of
# 1.5-4 s, every one of the 106 beats still came back NOR (mean confidence
# 0.63) and no episode was reported even with the gate off. RR_CLIP drops
# the single anomalous interval and the 10-beat window absorbs the rest.
# Resetting would instead cost ~10 s of RR warmup blindness after every
# movement, which is a real loss to buy nothing.
# Motion pauses shorter than this are not logged - see _on_motion_state.
MIN_MOTION_LOG_S = 0.30

_motion_active = False
_motion_started_at = 0.0
_motion_windows = collections.deque(maxlen=200)   # completed, for the UI
_motion_log = queue.Queue()                       # completed, for the CSV


def _on_motion_state(value, stream=None):
    """Bridge.provide("motion_state", ...) - edge-triggered by the sketch.

    Runs on the framework's dispatch thread like _on_ecg_batch, so it does
    the minimum: flip a flag, and hand a completed window to a queue for
    run_loop to write. Wall-clock is the right clock for DURATION - during
    motion no samples arrive, so there is no sample counter to measure the
    pause length with.

    `stream` (StreamAnalyzer) is passed only to read where, in SAMPLE time,
    the pause happened - `stream.offset + len(stream.buf)` is the sample
    index of the very next sample that will arrive, which is exactly the
    splice point: the buffer has no gap in it (no samples were ever missing
    from the array, motion just means fewer of them arrived), so this index
    is simultaneously "last sample before" and "first sample after" the
    pause. That is what lets the browser place a marker on the waveform at
    the right x position - the wave axis is sample time, not wall time.
    """
    global _motion_active, _motion_started_at
    # Only meaningful while the live sensor owns the stream. The IMU keeps
    # reporting during a demo replay or in idle - somebody can always bump
    # the bench - but what is being analysed then is a file, so recording
    # "ECG paused by movement" against it would be false. Caught by the
    # motion test: a bump during a demo left a phantom window that then
    # showed up in the next live session.
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
            # Too short to be worth a row. The gate still fired and the
            # banner still showed - this only decides what is worth
            # RECORDING. A floor here is belt-and-braces next to the
            # sketch's slow-release debounce: if a board is still running
            # older firmware with the symmetric debounce, this alone keeps
            # the Motion log and detections.csv readable instead of a wall
            # of sub-0.2s rows.
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
    """The demo's way into the analysis queue.

    Deliberately NOT _on_ecg_batch(): that now drops everything while a demo
    is active, which would make the demo silence itself.
    """
    _bridge_queue.put(samples)


def _on_ecg_batch(samples):
    """
    Registered as Bridge.provide("ecg_batch", _on_ecg_batch).

    Called by the App Lab framework on whatever thread it dispatches Bridge
    pushes on - not necessarily our own. Must never block and never raise,
    so it does nothing but validate the shape and drop the batch in a
    thread-safe queue; all real work happens later, off this callback.

    Unless live mode is the current owner of the stream this DISCARDS the
    batch. The MCU sketch streams continuously the moment the app starts,
    so without this gate the sensor's samples end up interleaved with the
    demo's MIT-BIH samples in one buffer - alternating clean and noisy
    segments that make the waveform look corrupted and drop the model to
    near-chance confidence - and, in idle, the app analyses a stream nobody
    asked it to analyse. (Neither was possible before the app shipped its
    own sketch; until then nothing was pushing.) Live is an explicit
    choice, so the sensor is heard only once it has been made.
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
    Throw away anything already queued but not yet analysed.

    Needed when switching between demo replay and real sensor data: both
    sources feed the SAME _bridge_queue via _on_ecg_batch(), so without this
    a demo that was just stopped leaves batches sitting in the queue which
    run_loop() then drains and reports as if they had come from the sensor.
    That is what made the dashboard flash "LIVE - receiving ECG data" with
    thousands of samples while no MCU was connected at all.
    """
    try:
        while True:
            _bridge_queue.get_nowait()
    except queue.Empty:
        pass


# Bumped on every demo start/stop. run_loop() watches it and throws away its
# own local per-run state (sample count, beat list, reported-episode set)
# when it changes - those live in locals, so clearing StreamAnalyzer and
# DashboardState alone would still leave run_loop happily grouping demo
# beats together with live ones into a single episode.
_stream_generation = 0


def request_stream_reset():
    global _stream_generation
    _stream_generation += 1


def read_bridge_batch():
    """
    Drain everything the Bridge has pushed since the last call.

    The analysis loop expects a pull-shaped read(): call it, get whatever is
    newly available, an empty array if nothing is, or None to mean "the
    stream has ended". A live sensor has no "end", so this never returns
    None - only selftest's file replay does that, at end of file.

    Units are raw, unconverted ADC counts (the sketch does a bare
    analogRead(A0), no scaling to volts on either side of the Bridge). That
    is fine here: every beat is z-normalised and polarity-normalised before
    the model sees it, so absolute amplitude never mattered.
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
# Demo replay - a "show it moving with no hardware attached" button
# --------------------------------------------------------------------------
# apps_start always launches --mode live (see the default-mode decision
# above), and with no MCU sketch flashed yet the dashboard has nothing to
# show but "waiting for the MCU sketch...". A teammate's own dashboard app
# solves this the same way: it has a simulated ECG data source alongside
# the real one, so it can be demoed with zero hardware. This is that same
# idea, wired onto our dashboard's own "Start Demo Replay" button.
#
# It works by feeding a bundled MIT-BIH capture into the same analysis queue
# a real Bridge.notify("ecg_batch", ...) push lands in, so from run_loop()'s
# point of view this is indistinguishable from a real sensor.
#
# Any of the records in data/demo_records.json can be chosen from the
# dashboard dropdown. Every one is TEST FOLD - a patient the model never
# trained on - which is what makes a correct detection here evidence of
# anything. scripts/export_demo_records.py writes them, and the split guard
# in step1b_check_split.py is what actually enforces the test-only rule.


def load_demo_records():
    """Read the demo manifest. Missing/corrupt -> just the legacy capture,
    so an older data/ directory still gives a working demo button."""
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
    # This feeds the SAME live-mode stream real Bridge pushes go into (that
    # stream is created with fs=MCU_SAMPLE_RATE), but the bundled captures
    # are stored at their own rate (250 Hz). Resample to MCU_SAMPLE_RATE
    # here so what arrives is already at the rate the stream expects -
    # otherwise the demo plays at the wrong speed and 250 Hz-shaped beats go
    # through a filter tuned for 125 Hz, warping every waveform the model
    # sees. resample_poly, NOT data[::2]: plain slicing aliases high
    # -frequency QRS energy down into the band the model reads.
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
    """Common teardown for every mode switch.

    Both directions matter: whatever was running must leave nothing behind
    that the next mode could pick up and report as its own - queued
    batches, StreamAnalyzer buffer, run_loop's locals, dashboard history,
    and motion windows.
    """
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
    """Return to idle: stop whatever is running and analyse nothing.

    This is the Stop button. It exists so a demo record can be swapped
    without going through live mode - stopping releases the dropdown, and
    the next Start picks up the newly chosen record.
    """
    global _run_mode, _demo_stop
    if _demo_stop is not None:
        _demo_stop.set()
    # The demo worker may be mid-sleep and push one more batch after the
    # flag is set, so drain AFTER setting it; run_loop's generation check
    # discards anything that still slips through that window.
    _reset_stream(dashboard, stream)
    _run_mode = "idle"
    if dashboard is not None:
        dashboard.gates = {c: 0.0 for c in C.CLASSES}
        dashboard.status = "IDLE - choose Live Sensor or Demo Replay to start"
        dashboard.source_desc = "nothing running"
        dashboard.warmup_note = ""
    print("  [IDLE] stopped", flush=True)


def start_demo_replay(dashboard, stream=None, record=None):
    """Idempotent: a second click while already running does nothing.

    Clears prior history first (both directions - see clear_history's
    docstring) so a demo run starts from a clean dashboard rather than
    appending onto whatever real (or previous demo) beats/episodes were
    already showing.
    """
    global _demo_stop, _run_mode
    if _demo_active():
        return
    meta = _demo_meta(record or DEFAULT_DEMO_RECORD)
    _demo_stop = threading.Event()
    _reset_stream(dashboard, stream)
    _run_mode = "demo"
    if dashboard is not None:
        dashboard.demo_record = meta["record"]
        # Demo replays database signal, so it is graded on the per-class
        # MIT-BIH map - tell the page the same thing run_loop will use.
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
    """Kept as a name because callers and tests use it; stopping the demo
    now lands in idle rather than silently starting live capture."""
    stop_capture(dashboard, stream)


def selftest_source():
    """Replay the bundled capture, paced to real time, at its own native
    rate (SELFTEST_SAMPLE_RATE) - independent of MCU_SAMPLE_RATE, since the
    two are known to differ."""
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
    Thread-safe snapshot of what the web dashboard shows.

    run_loop() pushes into this as beats and episodes happen; the HTTP
    handler (a different thread, one per request) reads a consistent
    snapshot out of it to answer /state.json. A single lock around list
    mutation and snapshotting is enough here - this is a display feed, not
    a hot path, and CPU load stays under 1% either way (see the printed
    CPU-load line, which measures the real analysis work, not this).
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
        # Shown while the model has not classified anything yet. Not an
        # error state - see run_loop for why the first beats are unlabelled.
        self.warmup_note = ""
        # Confidence an episode needs to be reported. Sent to the browser so
        # the waveform can label exactly the beats the table would report -
        # otherwise the page shows PVC/AFIB/RBBB text on beats that never
        # appear in the episode list, which reads as the table dropping
        # detections when it is really the gate doing its job.
        # Per-class, because that is how the gate actually works in demo
        # mode (MIN_EPISODE_CONFIDENCE: PVC 0.50, LBBB/RBBB/AFIB 0.0). A
        # single number here was wrong: the page was told 0.55 while the
        # backend grouped demo episodes on the per-class map, so beats were
        # drawn faded as "not reported" that the table then reported, and
        # vice versa. Sending the whole map keeps the two in step.
        self.gates = {c: 0.0 for c in C.CLASSES}
        # Which demo record is playing, so the dropdown can show the right
        # selection after a reconnect or a second browser opening the page.
        self.demo_record = DEFAULT_DEMO_RECORD

    def clear_history(self):
        """Wipe beats/episodes/samples_seen. Called on every demo-replay
        start AND stop, so demo results can never linger and get mistaken
        for real sensor results once demo is stopped (or vice versa) - the
        two data sources share the same _on_ecg_batch()/stream/dashboard
        objects with nothing else distinguishing them after the fact."""
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
        """
        Build the JSON-able dict for /state.json. `stream` is read for its
        raw sample buffer here rather than kept inside DashboardState,
        because StreamAnalyzer already IS the rolling buffer - copying
        samples into a second buffer on every push() would be pure
        duplication. Reading `stream.buf` from another thread is safe
        without a lock: push() replaces the attribute with a brand new
        array (np.concatenate never mutates the old one in place), so a
        reference taken here either sees the array whole, before or after a
        given update, never partially written.
        """
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
            # Motion is only meaningful for the live sensor - a demo replay
            # is a file, nothing is attached to anybody.
            "motion": bool(_motion_active) and _run_mode == "live",
            # Feeds three things on the page: the vertical marker on the
            # waveform (t, in the same sample-time axis as wave/beats), and
            # the Motion Log table (at, duration). Newest first, like
            # episodes below. Live only - see _on_motion_state.
            "motion_windows": ([
                {"t": w["sample_t"], "duration": w["duration"],
                 "at": datetime.fromtimestamp(w["end"]).strftime("%H:%M:%S")}
                for w in reversed(list(_motion_windows)[-20:])
            ] if _run_mode == "live" else []),
            # Live only: a beat the model is not confident enough about to
            # report is shown as NOR rather than as a faded guess. See the
            # comment in app.js draw() for why that is consistency rather
            # than whitewashing. Demo replay keeps the faded "?" labels
            # exactly as they were.
            "unconfident_as_nor": _run_mode == "live",
            "demo_records": DEMO_RECORDS,
            "wave": [round(float(v), 2) for v in tail.tolist()],
            "beats": beats[-60:],
            "episodes": list(reversed(episodes)),
        }


# --------------------------------------------------------------------------
# The one analysis loop - identical for selftest and live
# --------------------------------------------------------------------------
def run_loop(read, stream, writer, fs, stop_event=None, dashboard=None,
            truth_lookup=None, episode_gate=None):
    """
    Pull from `read()` until it returns None (selftest: end of file) or
    `stop_event` is set (live: shutdown requested). Everything about how a
    beat becomes a logged episode lives here, once, so selftest and live
    are provably running the same code and can never quietly drift apart.

    `dashboard`, if given, is fed every new beat and finalised episode so
    the web page can show the same thing this function is doing - it never
    changes what gets classified or logged, only what gets displayed.

    `episode_gate` is the minimum mean confidence an episode needs before it
    is reported at all. None keeps the per-class MIT-BIH-tuned map, which is
    right for selftest/demo replay of database signal. Live mode passes
    C.LIVE_MIN_EPISODE_CONFIDENCE instead - see that constant for the
    measurements behind it. Same analysis code either way; only the
    reporting threshold differs, because the input domain differs.

    `truth_lookup(sample) -> str|None` is selftest-only: the MIT-BIH
    annotation for a beat, shown on the dashboard next to the model's own
    call purely so a viewer can see the model is right, same as the desktop
    GUI already does. Live mode has no ground truth, so this is None there.

    Returns (n_samples_seen, reported_episode_keys, compute_seconds).
    compute_seconds excludes time spent waiting for data (sleep/pacing), so
    it is the number that answers "can the board keep up", not a mix of
    that question and however fast the source happens to deliver samples.
    """
    beats, reported = [], set()
    n, last_status = 0, time.time()
    compute_s = 0.0
    generation = _stream_generation

    while stop_event is None or not stop_event.is_set():
        # A demo start/stop invalidates everything accumulated so far.
        # StreamAnalyzer and DashboardState are cleared by the caller; these
        # locals are ours to clear, and without doing so the sample counter
        # would snap straight back to its pre-reset value and beats from two
        # different sources could be grouped into one episode.
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

        # Demo replay feeds the same queue a real Bridge push does, so this
        # loop cannot otherwise tell a demo-generated episode from a real
        # one. Two things follow from which source is active, so decide once
        # here, before the beats are grouped:
        #
        #  * which confidence gate applies. episode_gate is the strict
        #    live-input gate; demo replay is MIT-BIH database signal, the
        #    exact domain the per-class map in config was measured on, so
        #    applying the live gate to it would suppress real findings
        #    (LBBB and AFIB sit below it by design - see
        #    MIN_EPISODE_CONFIDENCE).
        #  * whether it reaches detections.csv. That file should only ever
        #    hold genuine sensor detections, so demo episodes are printed
        #    and displayed - clearly tagged - but never written.
        demo_active = _demo_active()
        demo_record_id = (dashboard.demo_record
                          if demo_active and dashboard is not None else "")
        tag = "[DEMO] " if demo_active else ""
        gate_now = None if demo_active else episode_gate
        # Sustained rhythms finalise in smaller slices during a demo so the
        # table fills while someone is watching - see DEMO_MAX_EPISODE_S.
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
            # The initial "waiting for the MCU sketch..." placeholder is set
            # once before this loop starts and is otherwise never touched -
            # without this, the label stays stuck at "waiting" forever even
            # once real batches are flowing, which reads as broken when it
            # is not (samples_seen/beats/episodes below all update fine
            # regardless - only this text lagged behind).
            if (_run_mode == "live"
                    and dashboard.status == "waiting for the MCU sketch..."):
                dashboard.status = "LIVE - receiving ECG data"
            # Until 12 R-peaks (RR_LOCAL_WINDOW + 2) are in the buffer,
            # analyse() returns NO beats at all - the rhythm-feature branch
            # of the model has no RR history to work from yet. So the first
            # ~10 s of any run (or any run restarted by a demo start/stop)
            # draws a waveform with zero markers, which looks exactly like
            # the model ignoring obvious beats. Say so instead.
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

        # Motion windows are written from here rather than from the Bridge
        # callback so that exactly one thread ever touches `writer`. They
        # get source=MOTION and condition=MOTION, so a movement can never
        # be read back as a cardiac finding - it is the opposite, a record
        # of when the MCU refused to supply cardiac data at all.
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
            # Demo episodes ARE logged, but the `source` column says DEMO
            # so a replay of a database patient can never be read back as a
            # real sensor detection. Silently dropping them made the CSV
            # look broken during a demo, which is worse than logging them
            # honestly labelled.
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
    Fallback for anywhere the App Lab framework does not exist (a desktop
    sanity check, mainly). This IS how the dashboard's data logic
    (DashboardState, snapshot()) was actually verified end-to-end before
    ever touching the board: launched as a real process, polled from
    outside like a browser would, confirmed against the known-correct CLI
    output. Kept working, not replaced, because it is the only way any of
    this can be tested off-board at all.

    Confirmed on the board that a raw http.server here is NOT externally
    reachable: the container runs on its own private Docker bridge network
    with no port published to the host (checked via `docker inspect` -
    Ports=map[], PortBindings=map[] - and the same is true of a teammate's
    own HTTP-dashboard app, so this isn't specific to our setup). See
    start_dashboard() below for what actually works on the board.
    """
    handler = make_dashboard_handler(stream, dashboard)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def start_dashboard(stream, dashboard, port):
    """
    Publish the dashboard so an external browser can actually reach it.

    The App Lab `arduino:web_ui` Brick is the only mechanism on this
    platform confirmed to get a port published from the container to the
    host - checked directly: searched all 156 app.yaml files in
    arduino/app-bricks-examples for any use of the (empty, in both our
    app.yaml and a teammate's) `ports:` field with a real value - zero
    hits. Every example that serves a browser page uses this Brick.

    Client-side API (from arduino/app-bricks-examples,
    08-web-ui-basics/02-data-transmission, files copied verbatim into
    assets/libs/): `ui.send_message(event, data)` server -> browser,
    `ui.on_connect(cb)` / `ui.on_disconnect(cb)`. assets/app.js listens for
    the "state" event and calls the SAME draw()/table-render code the local
    preview server's page uses - copied, not reimplemented, so a bug fixed
    in one is fixed in both.

    Falls back to the local http.server preview when the Brick cannot be
    imported (any off-board run) - this is what made local testing of
    DashboardState/snapshot() possible at all before ever reaching the
    board. The WebUI Brick call itself could not be executed off-board;
    this is the one piece of Task 2.5 that is genuinely first-run-on-board.
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
        # Send immediately on connect too, so a browser that opens the page
        # mid-run does not have to wait up to 400ms for its first picture.
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
    # Default is "live", not "selftest": App Lab's apps_start (the Start
    # button) has no mechanism to pass CLI arguments to a container -
    # confirmed by reading its own tool schema, which takes only an app id -
    # so whatever runs when a demo presses Start is whatever this default
    # is. That should be live monitoring, not a self-test replay. Run
    # selftest explicitly with --mode selftest from a terminal instead.
    ap.add_argument("--mode", choices=["selftest", "live"], default="live")
    # No static default: selftest's bundled capture and live's real Bridge
    # data are at different rates (250 Hz vs 125 Hz) - pick the right one
    # for whichever mode actually ran, below, unless the caller overrides it.
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
            # The sketch has been notifying this since the day it was
            # copied in; nothing was listening, so every movement the
            # MPU6050 caught was discarded here.
            Bridge.provide("motion_state",
                           lambda v: _on_motion_state(v, stream))
            print(f"mode              LIVE via MCU Bridge  "
                  f"(key 'ecg_batch', {MCU_SAMPLE_RATE} Hz, raw ADC)",
                  flush=True)
            print(f"                  waiting for the MCU sketch to start "
                  f"pushing batches ...", flush=True)
            print("-" * 66, flush=True)
            if dashboard is not None:
                # Start idle. Nothing is analysed until someone presses
                # Live Sensor or Demo Replay - see _run_mode.
                dashboard.status = ("IDLE - choose Live Sensor or "
                                    "Demo Replay to start")
                dashboard.source_desc = "nothing running"
                dashboard.gates = {c: 0.0 for c in C.CLASSES}

            # "Start Demo Replay" button on the dashboard - only meaningful
            # when the real WebUI Brick is in use (`via` is the real object,
            # not the local http.server preview fallback, which has no
            # on_message). Registered here rather than inside
            # start_dashboard() because it needs _on_ecg_batch's queue,
            # which is specific to live mode.
            if hasattr(via, "on_message"):
                def _record_of(data):
                    # The browser sends {"record": "214"}; tolerate a bare
                    # string, and fall back to the default on anything else
                    # rather than letting a malformed message kill the demo.
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
                # Older page builds only knew "stop_demo"; it now lands in
                # idle like the Stop button rather than starting live.
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

            # App.run() blocks and, going by every working Bridge app on
            # this board, is what actually dispatches pushes to
            # _on_ecg_batch. Whether it returns on Ctrl+C or swallows it is
            # not confirmed - if it swallows it, the process just exits
            # (worker is a daemon thread, so it cannot hang shutdown; the
            # CSV is line-buffered, so at most the very last row is at
            # risk). If it does raise back to us, the except block below
            # gives a clean stop and an accurate final report instead.
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
    # Paced sources (selftest) sleep between reads, so a "real time factor"
    # would sit just under 1.0 no matter how fast the board is and would
    # look like a false overload warning. CPU load - compute time divided
    # by signal time, with wait time excluded - is the number that actually
    # answers "can the board keep up".
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
