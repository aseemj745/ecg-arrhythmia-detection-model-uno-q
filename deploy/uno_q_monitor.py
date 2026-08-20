"""
Headless arrhythmia monitor for the Arduino UNO Q (QRB2210 Linux side).

This is the on-device counterpart of gui_app.py. No GUI, no matplotlib, no
PyTorch, no wfdb - just numpy, scipy and onnxruntime reading the INT8 model.

    # from the STM32 bridge over serial
    python3 uno_q_monitor.py --serial /dev/ttyACM0 --fs 250

    # from a bridge script that prints samples to stdout
    python3 /app/python/adc_reader_uno_q_bridge.py | python3 uno_q_monitor.py --stdin --fs 250

    # replay a captured CSV, to prove the board works with no sensor attached
    python3 uno_q_monitor.py --csv capture.csv --fs 250

Every detected episode is appended to a CSV immediately (line-buffered, so a
power cut costs at most the current line) and printed to stdout so it shows
up in journalctl if you run this as a service.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ecg import config as C                                   # noqa: E402
from ecg.pipeline import ECGAnalyzer, StreamAnalyzer, group_episodes  # noqa: E402


# --------------------------------------------------------------------------
def open_input(args):
    """Return a zero-argument callable giving the next batch of samples."""
    if args.stdin:
        def read():
            line = sys.stdin.readline()
            if not line:
                return None                      # upstream closed
            tok = line.replace(",", " ").split()
            try:
                return np.array([float(tok[0])])
            except (ValueError, IndexError):
                return np.zeros(0)
        return read, "stdin"

    if args.csv:
        data = np.loadtxt(args.csv, delimiter=",", usecols=args.csv_column,
                          skiprows=args.skip_rows, dtype=np.float64)
        state = {"i": 0, "t": time.perf_counter()}
        chunk = max(1, int(args.fs / 20))

        def read():
            if state["i"] >= len(data):
                return None
            # pace it like real time so timings in the log are meaningful
            time.sleep(chunk / args.fs)
            out = data[state["i"]:state["i"] + chunk]
            state["i"] += chunk
            return out
        return read, f"csv {args.csv}"

    import serial                                 # pyserial
    ser = serial.Serial(args.serial, args.baud, timeout=0.05)
    buf = {"partial": ""}

    def read():
        raw = ser.read(4096).decode("utf-8", errors="ignore")
        if not raw:
            return np.zeros(0)
        buf["partial"] += raw
        *lines, buf["partial"] = buf["partial"].split("\n")
        out = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(float(ln.replace(",", " ").split()[0]))
            except (ValueError, IndexError):
                continue                          # banner / status line
        return np.asarray(out, dtype=np.float64)
    return read, f"serial {args.serial}@{args.baud}"


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--serial", help="e.g. /dev/ttyACM0")
    src.add_argument("--stdin", action="store_true")
    src.add_argument("--csv", help="replay a captured CSV")

    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fs", type=int, default=250,
                    help="sample rate of the INPUT; resampled to 360 Hz")
    ap.add_argument("--csv-column", type=int, default=0)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--out", default="detections.csv")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, default=2,
                    help="ORT threads. The QRB2210 has 4 A53 cores; leaving "
                         "some free keeps the rest of the system responsive.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # Limit thread pools BEFORE onnxruntime builds them.
    import os
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    model = args.model or str(C.MODEL_DIR / "model_int8.onnx")
    analyzer = ECGAnalyzer(model)
    try:
        opts = analyzer.session.get_session_options()
        opts.intra_op_num_threads = args.threads
    except Exception:
        pass

    stream = StreamAnalyzer(analyzer, fs=args.fs)
    read, src_name = open_input(args)

    out_path = Path(args.out)
    new_file = not out_path.exists()
    fh = open(out_path, "a", newline="", buffering=1)   # line buffered
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["logged_at", "source", "condition", "start_s",
                         "end_s", "duration_s", "n_beats", "mean_hr_bpm",
                         "confidence", "flag"])

    print(f"# model      {Path(model).name}", flush=True)
    print(f"# input      {src_name} @ {args.fs} Hz -> resampled to {C.FS} Hz",
          flush=True)
    print(f"# output     {out_path.resolve()}", flush=True)
    print(f"# warmup     needs {C.RR_LOCAL_WINDOW + 2} beats of rhythm "
          f"history before the first classification", flush=True)

    beats, reported = [], set()
    n_samples, t_start, last_stat = 0, time.time(), time.time()

    try:
        while True:
            chunk = read()
            if chunk is None:
                break
            if len(chunk) == 0:
                time.sleep(0.005)
                continue

            n_samples += len(chunk)
            beats.extend(stream.push(chunk))
            if len(beats) > 400:
                beats = beats[-400:]

            for e in group_episodes(beats):
                key = (e.label, int(e.start_sample))
                if key in reported:
                    continue
                growing = (beats and e.end_sample >= beats[-1].sample
                           and e.duration < C.MAX_EPISODE_S)
                if growing:
                    continue
                reported.add(key)
                flag = "low confidence" if e.low_confidence else ""
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), src_name,
                    e.label, round(e.start_time, 2), round(e.end_time, 2),
                    round(e.duration, 2), e.n_beats, round(e.mean_hr, 1),
                    round(e.mean_confidence, 3), flag])
                print(f"{e.label:<5} {e.start_time:8.2f}-{e.end_time:8.2f}s  "
                      f"{e.n_beats:>3} beats  HR {e.mean_hr:5.1f}  "
                      f"conf {e.mean_confidence:.2f} {flag}", flush=True)

            if not args.quiet and time.time() - last_stat > 10:
                last_stat = time.time()
                secs = n_samples / args.fs
                rt = secs / max(time.time() - t_start, 1e-6)
                print(f"# {secs:8.1f}s processed, {len(stream.emitted)} beats,"
                      f" {len(reported)} episodes, {rt:.2f}x real time",
                      flush=True)
    except KeyboardInterrupt:
        print("\n# stopped by user", flush=True)
    finally:
        fh.close()
        print(f"# wrote {len(reported)} episodes to {out_path}", flush=True)


if __name__ == "__main__":
    main()
