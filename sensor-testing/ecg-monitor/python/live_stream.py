"""
live_stream.py
==============
Runs a data-acquisition command as a subprocess and continuously reads
ECG samples from its stdout into a thread-safe rolling buffer.

Contract for your acquisition script (e.g. the one that reads the BioAmp
EXG Pill via the Uno Q's ADC/GPIO):
  - Print ONE numeric sample per line to stdout.
  - Flush after every line (so the GUI sees samples as they happen, not
    in a delayed batch). In Python: print(value, flush=True)
  - Keep printing indefinitely at (roughly) your sampling rate, until
    killed.

Motion / "be still" contract (optional, used by adc_reader_uno_q_bridge.py
when the sketch has an MPU6050 wired up):
  - A line of the exact form "MOTION:0" or "MOTION:1" is treated as a
    motion-state sentinel, not a numeric sample: it is NOT added to the
    rolling buffer or the CSV log. "MOTION:1" means a big displacement was
    just detected (data collection is paused on the MCU); "MOTION:0" means
    it's still/stable again. LiveReader tracks the latest value and exposes
    it as status()["motion_detected"] so a GUI/dashboard can show a
    "please be still" popup while it's true.

See adc_reader_template.py for a working example (simulated data by
default, with a clearly marked spot to swap in real hardware reads).

This module has no GUI dependency, so it can be tested/run standalone.
"""

import csv
import os
import shlex
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone


class LiveReader:
    def __init__(self, command: str, fs: float, window_seconds: float = 10.0, log_path: str = None):
        """
        log_path: if given, every raw sample that comes in is appended to
        this CSV file (columns: sample_index, timestamp_utc, value) as it
        streams -- so whatever is shown live is also persisted to disk,
        whether the data source is simulated or real hardware.
        """
        self.command = command
        self.fs = fs
        self.buffer_len = max(1, int(fs * window_seconds))
        self.buffer = deque(maxlen=self.buffer_len)
        self.lock = threading.Lock()

        self._proc = None
        self._reader_thread = None
        self._stderr_thread = None
        self._running = False
        self._samples_received = 0
        self._last_error = None
        self._stderr_lines = deque(maxlen=20)

        # Latest motion-sensor state, updated from "MOTION:0"/"MOTION:1"
        # sentinel lines (see module docstring). None until the first
        # such line arrives -- e.g. no MPU6050/motion-aware source.
        self._motion_detected = None

        self.log_path = log_path
        self._log_file = None
        self._log_writer = None

    # ------------------------------------------------------------------

    def start(self):
        if self._running:
            return
        args = shlex.split(self.command)
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._running = True
        self._last_error = None

        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)) or ".", exist_ok=True)
            self._log_file = open(self.log_path, "w", newline="")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow(["sample_index", "timestamp_utc", "value"])
            self._log_file.flush()

        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()

    def stop(self):
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except Exception:
                pass
        self._proc = None
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
            self._log_writer = None

    # ------------------------------------------------------------------

    def _read_loop(self):
        try:
            for line in self._proc.stdout:
                if not self._running:
                    break
                line = line.strip()
                if not line:
                    continue

                if line.startswith("MOTION:"):
                    # Motion-state sentinel, not a data sample -- update
                    # the flag and skip buffering/logging it.
                    with self.lock:
                        self._motion_detected = (line[len("MOTION:"):].strip() == "1")
                    continue

                try:
                    value = float(line.split()[-1])  # tolerate "t, value" or "value"
                except ValueError:
                    continue
                with self.lock:
                    self.buffer.append(value)
                    idx = self._samples_received
                    self._samples_received += 1
                    if self._log_writer is not None:
                        ts = datetime.now(timezone.utc).isoformat()
                        self._log_writer.writerow([idx, ts, value])
                        # flush periodically rather than every line, to
                        # avoid disk I/O becoming the bottleneck at high fs
                        if idx % 25 == 0:
                            self._log_file.flush()
        except Exception as e:
            self._last_error = str(e)
        finally:
            self._running = False
            if self._log_file is not None:
                try:
                    self._log_file.flush()
                except Exception:
                    pass

    def _stderr_loop(self):
        try:
            for line in self._proc.stderr:
                line = line.rstrip()
                if line:
                    self._stderr_lines.append(line)
        except Exception:
            pass

    # ------------------------------------------------------------------

    def snapshot(self):
        """Thread-safe copy of the current buffer, oldest-first, as a list of floats."""
        with self.lock:
            return list(self.buffer)

    def is_running(self):
        return self._running and self._proc is not None and self._proc.poll() is None

    def status(self):
        with self.lock:
            n = self._samples_received
            motion = self._motion_detected
        return {
            "running": self.is_running(),
            "samples_received": n,
            "buffer_fill": len(self.buffer),
            "buffer_capacity": self.buffer_len,
            "last_error": self._last_error,
            "stderr_tail": list(self._stderr_lines),
            "log_path": self.log_path,
            # True/False once a motion-aware source has reported at least
            # once (see adc_reader_uno_q_bridge.py); None if the source
            # never sends "MOTION:" lines (e.g. the plain simulator).
            "motion_detected": motion,
        }


if __name__ == "__main__":
    # Quick standalone smoke test (no GUI): stream from adc_reader_template.py
    # for a few seconds and report what came in.
    import os

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adc_reader_template.py")
    reader = LiveReader(f"python3 {script} --fs 250", fs=250, window_seconds=5)
    reader.start()
    print("Streaming for 3 seconds...")
    time.sleep(3)
    reader.stop()
    snap = reader.snapshot()
    print("Status:", reader.status())
    print(f"Collected {len(snap)} samples. First 5: {snap[:5]}")
