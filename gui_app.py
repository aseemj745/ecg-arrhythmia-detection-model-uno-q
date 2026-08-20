"""
ECG Arrhythmia Detection - demo GUI.

    python gui_app.py

Two scrolling traces:
  TOP     a healthy held-out MIT-BIH patient, as the reference of what
          normal looks like coming through this exact pipeline
  BOTTOM  the monitored channel - either another held-out MIT-BIH patient
          with a known arrhythmia, or the live BioAmp sensor

Detected episodes on the BOTTOM trace are shaded on the plot and streamed to
a background process (log_worker.py) that appends them to an Excel workbook.

Honesty rule, straight from the project brief: the banner always states
whether the monitored trace is a recorded MIT-BIH patient or a live person,
and arrhythmias are never presented as having come from a live volunteer.
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ecg import config as C
from ecg import sources as S
from ecg.pipeline import ECGAnalyzer, StreamAnalyzer, group_episodes

WINDOW_S = 8.0          # seconds visible on screen
TICK_MS = 50            # redraw period

CLASS_COLOUR = {
    "NOR": "#1a9850",
    "LBBB": "#d73027",
    "RBBB": "#7b3294",
    "PVC": "#e08214",
    "AFIB": "#c51b7d",
}


# ==========================================================================
class Channel:
    """One trace: a source, its stream analyser, and the rolling display."""

    def __init__(self, source, analyzer, keep_s=30.0, classify=True):
        self.source = source
        self.classify = classify
        self.stream = StreamAnalyzer(analyzer, fs=source.fs)
        self.keep_n = int(keep_s * source.fs)

        self.samples = np.zeros(0)
        self.t0 = 0.0                      # time of samples[0]
        self.beats = []
        self.episodes = []
        self._reported = set()

    @property
    def fs(self):
        return self.source.fs

    def reset(self):
        self.stream.reset()
        self.samples = np.zeros(0)
        self.t0 = 0.0
        self.beats.clear()
        self.episodes.clear()
        self._reported.clear()

    def prime(self, seconds=C.DEMO_PRIME_S):
        """
        Push `seconds` of signal through the analyser before display begins.

        The rhythm features need 11 beats of RR history before ANY beat can be
        classified, so without this the first ~10 seconds of a replay show a
        moving trace with no markers - which looks like the detector is
        broken when it is simply warming up. Recorded sources can skip that
        wait; live capture genuinely cannot, and prefetch() returns nothing
        there, so the GUI falls back to showing the warmup countdown.
        """
        chunk = self.source.prefetch(int(seconds * self.fs))
        if len(chunk) == 0:
            return False
        self.samples = chunk[-self.keep_n:]
        self.t0 = max(0.0, (len(chunk) - len(self.samples)) / self.fs)
        if self.classify:
            self.beats.extend(self.stream.push(chunk))
            self.episodes = group_episodes(self.beats)
            for e in self.episodes:
                self._reported.add((e.label, int(e.start_sample)))
        return True

    @property
    def warmup_remaining(self):
        """Beats still needed before classification can start, or 0."""
        need = C.RR_LOCAL_WINDOW + 2
        return max(0, need - len(self.stream.emitted)) if not self.beats else 0

    def poll(self):
        """Pull whatever is available; return newly detected episodes."""
        new = self.source.read()
        if len(new) == 0:
            return []

        self.samples = np.concatenate([self.samples, new])
        if len(self.samples) > self.keep_n:
            drop = len(self.samples) - self.keep_n
            self.samples = self.samples[drop:]
            self.t0 += drop / self.fs

        if not self.classify:
            return []

        self.beats.extend(self.stream.push(new))
        if len(self.beats) > 400:
            self.beats = self.beats[-400:]

        # Display and reporting are deliberately separate.
        #
        # DISPLAY takes every episode including the one still in progress, so
        # the shading appears on screen the moment a run qualifies rather
        # than after it ends.
        #
        # REPORTING waits until a run is finalised - a different label has
        # arrived, or MAX_EPISODE_S has chunked it - so each Excel row has a
        # settled duration and beat count, and a sustained arrhythmia still
        # produces rows as it goes instead of one row that never arrives.
        eps = group_episodes(self.beats)
        self.episodes = eps[-200:]

        fresh = []
        for e in eps:
            key = (e.label, int(e.start_sample))
            if key in self._reported:
                continue
            still_growing = (self.beats
                             and e.end_sample >= self.beats[-1].sample
                             and e.duration < C.MAX_EPISODE_S)
            if still_growing:
                continue
            self._reported.add(key)
            fresh.append(e)
        return fresh

    @property
    def now_s(self):
        return self.t0 + len(self.samples) / self.fs


# ==========================================================================
class LoggerProcess:
    """Owns the log_worker.py subprocess and a writer thread."""

    def __init__(self, excel_path):
        self.excel_path = Path(excel_path)
        self.proc = None
        self.q = queue.Queue()
        self.thread = None
        self.rows = 0
        self.last_message = ""

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        if self.running:
            return
        self.proc = subprocess.Popen(
            [sys.executable, "log_worker.py", "--excel", str(self.excel_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=str(Path(__file__).parent),
        )
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()
        threading.Thread(target=self._drain, daemon=True).start()

    def _pump(self):
        while True:
            ev = self.q.get()
            if ev is None or not self.running:
                break
            try:
                self.proc.stdin.write(json.dumps(ev) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                break

    def _drain(self):
        try:
            for line in self.proc.stdout:
                self.last_message = line.strip()
        except Exception:
            pass

    def log(self, ev):
        if self.running:
            self.q.put(ev)
            self.rows += 1

    def stop(self):
        if not self.running:
            return
        # Let the writer thread drain first. Closing stdin immediately used to
        # discard whatever was still queued, so the last few detections of a
        # demo never reached the workbook.
        deadline = time.time() + 5.0
        while not self.q.empty() and time.time() < deadline:
            time.sleep(0.05)
        try:
            self.q.put({"cmd": "stop"})
            time.sleep(0.3)
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


# ==========================================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("ECG Arrhythmia Detection  -  Arduino UNO Q")
        root.geometry("1280x860")

        self.analyzer = ECGAnalyzer()
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = LoggerProcess(C.REPORT_DIR / "detections.xlsx")

        self.ref = None
        self.mon = None
        self.running = False
        self.live_mode = tk.BooleanVar(value=False)
        self.log_enabled = tk.BooleanVar(value=True)

        self._build_controls()
        self._build_plots()
        self._build_status()

        self.load_sources()
        self._tick_id = None
        self._closing = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._tick_id = self.root.after(TICK_MS, self.tick)

    # ------------------------------------------------------------- layout
    def _build_controls(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill="x")

        self.healthy_choices = [f"{r}  -  normal ({fold})"
                                for r, fold, _ in C.DEMO_HEALTHY]
        self.arr_choices = [f"{r}  -  {cls} ({fold})"
                            for r, cls, fold, _ in C.DEMO_ARRHYTHMIA]

        ttk.Label(bar, text="Reference (healthy):").grid(row=0, column=0,
                                                         sticky="w")
        self.ref_var = tk.StringVar(value=self.healthy_choices[0])
        ref_box = ttk.Combobox(bar, textvariable=self.ref_var, width=24,
                               state="readonly", values=self.healthy_choices)
        ref_box.grid(row=0, column=1, padx=(4, 16))
        ref_box.bind("<<ComboboxSelected>>", lambda e: self.load_sources())

        ttk.Label(bar, text="Monitored:").grid(row=0, column=2, sticky="w")
        self.mon_var = tk.StringVar(value=self.arr_choices[0])
        self.mon_box = ttk.Combobox(bar, textvariable=self.mon_var, width=26,
                                    state="readonly", values=self.arr_choices)
        self.mon_box.grid(row=0, column=3, padx=(4, 16))
        self.mon_box.bind("<<ComboboxSelected>>", lambda e: self.load_sources())

        ttk.Label(bar, text="Speed:").grid(row=0, column=4, sticky="w")
        self.speed_var = tk.StringVar(value="1.0")
        ttk.Combobox(bar, textvariable=self.speed_var, width=5,
                     state="readonly",
                     values=["0.5", "1.0", "2.0", "4.0"]).grid(row=0, column=5,
                                                               padx=(4, 16))

        self.play_btn = ttk.Button(bar, text="Start", command=self.toggle)
        self.play_btn.grid(row=0, column=6, padx=3)
        ttk.Button(bar, text="Restart",
                   command=self.load_sources).grid(row=0, column=7, padx=3)
        # Stop closes the application. Pause is the play button above and
        # only halts playback - the traces, detections and log stay put.
        ttk.Button(bar, text="Stop (close)",
                   command=self.on_close).grid(row=0, column=8, padx=(3, 12))

        ttk.Checkbutton(bar, text="Log detections to Excel",
                        variable=self.log_enabled).grid(row=0, column=9,
                                                        padx=(8, 4))
        ttk.Button(bar, text="Open log folder",
                   command=self.open_log_folder).grid(row=0, column=10)

        # ---- live row
        live = ttk.Frame(self.root, padding=(8, 0))
        live.pack(fill="x")
        ttk.Checkbutton(live, text="LIVE sensor (BioAmp EXG Pill via UNO Q)",
                        variable=self.live_mode,
                        command=self.on_live_toggle).grid(row=0, column=0)
        ttk.Label(live, text="Port:").grid(row=0, column=1, padx=(16, 2))
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(live, textvariable=self.port_var,
                                     width=12, values=S.list_serial_ports())
        self.port_box.grid(row=0, column=2)
        ttk.Button(live, text="Rescan", command=self.rescan_ports).grid(
            row=0, column=3, padx=4)
        ttk.Label(live, text="Sample rate:").grid(row=0, column=4, padx=(16, 2))
        self.baud_var = tk.StringVar(value="250")
        ttk.Entry(live, textvariable=self.baud_var, width=6).grid(row=0,
                                                                  column=5)
        self.live_note = ttk.Label(
            live, text="  (live mode streams a REAL person's normal rhythm; "
                       "arrhythmias below always come from MIT-BIH)",
            foreground="#666")
        self.live_note.grid(row=0, column=6, padx=(12, 0))

    def _build_plots(self):
        self.fig = Figure(figsize=(12.6, 6.4), dpi=100)
        self.fig.subplots_adjust(hspace=0.42, left=0.06, right=0.985,
                                 top=0.93, bottom=0.08)
        self.ax_ref = self.fig.add_subplot(211)
        self.ax_mon = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=8, pady=4)

    def _build_status(self):
        frame = ttk.Frame(self.root, padding=(8, 4))
        frame.pack(fill="both")

        self.status = tk.Text(frame, height=7, wrap="none",
                              font=("Consolas", 9))
        self.status.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame, command=self.status.yview)
        sb.pack(side="right", fill="y")
        self.status.config(yscrollcommand=sb.set)

        self.footer = ttk.Label(self.root, padding=(8, 2), foreground="#444")
        self.footer.pack(fill="x")

    # -------------------------------------------------------------- setup
    def load_sources(self):
        was_running = self.running
        self.running = False

        speed = float(self.speed_var.get())
        try:
            ref_src = S.DatasetSource(self.ref_var.get().split()[0],
                                      speed=speed)
        except Exception as e:
            messagebox.showerror("Reference record", str(e))
            return

        if self.live_mode.get():
            port = self.port_var.get().strip()
            if not port:
                messagebox.showwarning(
                    "Live mode", "Pick a serial port first, or untick LIVE.")
                self.live_mode.set(False)
                return
            try:
                fs = int(float(self.baud_var.get()))
                mon_src = S.SerialSource(port, fs=fs)
                mon_src.start()
            except Exception as e:
                messagebox.showerror("Serial", str(e))
                self.live_mode.set(False)
                return
        else:
            try:
                mon_src = S.DatasetSource(self.mon_var.get().split()[0],
                                          speed=speed)
            except Exception as e:
                messagebox.showerror("Monitored record", str(e))
                return

        self.ref = Channel(ref_src, self.analyzer)
        self.mon = Channel(mon_src, self.analyzer)

        # Skip the rhythm-feature warmup on recorded sources so detections
        # appear from the first visible beat instead of ~10 s in.
        primed = self.ref.prime() and self.mon.prime()

        self.status.delete("1.0", "end")
        self.log_line(f"session {self.session}")
        self.log_line(f"model    {Path(self.analyzer.model_path).name}")
        self.log_line(f"top      {ref_src.status}")
        self.log_line(f"bottom   {mon_src.status}")
        self.log_line(f"warmup   "
                      + (f"pre-seeded {C.DEMO_PRIME_S:.0f}s, detection is "
                         f"live from the first visible beat"
                         if primed else
                         f"live source - needs {C.RR_LOCAL_WINDOW + 2} beats "
                         f"of rhythm history before the first classification"))
        self.log_line("-" * 96)

        self.running = was_running
        self.play_btn.config(text="Pause" if self.running else "Start")
        self.redraw()

    def rescan_ports(self):
        ports = S.list_serial_ports()
        self.port_box["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
        if not ports:
            messagebox.showinfo(
                "Serial ports",
                "No serial ports found.\n\n"
                "If pyserial is not installed yet:\n    pip install pyserial")

    def on_live_toggle(self):
        if self.live_mode.get() and not S.list_serial_ports():
            messagebox.showinfo(
                "Live mode",
                "No serial ports detected.\n\n"
                "Connect the UNO Q over USB and press Rescan.\n"
                "If pyserial is missing:  pip install pyserial")
        self.load_sources()

    # ------------------------------------------------------------ running
    def toggle(self):
        if self.ref is None:
            self.load_sources()
        self.running = not self.running
        self.play_btn.config(text="Pause" if self.running else "Start")
        if self.running:
            self.ref.source.start()
            self.mon.source.start()
            if self.log_enabled.get():
                self.logger.start()

    def tick(self):
        try:
            if self.running and self.ref is not None:
                self.ref.poll()
                for ep in self.mon.poll():
                    self.on_detection(ep)
                self.redraw()
        except Exception as e:                       # never kill the loop
            self.log_line(f"ERROR {type(e).__name__}: {e}")
        if not self._closing:
            self._tick_id = self.root.after(TICK_MS, self.tick)

    def on_detection(self, ep):
        truth = ""
        src = self.mon.source
        if isinstance(src, S.DatasetSource):
            truth = src.truth_near(ep.start_sample) or ""

        flag = "low confidence" if ep.low_confidence else ""
        self.log_line(
            f"{ep.label:<5} {ep.start_time:7.2f}-{ep.end_time:7.2f}s  "
            f"{ep.n_beats:>3} beats  conf {ep.mean_confidence:.2f}  "
            f"HR {ep.mean_hr:5.1f}  "
            + (f"[annotated: {truth}]" if truth else "")
            + ("  <low confidence>" if flag else ""),
            colour=CLASS_COLOUR.get(ep.label))

        if self.log_enabled.get():
            self.logger.start()
            self.logger.log({
                "session": self.session,
                "source": src.name,
                "data_origin": ("live sensor"
                                if isinstance(src, S.SerialSource)
                                else "MIT-BIH recording"),
                "condition": ep.label,
                "start_s": ep.start_time,
                "end_s": ep.end_time,
                "n_beats": ep.n_beats,
                "mean_hr": ep.mean_hr,
                "confidence": ep.mean_confidence,
                "flag": flag,
                "annotated_truth": truth,
            })

    # ------------------------------------------------------------ drawing
    def redraw(self):
        self._draw_channel(
            self.ax_ref, self.ref, shade=False,
            title=f"REFERENCE  -  healthy patient {self.ref_var.get()}  "
                  f"(MIT-BIH recording)")
        live = isinstance(self.mon.source, S.SerialSource)
        title = ("MONITORED  -  LIVE sensor, real person "
                 "(normal rhythm only - no arrhythmia is induced)"
                 if live else
                 f"MONITORED  -  patient {self.mon_var.get()}  "
                 f"(MIT-BIH recording, not a live patient)")
        self._draw_channel(self.ax_mon, self.mon, shade=True, title=title)
        self.canvas.draw_idle()
        self.update_footer()

    def _draw_channel(self, ax, ch, shade, title):
        ax.clear()
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("mV")
        ax.grid(alpha=0.25)

        if ch is None or len(ch.samples) == 0:
            ax.text(0.5, 0.5, "waiting for data", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
            return

        end = ch.now_s
        start = max(ch.t0, end - WINDOW_S)
        i0 = int((start - ch.t0) * ch.fs)
        seg = ch.samples[i0:]
        t = start + np.arange(len(seg)) / ch.fs

        ax.plot(t, seg, color="#2b6cb0", linewidth=0.9)
        ax.set_xlim(start, start + WINDOW_S)
        if len(seg) > 10:
            pad = 0.15 * (np.ptp(seg) or 1.0)
            ax.set_ylim(seg.min() - pad, seg.max() + pad)

        for b in ch.beats:
            if not (start <= b.time_s <= start + WINDOW_S):
                continue
            idx = int((b.time_s - ch.t0) * ch.fs)
            if not (0 <= idx < len(ch.samples)):
                continue
            colour = CLASS_COLOUR.get(b.label, "#888")
            alpha = 1.0 if b.confidence >= C.LOW_CONFIDENCE else 0.45
            ax.plot([b.time_s], [ch.samples[idx]], "v", color=colour,
                    markersize=7, alpha=alpha)
            if b.is_abnormal:
                ax.annotate(b.label, (b.time_s, ch.samples[idx]),
                            textcoords="offset points", xytext=(0, 11),
                            ha="center", fontsize=7.5, color=colour,
                            fontweight="bold")

        # Live sources cannot be pre-seeded, so say so rather than showing an
        # empty trace that looks like a dead detector.
        if ch.classify and not ch.beats:
            need = C.RR_LOCAL_WINDOW + 2
            ax.text(0.99, 0.05,
                    f"warming up - rhythm features need {need} beats of "
                    f"history",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8.5, color="#b45309",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7",
                              ec="#b45309", alpha=0.9))

        if shade:
            for e in ch.episodes:
                if e.end_time < start or e.start_time > start + WINDOW_S:
                    continue
                colour = CLASS_COLOUR.get(e.label, "#888")
                ax.axvspan(max(e.start_time - 0.25, start),
                           min(e.end_time + 0.25, start + WINDOW_S),
                           color=colour, alpha=0.13, zorder=0)

        ax.set_xlabel("time (s)")

    # ------------------------------------------------------------- status
    def log_line(self, text, colour=None):
        tag = None
        if colour:
            tag = f"c{colour}"
            self.status.tag_config(tag, foreground=colour)
        self.status.insert("end", text + "\n", tag)
        self.status.see("end")

    def update_footer(self):
        if self.mon is None:
            return
        beats = self.mon.beats
        hr = 0.0
        if len(beats) >= 2:
            rrs = [b.rr_s for b in beats[-10:] if 0.2 < b.rr_s < 3.0]
            if rrs:
                hr = 60.0 / float(np.mean(rrs))
        counts = {}
        for b in beats:
            counts[b.label] = counts.get(b.label, 0) + 1
        breakdown = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))

        logger = (f"logger running (pid {self.logger.proc.pid}, "
                  f"{self.logger.rows} rows)"
                  if self.logger.running else "logger idle")
        self.footer.config(
            text=f"HR {hr:5.1f} bpm   |   recent beats  {breakdown}   |   "
                 f"episodes {len(self.mon.episodes)}   |   {logger}   |   "
                 f"{C.REPORT_DIR / 'detections.xlsx'}")

    def open_log_folder(self):
        path = C.REPORT_DIR
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            messagebox.showinfo("Log folder", f"{path}\n\n{e}")

    def on_close(self):
        """Shut down in an order that cannot raise.

        The redraw timer must be cancelled BEFORE the widgets go away,
        otherwise the already-queued callback fires against a destroyed
        interpreter and Tk reports `invalid command name ...tick`. The
        re-entrancy guard matters because this is reachable from two places -
        the Stop button and the window-manager close box.
        """
        if self._closing:
            return
        self._closing = True
        self.running = False

        if self._tick_id is not None:
            try:
                self.root.after_cancel(self._tick_id)
            except Exception:
                pass
            self._tick_id = None

        for ch in (self.mon, self.ref):
            if ch is not None:
                try:
                    ch.source.stop()
                except Exception:
                    pass

        self.logger.stop()          # drains the queue, so nothing is lost
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    if not (C.MODEL_DIR / "model_int8.onnx").exists():
        print("model_int8.onnx not found - run step3_export_onnx.py first")
        return
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
