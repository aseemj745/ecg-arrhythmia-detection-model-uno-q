"""
ECG Live Viewer GUI
====================
Runs a data-source command (your ADC/GPIO reading script) as a subprocess,
plots incoming ECG samples in a scrolling live window, and periodically
(or on demand) runs your analysis module against the current window.

Meant to run directly on the Uno Q ("tally") with a monitor attached, so
no networking is involved -- it just launches your acquisition script
locally and reads its stdout.

Data source contract (see adc_reader_template.py for a working example):
  Your script must print ONE numeric sample per line to stdout, flushed
  immediately, at roughly your sampling rate, forever until killed.

Analysis module contract (same as ecg_gui.py / pqrst_adapter.py):
  def analyze(signal: np.ndarray, fs: float) -> dict
  Optional "peaks" / "annotations" keys get drawn on the plot.

Run with:
    python3 ecg_gui_live.py
"""

import os
import sys
import traceback
import importlib.util

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from live_stream import LiveReader


DEFAULT_COMMAND = "python3 adc_reader_template.py --fs 250"
PLOT_REFRESH_MS = 200          # how often the plot redraws
STATUS_REFRESH_MS = 1000       # how often the status line updates


class ECGLiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Live Viewer")
        self.root.geometry("1150x780")

        self.reader = None
        self.overlay_artists = []
        self.last_module = None
        self.last_module_path = None
        self._auto_job = None
        self._plot_job = None
        self._status_job = None
        self._analysis_running = False

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Data source command:").grid(row=0, column=0, sticky="w")
        self.cmd_var = tk.StringVar(value=DEFAULT_COMMAND)
        ttk.Entry(top, textvariable=self.cmd_var, width=55).grid(row=0, column=1, padx=5, sticky="we")
        ttk.Button(top, text="Browse script...", command=self.browse_script).grid(row=0, column=2, padx=2)

        ttk.Label(top, text="Sampling rate (Hz):").grid(row=0, column=3, sticky="e", padx=(10, 0))
        self.fs_var = tk.StringVar(value="250")
        ttk.Entry(top, textvariable=self.fs_var, width=8).grid(row=0, column=4, padx=5)

        ttk.Label(top, text="Window (s):").grid(row=0, column=5, sticky="e")
        self.window_var = tk.StringVar(value="10")
        ttk.Entry(top, textvariable=self.window_var, width=6).grid(row=0, column=6, padx=5)

        self.start_btn = ttk.Button(top, text="Start Live", command=self.start_live)
        self.start_btn.grid(row=1, column=0, pady=(8, 0), sticky="w")
        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop_live, state=tk.DISABLED)
        self.stop_btn.grid(row=1, column=1, pady=(8, 0), sticky="w")

        ttk.Label(top, text="Analysis module (.py):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.module_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.module_var, width=55).grid(row=2, column=1, padx=5, pady=(8, 0), sticky="we")
        ttk.Button(top, text="Browse...", command=self.browse_module).grid(row=2, column=2, padx=2, pady=(8, 0))
        ttk.Button(top, text="Run Module Now", command=self.run_module_now).grid(row=2, column=3, padx=2, pady=(8, 0))

        self.auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Auto-run every", variable=self.auto_var,
                         command=self._toggle_auto).grid(row=2, column=4, pady=(8, 0), sticky="e")
        self.auto_interval_var = tk.StringVar(value="5")
        ttk.Entry(top, textvariable=self.auto_interval_var, width=5).grid(row=2, column=5, pady=(8, 0), sticky="w")
        ttk.Label(top, text="sec").grid(row=2, column=6, pady=(8, 0), sticky="w")

        top.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Idle. Click 'Start Live' to begin streaming.")
        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=(5, 2))
        status.pack(side=tk.BOTTOM, fill=tk.X)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        plot_frame = ttk.Frame(main)
        main.add(plot_frame, weight=3)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Live ECG Signal")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        results_frame = ttk.Frame(main)
        main.add(results_frame, weight=2)
        ttk.Label(results_frame, text="Module output:").pack(anchor="w")
        self.results_box = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=45)
        self.results_box.pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------- browse

    def browse_script(self):
        path = filedialog.askopenfilename(title="Select data source script", filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if path:
            self.cmd_var.set(f"python3 {path}")

    def browse_module(self):
        path = filedialog.askopenfilename(title="Select analysis module", filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if path:
            self.module_var.set(path)

    # ------------------------------------------------------------- streaming

    def start_live(self):
        if self.reader is not None and self.reader.is_running():
            return
        try:
            fs = float(self.fs_var.get())
            window_s = float(self.window_var.get())
            if fs <= 0 or window_s <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Sampling rate and window must be positive numbers.")
            return

        command = self.cmd_var.get().strip()
        if not command:
            messagebox.showerror("Error", "Please provide a data source command.")
            return

        self.reader = LiveReader(command, fs=fs, window_seconds=window_s)
        try:
            self.reader.start()
        except Exception as e:
            messagebox.showerror("Failed to start", f"{type(e).__name__}: {e}")
            self.reader = None
            return

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Streaming from: {command}")

        self._schedule_plot_update()
        self._schedule_status_update()
        if self.auto_var.get():
            self._schedule_auto_run()

    def stop_live(self):
        if self.reader is not None:
            self.reader.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self._plot_job:
            self.root.after_cancel(self._plot_job)
            self._plot_job = None
        if self._status_job:
            self.root.after_cancel(self._status_job)
            self._status_job = None
        if self._auto_job:
            self.root.after_cancel(self._auto_job)
            self._auto_job = None
        self.status_var.set("Stopped.")

    # --------------------------------------------------------- plot updates

    def _schedule_plot_update(self):
        self._update_plot()
        self._plot_job = self.root.after(PLOT_REFRESH_MS, self._schedule_plot_update)

    def _update_plot(self):
        if self.reader is None:
            return
        snap = self.reader.snapshot()
        if not snap:
            return
        signal = np.array(snap)
        fs = self.reader.fs
        t = np.arange(len(signal)) / fs

        self.ax.clear()
        self.ax.plot(t, signal, linewidth=0.8, color="tab:blue")
        self.ax.set_title("Live ECG Signal")
        self.ax.set_xlabel("Time (s, rolling window)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)

        # redraw last analysis overlays (if any) mapped onto the current window
        if self.last_module is not None:
            self._draw_overlays_on_axis(signal, fs)

        self.canvas.draw()

    def _schedule_status_update(self):
        if self.reader is None:
            return
        st = self.reader.status()
        running = "running" if st["running"] else "STOPPED"
        msg = f"[{running}] {st['samples_received']} samples received | buffer {st['buffer_fill']}/{st['buffer_capacity']}"
        if st["last_error"]:
            msg += f" | error: {st['last_error']}"
        elif st["stderr_tail"]:
            msg += f" | stderr: {st['stderr_tail'][-1]}"
        self.status_var.set(msg)
        self._status_job = self.root.after(STATUS_REFRESH_MS, self._schedule_status_update)

    # ------------------------------------------------------------- analysis

    def _toggle_auto(self):
        if self.auto_var.get() and self.reader is not None and self.reader.is_running():
            self._schedule_auto_run()
        elif self._auto_job:
            self.root.after_cancel(self._auto_job)
            self._auto_job = None

    def _schedule_auto_run(self):
        try:
            interval_s = max(1.0, float(self.auto_interval_var.get()))
        except ValueError:
            interval_s = 5.0
        self.run_module_now(silent_if_busy=True)
        self._auto_job = self.root.after(int(interval_s * 1000), self._schedule_auto_run)

    def run_module_now(self, silent_if_busy=False):
        if self._analysis_running:
            if not silent_if_busy:
                messagebox.showinfo("Busy", "Previous analysis run is still finishing.")
            return
        if self.reader is None or not self.reader.snapshot():
            if not silent_if_busy:
                messagebox.showwarning("No data", "No live data collected yet.")
            return

        module_path = self.module_var.get().strip().strip('"')
        if not module_path or not os.path.isfile(module_path):
            if not silent_if_busy:
                messagebox.showerror("Error", "Please select a valid analysis module (.py) file.")
            return

        signal = np.array(self.reader.snapshot())
        fs = self.reader.fs

        self._analysis_running = True
        self.status_var.set("Running analysis module...")

        try:
            module = self._load_module(module_path)
            if not hasattr(module, "analyze"):
                raise AttributeError("Module has no analyze(signal, fs) function.")
            result = module.analyze(signal, fs)
            if not isinstance(result, dict):
                raise TypeError(f"analyze() must return a dict, got {type(result).__name__}.")

            self.last_module = result
            self.last_module_path = module_path
            self._display_results(result)
            self._draw_overlays_on_axis(signal, fs)
            self.canvas.draw()
        except Exception:
            tb = traceback.format_exc()
            self.results_box.delete("1.0", tk.END)
            self.results_box.insert(tk.END, f"ERROR running module:\n{tb}")
        finally:
            self._analysis_running = False

    def _load_module(self, path):
        module_name = "live_analysis_module_" + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _display_results(self, result):
        self.results_box.delete("1.0", tk.END)
        for key, value in result.items():
            if key in ("peaks", "annotations"):
                continue
            self.results_box.insert(tk.END, f"{key}: {value}\n")
        if "peaks" in result:
            self.results_box.insert(tk.END, f"\npeaks: {len(result['peaks'])} indices\n")
        if "annotations" in result:
            self.results_box.insert(tk.END, "\nannotations:\n")
            for label, idxs in result["annotations"].items():
                self.results_box.insert(tk.END, f"  {label}: {len(idxs)} indices\n")

    def _draw_overlays_on_axis(self, signal, fs):
        for artist in self.overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.overlay_artists = []

        if self.last_module is None:
            return
        result = self.last_module
        t = np.arange(len(signal)) / fs
        n = len(signal)

        if "peaks" in result and result["peaks"]:
            peaks = np.array([p for p in result["peaks"] if 0 <= p < n], dtype=int)
            if len(peaks):
                sc = self.ax.scatter(t[peaks], signal[peaks], color="red", s=25, zorder=5, label="peaks")
                self.overlay_artists.append(sc)

        if "annotations" in result and result["annotations"]:
            colors = ["orange", "green", "purple", "brown", "magenta"]
            for i, (label, idxs) in enumerate(result["annotations"].items()):
                idxs = np.array([ix for ix in idxs if 0 <= ix < n], dtype=int)
                if len(idxs):
                    sc = self.ax.scatter(t[idxs], signal[idxs], color=colors[i % len(colors)],
                                          marker="x", s=40, zorder=5, label=label)
                    self.overlay_artists.append(sc)

        if self.overlay_artists:
            self.ax.legend(loc="upper right", fontsize=8)

    # --------------------------------------------------------------- close

    def on_close(self):
        self.stop_live()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ECGLiveApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
