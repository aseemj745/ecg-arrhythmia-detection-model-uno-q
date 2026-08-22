"""
ECG Viewer GUI
==============

A small Tkinter desktop app to:
  1. Paste/browse the path to an ECG signal file (CSV or TXT).
  2. Plot the signal.
  3. Point at your own Python module (a .py file with an `analyze(signal, fs)`
     function) and run it against the loaded signal, showing whatever it
     returns.

Your module contract
---------------------
Your .py file must define:

    def analyze(signal: np.ndarray, fs: float) -> dict:
        ...
        return {
            "Heart Rate (bpm)": 72.3,          # any scalar/string values are
            "Num beats": 54,                    # shown in the results panel
            "peaks": [123, 456, 789, ...],       # optional: sample indices,
                                                  # drawn as red dots on the plot
            "annotations": {                     # optional: extra marker sets
                "Ectopic beats": [200, 900]       # label -> list of indices
            },
        }

Only "peaks" and "annotations" are treated specially (for plotting).
Every other key/value pair in the returned dict is just displayed as text.
See user_module.py for a working example you can copy and modify.

Run with:
    python ecg_gui.py
"""

import os
import sys
import traceback
import importlib.util

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class ECGViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Viewer")
        self.root.geometry("1100x750")

        # State
        self.df = None
        self.signal = None
        self.fs = 250.0
        self.overlay_artists = []

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        # --- ECG file row ---
        ttk.Label(top, text="ECG file (CSV/TXT):").grid(row=0, column=0, sticky="w")
        self.ecg_path_var = tk.StringVar()
        ecg_entry = ttk.Entry(top, textvariable=self.ecg_path_var, width=70)
        ecg_entry.grid(row=0, column=1, padx=5, sticky="we")
        ttk.Button(top, text="Browse...", command=self.browse_ecg_file).grid(row=0, column=2, padx=2)
        ttk.Button(top, text="Load", command=self.load_file).grid(row=0, column=3, padx=2)

        # --- Column + sampling rate row ---
        ttk.Label(top, text="Signal column:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(top, textvariable=self.column_var, state="readonly", width=30)
        self.column_combo.grid(row=1, column=1, padx=5, pady=(6, 0), sticky="w")

        ttk.Label(top, text="Sampling rate (Hz):").grid(row=1, column=2, sticky="e", pady=(6, 0))
        self.fs_var = tk.StringVar(value="250")
        ttk.Entry(top, textvariable=self.fs_var, width=10).grid(row=1, column=3, padx=5, pady=(6, 0), sticky="w")

        ttk.Button(top, text="Plot Signal", command=self.plot_signal).grid(row=1, column=4, padx=5, pady=(6, 0))

        # --- Module file row ---
        ttk.Label(top, text="Analysis module (.py):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.module_path_var = tk.StringVar()
        mod_entry = ttk.Entry(top, textvariable=self.module_path_var, width=70)
        mod_entry.grid(row=2, column=1, padx=5, pady=(6, 0), sticky="we")
        ttk.Button(top, text="Browse...", command=self.browse_module_file).grid(row=2, column=2, padx=2, pady=(6, 0))
        ttk.Button(top, text="Run Module", command=self.run_module).grid(row=2, column=3, padx=2, pady=(6, 0))
        ttk.Button(top, text="Clear Results", command=self.clear_results).grid(row=2, column=4, padx=5, pady=(6, 0))

        top.columnconfigure(1, weight=1)

        # --- Status bar ---
        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=(5, 2))
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Main split: plot (left) / results (right) ---
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        plot_frame = ttk.Frame(main)
        main.add(plot_frame, weight=3)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("ECG Signal")
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

    # ------------------------------------------------------------- actions

    def browse_ecg_file(self):
        path = filedialog.askopenfilename(
            title="Select ECG file",
            filetypes=[("CSV/Text files", "*.csv *.txt *.tsv"), ("All files", "*.*")],
        )
        if path:
            self.ecg_path_var.set(path)
            self.load_file()

    def browse_module_file(self):
        path = filedialog.askopenfilename(
            title="Select analysis module",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if path:
            self.module_path_var.set(path)

    def load_file(self):
        path = self.ecg_path_var.get().strip().strip('"')
        if not path:
            messagebox.showwarning("No file", "Please provide a path to an ECG file.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("File not found", f"Could not find file:\n{path}")
            return

        try:
            sep = "\t" if path.lower().endswith((".tsv", ".txt")) else None
            try:
                df = pd.read_csv(path, sep=sep, engine="python")
            except Exception:
                # Fall back: no header, whitespace/comma separated numeric data
                df = pd.read_csv(path, sep=None, engine="python", header=None)

            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] == 0:
                raise ValueError("No numeric columns found in the file.")

            self.df = numeric_df
            cols = list(numeric_df.columns.astype(str))
            self.column_combo["values"] = cols
            self.column_var.set(cols[0])
            self.status_var.set(f"Loaded {path}  ({numeric_df.shape[0]} samples, {numeric_df.shape[1]} numeric column(s))")
            self.plot_signal()
        except Exception as e:
            messagebox.showerror("Error loading file", f"{type(e).__name__}: {e}")
            self.status_var.set("Failed to load file.")

    def _get_signal_and_fs(self):
        if self.df is None:
            raise ValueError("No ECG file loaded yet.")
        col = self.column_var.get()
        if not col:
            raise ValueError("No signal column selected.")
        # columns were stringified for the combobox; match back to original dtype key
        matching = [c for c in self.df.columns if str(c) == col]
        if not matching:
            raise ValueError(f"Column '{col}' not found.")
        signal = self.df[matching[0]].to_numpy(dtype=float)
        signal = signal[~np.isnan(signal)]

        try:
            fs = float(self.fs_var.get())
            if fs <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("Sampling rate must be a positive number.")

        return signal, fs

    def plot_signal(self):
        try:
            signal, fs = self._get_signal_and_fs()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.signal = signal
        self.fs = fs
        self._clear_overlays()

        t = np.arange(len(signal)) / fs
        self.ax.clear()
        self.ax.plot(t, signal, linewidth=0.8, color="tab:blue")
        self.ax.set_title("ECG Signal")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        self.status_var.set(f"Plotted {len(signal)} samples at {fs} Hz ({len(signal)/fs:.2f} s).")

    def _clear_overlays(self):
        for artist in self.overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.overlay_artists = []

    def clear_results(self):
        self.results_box.delete("1.0", tk.END)
        self._clear_overlays()
        self.canvas.draw()

    def run_module(self):
        module_path = self.module_path_var.get().strip().strip('"')
        if not module_path:
            messagebox.showwarning("No module", "Please provide a path to your analysis .py module.")
            return
        if not os.path.isfile(module_path):
            messagebox.showerror("File not found", f"Could not find module file:\n{module_path}")
            return
        if self.signal is None:
            messagebox.showwarning("No signal", "Load and plot an ECG signal first.")
            return

        self.results_box.delete("1.0", tk.END)
        self.status_var.set("Running module...")
        self.root.update_idletasks()

        try:
            module = self._load_module(module_path)
            if not hasattr(module, "analyze"):
                raise AttributeError(
                    "Module does not define an 'analyze(signal, fs)' function. "
                    "See user_module.py for the expected format."
                )
            result = module.analyze(self.signal, self.fs)
            if not isinstance(result, dict):
                raise TypeError(f"analyze() must return a dict, got {type(result).__name__}.")

            self._display_results(result)
            self._overlay_results(result)
            self.status_var.set("Module ran successfully.")
        except Exception as e:
            tb = traceback.format_exc()
            self.results_box.insert(tk.END, f"ERROR running module:\n{type(e).__name__}: {e}\n\n{tb}")
            self.status_var.set("Module raised an error. See results panel.")

    def _load_module(self, path):
        module_name = "user_analysis_module_" + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _display_results(self, result):
        for key, value in result.items():
            if key in ("peaks", "annotations"):
                continue
            self.results_box.insert(tk.END, f"{key}: {value}\n")

        if "peaks" in result:
            self.results_box.insert(tk.END, f"\npeaks: {len(result['peaks'])} indices found\n")
        if "annotations" in result:
            self.results_box.insert(tk.END, "\nannotations:\n")
            for label, idxs in result["annotations"].items():
                self.results_box.insert(tk.END, f"  {label}: {len(idxs)} indices\n")

    def _overlay_results(self, result):
        self._clear_overlays()
        t = np.arange(len(self.signal)) / self.fs

        if "peaks" in result and result["peaks"] is not None:
            peaks = np.array(result["peaks"], dtype=int)
            peaks = peaks[(peaks >= 0) & (peaks < len(self.signal))]
            scatter = self.ax.scatter(
                t[peaks], self.signal[peaks], color="red", marker="o", s=25, zorder=5, label="peaks"
            )
            self.overlay_artists.append(scatter)

        if "annotations" in result and result["annotations"]:
            colors = ["orange", "green", "purple", "brown", "magenta"]
            for i, (label, idxs) in enumerate(result["annotations"].items()):
                idxs = np.array(idxs, dtype=int)
                idxs = idxs[(idxs >= 0) & (idxs < len(self.signal))]
                scatter = self.ax.scatter(
                    t[idxs], self.signal[idxs],
                    color=colors[i % len(colors)], marker="x", s=40, zorder=5, label=label,
                )
                self.overlay_artists.append(scatter)

        if "peaks" in result or "annotations" in result:
            self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = ECGViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
