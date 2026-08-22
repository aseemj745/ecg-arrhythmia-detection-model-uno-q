"""
Template analysis module for ecg_gui.py
========================================

Point the GUI's "Analysis module (.py)" field at this file (or a copy of it)
and click "Run Module". The GUI will import this file and call:

    analyze(signal, fs)

`signal` is a 1D numpy array of the loaded ECG samples.
`fs`     is the sampling rate in Hz (float), taken from the GUI's input box.

Replace the body of analyze() below with your own logic. Keep the same
function name and signature, and keep returning a dict.

Special (optional) keys in the returned dict:
  - "peaks":       list/array of sample indices -> drawn as red dots on the plot
  - "annotations": dict of {label: list/array of sample indices} -> drawn as
                    labeled 'x' markers in different colors

Every other key/value pair is just shown as text in the results panel,
so you can report whatever metrics your module computes (heart rate,
QRS duration, classification label, confidence score, etc).

The example below is a simple, dependency-light R-peak detector + heart
rate estimator using scipy, just so you have a working reference.
"""

import numpy as np
from scipy.signal import find_peaks


def analyze(signal: np.ndarray, fs: float) -> dict:
    # --- 1. Basic signal stats -------------------------------------------------
    duration_s = len(signal) / fs
    stats = {
        "Duration (s)": round(duration_s, 2),
        "Num samples": len(signal),
        "Sampling rate (Hz)": fs,
        "Mean amplitude": round(float(np.mean(signal)), 4),
        "Std amplitude": round(float(np.std(signal)), 4),
        "Min amplitude": round(float(np.min(signal)), 4),
        "Max amplitude": round(float(np.max(signal)), 4),
    }

    # --- 2. R-peak detection (very simple, replace with your own method) -------
    # Normalize so the height threshold is scale-independent.
    norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
    min_distance = max(1, int(0.25 * fs))  # refractory period ~250 ms
    peaks, _ = find_peaks(norm, height=1.0, distance=min_distance)

    # --- 3. Heart rate from RR intervals ---------------------------------------
    if len(peaks) >= 2:
        rr_intervals_s = np.diff(peaks) / fs
        heart_rate_bpm = 60.0 / rr_intervals_s.mean()
        stats["Num beats detected"] = len(peaks)
        stats["Mean RR interval (s)"] = round(float(rr_intervals_s.mean()), 3)
        stats["Heart rate (bpm)"] = round(float(heart_rate_bpm), 1)
        stats["HRV - SDNN (ms)"] = round(float(np.std(rr_intervals_s) * 1000), 1)
    else:
        stats["Num beats detected"] = len(peaks)
        stats["Heart rate (bpm)"] = "N/A (fewer than 2 peaks found)"

    # --- 4. Return results -------------------------------------------------
    return {
        **stats,
        "peaks": peaks.tolist(),
    }
