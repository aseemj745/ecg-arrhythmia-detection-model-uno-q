"""
ECG P-QRS-T Complex Detector & Parameter Extractor
====================================================
Designed for signals acquired from the Upside Down Labs BioAmp EXG Pill.

Pipeline:
  1. Bandpass + Notch filtering (removes baseline wander, muscle noise, powerline hum)
  2. Pan-Tompkins algorithm -> R-peak detection
  3. Windowed search around each R-peak -> Q, S, P, T wave detection
  4. Clinical parameter extraction -> HR, RR, PR, QRS duration, QT, ST level

Usage:
    python ecg_pqrst_detector.py your_ecg_data.csv --fs 250

Input CSV: single column of raw ADC/voltage samples (no header), OR
           two columns: time, amplitude
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import matplotlib.pyplot as plt
import argparse


# ---------------------------------------------------------------------------
# 1. PREPROCESSING
# ---------------------------------------------------------------------------

def bandpass_filter(signal, fs, low=0.5, high=40.0, order=4):
    """Removes baseline wander (<0.5Hz) and high-freq noise (>40Hz)."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, freq=50.0, q=30.0):
    """Removes powerline interference. Use freq=60.0 if you're in a 60Hz region."""
    nyq = 0.5 * fs
    b, a = iirnotch(freq / nyq, q)
    return filtfilt(b, a, signal)


def preprocess(raw_signal, fs, powerline_freq=50.0):
    sig = bandpass_filter(raw_signal, fs)
    sig = notch_filter(sig, fs, freq=powerline_freq)
    # remove DC offset
    sig = sig - np.mean(sig)
    return sig


# ---------------------------------------------------------------------------
# 2. PAN-TOMPKINS R-PEAK DETECTION
# ---------------------------------------------------------------------------

def pan_tompkins_r_peaks(filtered_signal, fs):
    """
    Classic Pan-Tompkins QRS detector.
    Returns indices of detected R-peaks in the filtered_signal.
    """
    # --- Derivative filter (emphasizes slope) ---
    derivative_kernel = np.array([1, 2, 0, -2, -1]) * (fs / 8.0)
    diff_signal = np.convolve(filtered_signal, derivative_kernel, mode="same")

    # --- Squaring (all positive, emphasizes high-freq QRS energy) ---
    squared_signal = diff_signal ** 2

    # --- Moving window integration (~150ms window, smooths into one pulse/beat) ---
    window_size = int(0.150 * fs)
    integrated_signal = np.convolve(
        squared_signal, np.ones(window_size) / window_size, mode="same"
    )

    # --- Adaptive threshold peak detection on integrated signal ---
    min_distance = int(0.25 * fs)  # refractory period ~250ms (max ~240bpm)
    threshold = 0.35 * np.max(integrated_signal)
    peaks, _ = find_peaks(integrated_signal, height=threshold, distance=min_distance)

    # --- Refine: snap each detected peak to the true local max in filtered_signal ---
    search_radius = int(0.075 * fs)  # +/- 75ms
    r_peaks = []
    for p in peaks:
        lo = max(0, p - search_radius)
        hi = min(len(filtered_signal), p + search_radius)
        local_max_idx = lo + np.argmax(filtered_signal[lo:hi])
        r_peaks.append(local_max_idx)

    return np.array(sorted(set(r_peaks)))


# ---------------------------------------------------------------------------
# 3. P-QRS-T COMPLEX RECONSTRUCTION
# ---------------------------------------------------------------------------

def detect_pqrst_complex(filtered_signal, fs, r_peaks):
    """
    For each R-peak, searches nearby windows to locate Q, S, P, T waves.
    Returns a list of dicts, one per detected beat.
    """
    complexes = []

    for i, r in enumerate(r_peaks):
        beat = {"R": r}

        # ---- Q-wave: local min within 40ms before R ----
        q_lo = max(0, r - int(0.04 * fs))
        if q_lo < r:
            beat["Q"] = q_lo + np.argmin(filtered_signal[q_lo:r])
        else:
            beat["Q"] = None

        # ---- S-wave: local min within 60ms after R ----
        s_hi = min(len(filtered_signal), r + int(0.06 * fs))
        if r < s_hi:
            beat["S"] = r + np.argmin(filtered_signal[r:s_hi])
        else:
            beat["S"] = None

        # ---- P-wave: local max in 120-200ms window before Q ----
        ref = beat["Q"] if beat["Q"] else r
        p_lo = max(0, ref - int(0.20 * fs))
        p_hi = max(0, ref - int(0.08 * fs))
        if p_lo < p_hi:
            beat["P"] = p_lo + np.argmax(filtered_signal[p_lo:p_hi])
        else:
            beat["P"] = None

        # ---- T-wave: local max in 150-350ms window after S ----
        # window scales loosely with next RR interval when available
        ref = beat["S"] if beat["S"] else r
        rr_next = (r_peaks[i + 1] - r) if i + 1 < len(r_peaks) else int(0.8 * fs)
        t_lo = ref + int(0.10 * fs)
        t_hi = min(len(filtered_signal), ref + min(int(0.40 * fs), int(0.65 * rr_next)))
        if t_lo < t_hi:
            beat["T"] = t_lo + np.argmax(filtered_signal[t_lo:t_hi])
        else:
            beat["T"] = None

        complexes.append(beat)

    return complexes


# ---------------------------------------------------------------------------
# 4. CLINICAL PARAMETER EXTRACTION
# ---------------------------------------------------------------------------

def extract_parameters(complexes, fs):
    """
    Computes clinically meaningful intervals for each beat (in milliseconds)
    plus overall heart rate / HRV stats.
    """
    rows = []
    r_times = [c["R"] / fs for c in complexes]

    for i, c in enumerate(complexes):
        row = {"beat_index": i}

        # RR interval (this R to next R) -> heart rate
        if i + 1 < len(complexes):
            rr_sec = (complexes[i + 1]["R"] - c["R"]) / fs
            row["RR_interval_ms"] = round(rr_sec * 1000, 1)
            row["instant_HR_bpm"] = round(60.0 / rr_sec, 1) if rr_sec > 0 else None
        else:
            row["RR_interval_ms"] = None
            row["instant_HR_bpm"] = None

        # PR interval: P onset (approx = P peak) to R
        if c["P"] is not None:
            row["PR_interval_ms"] = round((c["R"] - c["P"]) / fs * 1000, 1)
        else:
            row["PR_interval_ms"] = None

        # QRS duration: Q to S
        if c["Q"] is not None and c["S"] is not None:
            row["QRS_duration_ms"] = round((c["S"] - c["Q"]) / fs * 1000, 1)
        else:
            row["QRS_duration_ms"] = None

        # QT interval: Q to T
        if c["Q"] is not None and c["T"] is not None:
            row["QT_interval_ms"] = round((c["T"] - c["Q"]) / fs * 1000, 1)
        else:
            row["QT_interval_ms"] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Overall summary stats
    valid_hr = df["instant_HR_bpm"].dropna()
    summary = {
        "mean_HR_bpm": round(valid_hr.mean(), 1) if len(valid_hr) else None,
        "HRV_SDNN_ms": round(df["RR_interval_ms"].dropna().std(), 1)
        if df["RR_interval_ms"].notna().sum() > 1
        else None,
        "mean_PR_ms": round(df["PR_interval_ms"].mean(), 1),
        "mean_QRS_ms": round(df["QRS_duration_ms"].mean(), 1),
        "mean_QT_ms": round(df["QT_interval_ms"].mean(), 1),
    }

    return df, summary


# ---------------------------------------------------------------------------
# 5. VISUALIZATION (sanity check / report figure)
# ---------------------------------------------------------------------------

def plot_pqrst(filtered_signal, fs, complexes, out_path="pqrst_plot.png", n_beats=6):
    t = np.arange(len(filtered_signal)) / fs
    plt.figure(figsize=(14, 5))
    plt.plot(t, filtered_signal, label="Filtered ECG", linewidth=1)

    colors = {"P": "green", "Q": "orange", "R": "red", "S": "purple", "T": "blue"}
    for c in complexes[:n_beats]:
        for wave, idx in c.items():
            if idx is not None:
                plt.scatter(idx / fs, filtered_signal[idx], color=colors[wave], zorder=5)
                plt.annotate(wave, (idx / fs, filtered_signal[idx]),
                             textcoords="offset points", xytext=(0, 8), fontsize=9)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Detected P-QRS-T Complex")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ECG P-QRS-T detector for BioAmp Pill data")
    parser.add_argument("input_csv", help="Path to CSV file with raw ECG samples")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling rate in Hz (default 250)")
    parser.add_argument("--powerline", type=float, default=50.0, help="Powerline freq: 50 or 60 Hz")
    parser.add_argument("--plot", action="store_true", help="Generate a P-QRS-T annotated plot")
    args = parser.parse_args()

    # --- Load data ---
    data = pd.read_csv(args.input_csv, header=None)
    raw_signal = data.iloc[:, -1].to_numpy(dtype=float)  # last column = amplitude

    # --- Pipeline ---
    filtered = preprocess(raw_signal, args.fs, powerline_freq=args.powerline)
    r_peaks = pan_tompkins_r_peaks(filtered, args.fs)
    print(f"Detected {len(r_peaks)} R-peaks")

    complexes = detect_pqrst_complex(filtered, args.fs, r_peaks)
    df, summary = extract_parameters(complexes, args.fs)

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    out_csv = "ecg_parameters.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nPer-beat parameters saved to {out_csv}")

    if args.plot:
        plot_pqrst(filtered, args.fs, complexes)


if __name__ == "__main__":
    main()
