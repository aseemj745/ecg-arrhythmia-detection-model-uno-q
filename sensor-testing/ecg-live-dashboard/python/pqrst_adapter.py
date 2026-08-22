"""
Adapter: makes ecg_pqrst_detector.py usable inside ecg_gui.py
================================================================

ecg_pqrst_detector.py is a standalone command-line script (it takes a CSV
path via argparse and runs everything in main()). The GUI, however, expects
a module with a simple `analyze(signal, fs) -> dict` function it can call
directly on an already-loaded signal.

This file bridges the two: it imports the real detector's pipeline
functions and calls them in the same order main() does, then reshapes the
output into the dict format ecg_gui.py understands (scalar results +
"peaks" + "annotations" for plotting).

SETUP:
  Put this file in the SAME FOLDER as ecg_pqrst_detector.py.

IN THE GUI:
  Set "Analysis module (.py)" to the path of THIS file (not
  ecg_pqrst_detector.py itself), then click "Run Module".

You'll see, in the results panel:
  - mean_HR_bpm, HRV_SDNN_ms, mean_PR_ms, mean_QRS_ms, mean_QT_ms
  - Num R-peaks detected
  - path to the saved per-beat CSV (same ecg_parameters.csv the original
    script writes)
And on the plot:
  - red dots  = R peaks
  - colored x = P, Q, S, T waves for each detected beat
"""

import os
import sys

import numpy as np

# Make sure ecg_pqrst_detector.py (expected to sit next to this file) is
# importable regardless of the current working directory.
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

try:
    import ecg_pqrst_detector as pqrst
except ImportError as e:
    raise ImportError(
        "Could not import ecg_pqrst_detector.py. Make sure it is saved in "
        f"the same folder as this adapter file ({_this_dir})."
    ) from e


# Powerline frequency used for the notch filter. Change to 60.0 if you're
# in a 60Hz region (e.g. North America).
POWERLINE_FREQ_HZ = 50.0


def analyze(signal: np.ndarray, fs: float) -> dict:
    # --- 1. Preprocess (bandpass + notch filter) ---
    filtered = pqrst.preprocess(signal, fs, powerline_freq=POWERLINE_FREQ_HZ)

    # --- 2. Pan-Tompkins R-peak detection ---
    r_peaks = pqrst.pan_tompkins_r_peaks(filtered, fs)

    # --- 3. P/Q/S/T detection around each R-peak ---
    complexes = pqrst.detect_pqrst_complex(filtered, fs, r_peaks)

    # --- 4. Clinical interval extraction (per-beat table + summary) ---
    df, summary = pqrst.extract_parameters(complexes, fs)

    # Save the per-beat table to disk, same as the original CLI script does.
    out_csv = os.path.join(_this_dir, "ecg_parameters.csv")
    try:
        df.to_csv(out_csv, index=False)
        csv_note = out_csv
    except Exception as e:
        csv_note = f"(could not save: {e})"

    # --- Reshape into the GUI's expected dict format ---
    results = dict(summary)
    results["Num R-peaks detected"] = len(r_peaks)
    results["Per-beat parameters CSV"] = csv_note

    annotations = {}
    for wave in ("P", "Q", "S", "T"):
        idxs = [c[wave] for c in complexes if c.get(wave) is not None]
        if idxs:
            annotations[wave] = idxs

    results["peaks"] = [int(p) for p in r_peaks]       # -> red dots (R)
    results["annotations"] = annotations                # -> P, Q, S, T markers

    return results
