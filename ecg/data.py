"""
MIT-BIH loading, beat labelling, windowing and rhythm features.

One function, `extract_record`, turns a record name into (waveforms, features,
labels). The live pipeline reuses the same windowing and feature code, so what
the model sees at inference is built the same way the training data was.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from . import config as C

# wfdb is imported lazily inside the functions that read MIT-BIH files. The
# board only runs the live path and has no dataset, so importing it up here
# would push a dependency onto the UNO Q for nothing.

NUL = chr(0)


# --------------------------------------------------------------------------
# Signal conditioning
# --------------------------------------------------------------------------
def _bandpass_coeffs(fs):
    lo, hi = C.BANDPASS
    nyq = 0.5 * fs
    return butter(C.BANDPASS_ORDER, [lo / nyq, hi / nyq], btype="band")


def bandpass(sig, fs=C.FS):
    """Zero-phase 0.5-40 Hz bandpass. Zero-phase matters, a causal filter would
    shift the R-peak off its annotated sample."""
    if C.BANDPASS is None:
        return sig
    b, a = _bandpass_coeffs(fs)
    return filtfilt(b, a, sig)


def znorm(beat):
    """Per-beat z-normalisation, so the model learns shape and not amplitude.
    Electrode placement and skin impedance swing the absolute mV a lot between
    patients, and between MIT-BIH and the BioAmp."""
    sd = beat.std()
    return (beat - beat.mean()) / (sd if sd > 1e-6 else 1e-6)


def polarity_normalise(beat, r_index=C.BEFORE, fs=C.FS):
    """
    Flip the beat if its dominant QRS deflection points down.

    Whether a QRS shows up as a tall R or a deep QS depends on where the
    electrodes sit, not on what the heart is doing. MIT-BIH has LBBB in both
    polarities (207 inverted, 109/111 upright), and with only 4 LBBB patients
    there's no way for the model to learn that invariance, so we impose it.

    The same rule runs unchanged on live input, where a swapped electrode pair
    gives exactly this inversion.
    """
    if not C.POLARITY_NORMALISE:
        return beat
    half = max(1, int(C.POLARITY_WINDOW_MS * fs / 1000))
    lo = max(0, r_index - half)
    hi = min(len(beat), r_index + half + 1)
    seg = beat[lo:hi]
    if seg[np.argmax(np.abs(seg))] < 0:
        return -beat
    return beat


def preprocess_beat(window):
    """
    z-normalise, then fix polarity. Training data and live sensor data both go
    through this one function, otherwise the model sees a different
    distribution at inference than it trained on. Mean removal comes first so a
    DC offset can't fool the polarity test.
    """
    return polarity_normalise(znorm(window))


# --------------------------------------------------------------------------
# Lead selection
# --------------------------------------------------------------------------
def find_lead_channel(header, lead=C.TARGET_LEAD):
    """Return the channel index holding `lead`, or None. Picking by name is
    what stops record 114 (MLII in channel 1) from quietly contributing a
    chest-lead view to an otherwise limb-lead dataset."""
    for i, name in enumerate(header.sig_name):
        if name.strip() == lead:
            return i
    return None


# --------------------------------------------------------------------------
# Beat labelling
# --------------------------------------------------------------------------
def label_beats(record_name):
    """
    Read the annotation file and return one dict per heartbeat:
        sample, symbol, rhythm, cls
    `cls` is None when the beat is not one of our five classes and should be
    discarded.

    Labelling rules:
      AFIB          - beat sits inside an "(AFIB" rhythm annotation, whatever
                      its own symbol says. AFib is irregular timing and its
                      beats mostly look normal, so the symbol can't find it.
      LBBB/RBBB/PVC - beat symbol L / R / V.
      NOR           - symbol N and the prevailing rhythm is normal or unset.
                      Normal-looking beats inside bigeminy, flutter and other
                      abnormal rhythms get dropped rather than called normal.
    """
    import wfdb
    ann = wfdb.rdann(str(C.DATA_DIR / record_name), "atr")
    beats = []
    rhythm = None

    for sample, symbol, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        aux_clean = aux.strip().strip(NUL).strip()
        if aux_clean.startswith("("):
            rhythm = aux_clean
        if symbol not in C.BEAT_SYMBOLS:
            continue

        cls = None
        if rhythm == "(AFIB" and C.AFIB_OVERRIDES_ALL_SYMBOLS:
            cls = "AFIB"
        elif symbol == "L":
            cls = "LBBB"
        elif symbol == "R":
            cls = "RBBB"
        elif symbol == "V":
            cls = "PVC"
        elif rhythm == "(AFIB":
            cls = "AFIB"
        elif symbol == "N" and rhythm in ("(N", None):
            cls = "NOR"

        beats.append({"sample": int(sample), "symbol": symbol,
                      "rhythm": rhythm, "cls": cls})
    return beats


# --------------------------------------------------------------------------
# Rhythm features
# --------------------------------------------------------------------------
def rhythm_features(window_rr, rr_prev, rr_prev2):
    """
    Turn a window of recent RR intervals into the model's numeric input.

    `window_rr` is the last RR_LOCAL_WINDOW intervals including the current
    one, oldest first. `rr_prev` is the current interval, `rr_prev2` the one
    before it.

    Its own function because the live pipeline calls this same code from a ring
    buffer, so training and deployment can't drift apart. Returns None when the
    window isn't usable yet.
    """
    w = np.asarray(window_rr, dtype=np.float64)
    if len(w) < C.RR_LOCAL_WINDOW or not np.all(np.isfinite(w)):
        return None

    mean = w.mean()
    median = np.median(w)
    if mean <= 0 or median <= 0:
        return None

    prior = w[:-1]                      # window excluding the current beat
    diffs = np.diff(w)

    # Ectopy-robust copy. A PVC gives a short interval then a compensatory long
    # one, and to a plain rmssd that pair looks like AFib. Median-replacing the
    # outliers flattens it but leaves AFib alone, since AFib is broad dispersion
    # not two spikes. Without this 93% of record 233's normal beats read AFib.
    clean = np.where(np.abs(w - median) / median > C.ECTOPIC_TOL, median, w)
    clean_diffs = np.diff(clean)

    return np.array([
        rr_prev,
        rr_prev - rr_prev2,
        rr_prev / prior.mean(),
        np.sqrt(np.mean(diffs ** 2)),
        w.std() / mean,
        float(np.mean(np.abs(diffs) > 0.05)),
        (w.max() - w.min()) / mean,
        np.mean(np.abs(w - median)) / median,
        np.sqrt(np.mean(clean_diffs ** 2)),
        clean.std() / clean.mean(),
    ], dtype=np.float32)


def rr_features(samples, fs=C.FS):
    """
    Build the causal timing features for a whole series of R-peak positions.

    Returns (features, valid_mask). The first RR_LOCAL_WINDOW+1 beats of a
    record are marked invalid, their window isn't filled yet.
    """
    n = len(samples)
    feats = np.zeros((n, C.N_FEATURES), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    rr = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        rr[1:] = np.diff(samples) / fs
    rr = np.clip(rr, *C.RR_CLIP)

    w = C.RR_LOCAL_WINDOW
    for i in range(w + 1, n):
        f = rhythm_features(rr[i - w + 1:i + 1], rr[i], rr[i - 1])
        if f is None or not np.all(np.isfinite(f)):
            continue
        feats[i] = f
        valid[i] = True

    return feats, valid


# --------------------------------------------------------------------------
# Full record extraction
# --------------------------------------------------------------------------
def extract_record(record_name, verbose=False):
    """
    Turn one MIT-BIH record into training rows.

    Returns a dict with
        X       (n, 288) float32  z-normalised MLII beat windows
        F       (n, 3)   float32  causal rhythm features
        y       (n,)     int64    class index
        sample  (n,)     int64    R-peak position within the record
        record  (n,)     int64    record number, for patient-level splitting
    or None if the record has no usable MLII channel or no labelled beats.
    """
    import wfdb
    path = str(C.DATA_DIR / record_name)
    header = wfdb.rdheader(path)
    ch = find_lead_channel(header)
    if ch is None:
        if verbose:
            print(f"  [skip] {record_name}: no {C.TARGET_LEAD} lead "
                  f"(channels: {header.sig_name})")
        return None

    rec = wfdb.rdrecord(path)
    sig = bandpass(rec.p_signal[:, ch].astype(np.float64))

    beats = label_beats(record_name)
    if not beats:
        return None

    samples = np.array([b["sample"] for b in beats], dtype=np.int64)
    feats, valid = rr_features(samples)

    X, F, y, S = [], [], [], []
    for i, b in enumerate(beats):
        if b["cls"] is None or not valid[i]:
            continue
        s = b["sample"]
        if s - C.BEFORE < 0 or s + C.AFTER > len(sig):
            continue
        X.append(preprocess_beat(sig[s - C.BEFORE:s + C.AFTER]))
        F.append(feats[i])
        y.append(C.CLASS_TO_IDX[b["cls"]])
        S.append(s)

    if not X:
        return None

    out = {
        "X": np.asarray(X, dtype=np.float32),
        "F": np.asarray(F, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "sample": np.asarray(S, dtype=np.int64),
        "record": np.full(len(X), int(record_name), dtype=np.int64),
    }
    if verbose:
        counts = {c: int((out["y"] == i).sum())
                  for i, c in enumerate(C.CLASSES)}
        counts = {k: v for k, v in counts.items() if v}
        print(f"  {record_name}  lead ch{ch}  {len(X):>6} beats   {counts}")
    return out


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------
def load_dataset(path=None):
    """Load the npz produced by step1_build_dataset.py."""
    path = path or (C.DATASET_DIR / "beats.npz")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def mask_for(record_ids, records):
    """Boolean mask selecting rows whose record is in `records`."""
    wanted = np.array([int(r) for r in records], dtype=np.int64)
    return np.isin(record_ids, wanted)
