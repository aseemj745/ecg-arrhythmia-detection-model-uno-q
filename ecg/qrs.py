"""
Pan-Tompkins R-peak detection.

Needed in two places: the GUI demo, so a MIT-BIH record goes through the same
path the hardware uses instead of being handed the answers from its annotation
file, and live BioAmp input, where there are no annotations at all.

On the finished product the STM32U585 does this and the Linux side only gets
R-peak positions. This is the reference version and the fallback.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from . import config as C


def _bandpass(sig, fs, lo=5.0, hi=15.0, order=2):
    """5-15 Hz is where QRS energy sits. P waves, T waves and baseline wander
    mostly fall outside it, which is the whole trick."""
    nyq = 0.5 * fs
    hi = min(hi, nyq * 0.95)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def detect_r_peaks(sig, fs=C.FS, refractory_ms=200, integ_ms=150):
    """
    Return R-peak sample indices.

    bandpass -> derivative -> square -> moving-window integration -> adaptive
    threshold with a refractory period. The refractory period is what stops a
    tall T wave being counted as a second beat.
    """
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < fs:
        return np.array([], dtype=np.int64)

    filt = _bandpass(sig, fs)
    diff = np.diff(filt, prepend=filt[0])
    sq = diff ** 2

    win = max(1, int(integ_ms * fs / 1000))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")

    # Track running signal and noise levels the way Pan-Tompkins does, so the
    # detector survives amplitude drift.
    thresh = 0.3 * np.mean(integ) + 0.2 * np.std(integ)
    refractory = int(refractory_ms * fs / 1000)

    peaks = []
    i, n = 1, len(integ) - 1
    while i < n:
        if integ[i] > thresh and integ[i] >= integ[i - 1] and integ[i] >= integ[i + 1]:
            peaks.append(i)
            # adapt towards the level of accepted peaks
            thresh = 0.875 * thresh + 0.125 * (0.4 * integ[i])
            i += refractory
        else:
            i += 1

    if not peaks:
        return np.array([], dtype=np.int64)

    # The integrator delays and smears the peak, so snap each detection onto
    # the real extremum of the raw trace within +/-100 ms. The beat window we
    # cut later has to be centred on the actual R-peak.
    snap = int(0.10 * fs)
    refined = []
    for p in peaks:
        lo, hi = max(0, p - snap), min(len(sig), p + snap)
        if hi <= lo:
            continue
        seg = sig[lo:hi] - np.mean(sig[lo:hi])
        refined.append(lo + int(np.argmax(np.abs(seg))))

    out = np.unique(np.array(refined, dtype=np.int64))
    # drop anything closer than the refractory period after snapping
    if len(out) > 1:
        keep = [out[0]]
        for p in out[1:]:
            if p - keep[-1] >= refractory:
                keep.append(p)
        out = np.array(keep, dtype=np.int64)
    return out
