"""
The inference path.

Everything downstream calls ECGAnalyzer.analyse(): the GUI's dataset playback
and live mode, the Excel logger and the UNO Q app. There is no second copy of
the preprocessing anywhere, because the usual way a model like this fails is
that the live path windows or normalises a beat slightly differently from the
training path and nobody notices.

Input can be at any sample rate, it gets resampled to the model's 360 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly

from . import config as C
from . import data as D
from . import qrs


@dataclass
class BeatResult:
    """One classified heartbeat."""
    sample: int                  # R-peak index in the (360 Hz) signal
    time_s: float
    label: str
    class_idx: int
    confidence: float
    probs: np.ndarray
    rr_s: float
    features: np.ndarray
    is_abnormal: bool = field(init=False)

    def __post_init__(self):
        self.is_abnormal = self.label != "NOR"


class ECGAnalyzer:
    """The INT8 ONNX model plus the preprocessing it expects."""

    def __init__(self, model_path=None, providers=("CPUExecutionProvider",)):
        self.model_path = str(model_path or (C.MODEL_DIR / "model_int8.onnx"))
        self.session = ort.InferenceSession(self.model_path,
                                            providers=list(providers))

    # ------------------------------------------------------------------
    @staticmethod
    def to_model_rate(sig, fs):
        """Resample to 360 Hz. The BioAmp bridge streams 125 and CPSC is 500,
        but the model only ever sees 360."""
        sig = np.asarray(sig, dtype=np.float64)
        if fs == C.FS:
            return sig, C.FS
        from math import gcd
        g = gcd(int(fs), C.FS)
        return resample_poly(sig, C.FS // g, int(fs) // g), C.FS

    # ------------------------------------------------------------------
    def analyse(self, sig, fs=C.FS, r_peaks=None, filter_signal=True):
        """
        Classify every complete beat in `sig`.

        r_peaks : optional, in samples of the original `sig` before resampling.
                  Pass MIT-BIH annotations to separate model error from
                  detector error, or leave None to run the full
                  detect-then-classify path the hardware uses.

        Returns (beats, signal_360hz).
        """
        sig, _ = self.to_model_rate(sig, fs)

        if r_peaks is None:
            peaks = qrs.detect_r_peaks(sig, C.FS)
        else:
            scale = C.FS / fs
            peaks = np.unique((np.asarray(r_peaks) * scale).astype(np.int64))

        proc = D.bandpass(sig) if filter_signal else sig

        if len(peaks) < C.RR_LOCAL_WINDOW + 2:
            return [], proc

        feats, valid = D.rr_features(peaks, C.FS)

        rows, keep = [], []
        for i, p in enumerate(peaks):
            if not valid[i]:
                continue
            if p - C.BEFORE < 0 or p + C.AFTER > len(proc):
                continue
            rows.append(D.preprocess_beat(proc[p - C.BEFORE:p + C.AFTER]))
            keep.append(i)

        if not rows:
            return [], proc

        X = np.asarray(rows, dtype=np.float32)[:, None, :]
        F = feats[keep].astype(np.float32)
        logits = self.session.run(["logits"],
                                  {"beat": X, "rr_features": F})[0]
        probs = _softmax(logits)
        idx = probs.argmax(1)

        rr = np.diff(peaks, prepend=peaks[0]) / C.FS

        beats = []
        for j, i in enumerate(keep):
            beats.append(BeatResult(
                sample=int(peaks[i]),
                time_s=float(peaks[i]) / C.FS,
                label=C.CLASSES[idx[j]],
                class_idx=int(idx[j]),
                confidence=float(probs[j, idx[j]]),
                probs=probs[j],
                rr_s=float(rr[i]),
                features=F[j],
            ))
        return beats, proc


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Incremental / streaming analysis
# --------------------------------------------------------------------------
class StreamAnalyzer:
    """
    Feed samples in, get newly-classified beats out.

    Used by both the GUI's dataset replay and live sensor mode. Replay could
    have analysed the whole record up front and animated the answers, which
    would look the same on screen, but then the demo wouldn't exercise the code
    path the device runs and any bug that only shows up when beats arrive
    incrementally would stay hidden until the hardware demo.

    Keeps a trailing buffer, re-analyses it periodically, and only emits beats
    it hasn't emitted before. A beat is held back until it's far enough from
    the leading edge for its full window and RR history to exist, which is the
    same latency the real device has.
    """

    def __init__(self, analyzer, fs=C.FS, buffer_s=20.0, reanalyse_s=0.5):
        self.analyzer = analyzer
        self.fs = fs
        self.buffer_n = int(buffer_s * fs)
        self.reanalyse_n = int(reanalyse_s * fs)

        self.buf = np.zeros(0, dtype=np.float64)
        self.offset = 0            # absolute index of buf[0]
        self.total = 0             # absolute samples seen
        self.since_analysis = 0
        self.emitted = set()
        self._last_emitted = -10**9   # absolute index of the newest beat out

    def reset(self):
        self.buf = np.zeros(0, dtype=np.float64)
        self.offset = self.total = self.since_analysis = 0
        self.emitted.clear()
        self._last_emitted = -10**9

    def push(self, samples):
        """Append new samples; return the list of beats newly classified."""
        samples = np.asarray(samples, dtype=np.float64).ravel()
        if len(samples) == 0:
            return []

        self.buf = np.concatenate([self.buf, samples])
        self.total += len(samples)
        self.since_analysis += len(samples)

        if len(self.buf) > self.buffer_n:
            drop = len(self.buf) - self.buffer_n
            self.buf = self.buf[drop:]
            self.offset += drop

        if self.since_analysis < self.reanalyse_n:
            return []
        self.since_analysis = 0

        # Not enough history for the 10-beat RR window yet.
        if len(self.buf) < self.fs * 8:
            return []

        beats, _ = self.analyzer.analyse(self.buf, self.fs)

        # analyse() resamples to 360 Hz internally, so its b.sample values
        # index the resampled buffer while self.buf and self.offset are in
        # input samples. Mixing the two gave negative durations and impossible
        # heart rates on any input that wasn't already 360 Hz, i.e. every real
        # BioAmp stream. Convert before combining.
        to_input = self.fs / C.FS
        n_model = len(self.buf) / to_input
        edge_model = n_model - C.AFTER      # incomplete windows live past here

        # Dedup on exact index isn't enough. Each analysis sees a different
        # slice, so the adaptive threshold can place the same beat a sample or
        # two from where it put it last time, and the shifted copy gets emitted
        # again out of order. That's what produced episodes whose end came
        # before their start. Gate on the refractory period instead: two beats
        # can't be 200 ms apart, so anything closer is the same beat.
        min_gap = int(0.20 * self.fs)
        new = []
        for b in sorted(beats, key=lambda x: x.sample):
            if b.sample >= edge_model:
                continue
            absolute = int(round(b.sample * to_input)) + self.offset
            if absolute <= self._last_emitted + min_gap:
                continue
            self._last_emitted = absolute
            self.emitted.add(absolute)
            b.sample = absolute
            b.time_s = absolute / self.fs
            new.append(b)

        # Keep the emitted set from growing without bound on long sessions.
        if len(self.emitted) > 20000:
            cutoff = self.offset - self.buffer_n
            self.emitted = {s for s in self.emitted if s >= cutoff}

        return new


# --------------------------------------------------------------------------
# Episode grouping - what the GUI actually highlights
# --------------------------------------------------------------------------
@dataclass
class Episode:
    """A run of consecutive abnormal beats with the same label. The GUI marks
    episodes rather than individual beats and the log gets one row each. How
    many beats it takes is per-class, see C.MIN_EPISODE_BEATS."""
    label: str
    start_time: float
    end_time: float
    n_beats: int
    mean_confidence: float
    mean_hr: float
    start_sample: int
    end_sample: int

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def low_confidence(self):
        return self.mean_confidence < C.LOW_CONFIDENCE


def group_episodes(beats, min_beats=None, min_confidence=None,
                   max_episode_s=None):
    """
    Collapse consecutive same-label abnormal beats into episodes.

    min_beats : None uses the per-class thresholds in config (PVC fires on one
                beat, sustained rhythms need three). Pass an int to override.

    min_confidence : None uses the per-class MIN_EPISODE_CONFIDENCE map, which
                was measured on MIT-BIH validation patients and is permissive
                on purpose, since on clean database signal a gate costs more
                recall than it buys precision.

                That doesn't hold for live sensor input, where mean confidence
                is about 0.40 against a 0.20 chance level. Pass a float (live
                mode passes C.LIVE_MIN_EPISODE_CONFIDENCE) to apply one
                stricter gate to every class.
    """
    episodes, run = [], []

    def threshold(label):
        if min_beats is not None:
            return min_beats
        return C.MIN_EPISODE_BEATS.get(label, C.DEFAULT_MIN_EPISODE_BEATS)

    def flush():
        if run and len(run) >= threshold(run[0].label):
            conf = float(np.mean([b.confidence for b in run]))
            gate = (min_confidence if min_confidence is not None
                    else C.MIN_EPISODE_CONFIDENCE.get(run[0].label, 0.0))
            if conf < gate:
                run.clear()
                return
            rrs = [b.rr_s for b in run if b.rr_s > 0]
            episodes.append(Episode(
                label=run[0].label,
                start_time=run[0].time_s,
                end_time=run[-1].time_s,
                n_beats=len(run),
                mean_confidence=float(np.mean([b.confidence for b in run])),
                mean_hr=float(60.0 / np.mean(rrs)) if rrs else 0.0,
                start_sample=run[0].sample,
                end_sample=run[-1].sample,
            ))
        run.clear()

    chunk_s = max_episode_s if max_episode_s is not None else C.MAX_EPISODE_S

    for b in beats:
        same = run and b.label == run[0].label
        # Split a long sustained run so it finalises in bounded chunks.
        too_long = same and (b.time_s - run[0].time_s) >= chunk_s
        if b.is_abnormal and same and not too_long:
            run.append(b)
        else:
            flush()
            if b.is_abnormal:
                run.append(b)
    flush()
    return episodes
