"""
MIT-BIH exploration, visual version. Everything is compared against normal,
since "abnormal" only means something next to "normal".

Produces 4 figures:

  Fig 3  Average beat shape, each class drawn on top of normal
  Fig 4  10-second strips, all 5 classes one under the other
  Fig 5  Tachogram, RR interval beat after beat, normal vs AFib
  Fig 6  Poincare plot, the classic AFib picture

Also prints rhythm statistics using successive RR differences rather than
plain standard deviation.

"NOR" here is just this script's shorthand for a normal beat during normal
sinus rhythm. It isn't an official PhysioNet code.
"""

import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import wfdb

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "data/mitdb"
PLOT_DIR = "plots"
FS = 360

ALL_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221', '222', '223',
    '228', '230', '231', '232', '233', '234',
]
PACED_RECORDS = {'102', '104', '107', '217'}

BEAT_SYMBOLS = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
    'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?',
}

TARGET_CLASSES = ['NOR', 'LBBB', 'RBBB', 'PVC', 'AFIB']

# Beat window around the R-peak. These numbers matter: a heartbeat's useful
# information sits roughly 0.3 s before the R-peak (the P wave and QRS onset)
# and 0.5 s after (the S wave and T wave). This is close to the window we
# will actually use to train the model in Phase 2.
BEFORE = int(0.30 * FS)     # 108 samples
AFTER = int(0.50 * FS)      # 180 samples

MAX_BEATS_PER_CLASS = 400   # how many beats to average for the mean shape


# ---------------------------------------------------------------------------
# BEAT LABELLING  (same logic as the first script)
# ---------------------------------------------------------------------------
def label_beats_in_record(record_name):
    ann = wfdb.rdann(os.path.join(DATA_DIR, record_name), 'atr')
    beats = []
    current_rhythm = None

    for sample, symbol, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        aux_clean = aux.strip().strip('\x00').strip()
        if aux_clean.startswith('('):
            current_rhythm = aux_clean

        if symbol not in BEAT_SYMBOLS:
            continue

        cls = None
        if current_rhythm == '(AFIB':
            cls = 'AFIB'
        elif symbol == 'L':
            cls = 'LBBB'
        elif symbol == 'R':
            cls = 'RBBB'
        elif symbol == 'V':
            cls = 'PVC'
        elif symbol == 'N' and current_rhythm in ('(N', None):
            cls = 'NOR'

        beats.append({'sample': int(sample), 'symbol': symbol,
                      'rhythm': current_rhythm, 'cls': cls})
    return beats


def znorm(x):
    """
    Z-normalisation: subtract the mean, divide by the standard deviation.

    WHY THIS MATTERS (this is a real preprocessing step, not just for plots):
    different patients and different electrode placements give wildly
    different signal amplitudes. One person's R-peak might be 1 mV, another's
    2.5 mV, for exactly the same condition. If we feed raw millivolts to the
    model it can waste capacity learning amplitude differences that mean
    nothing clinically. Z-normalising each beat throws away the amplitude and
    keeps the SHAPE, which is what actually carries the diagnosis.
    """
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else x - x.mean()


# ---------------------------------------------------------------------------
# COLLECT BEATS AND RR DATA
# ---------------------------------------------------------------------------
def collect():
    usable = [r for r in ALL_RECORDS if r not in PACED_RECORDS]

    beat_windows = defaultdict(list)      # class -> list of z-normed windows
    per_record_counts = {}                # record -> Counter of classes
    succ_diffs = defaultdict(list)        # class -> |RR(n) - RR(n-1)|
    rr_series = {}                        # record -> list of RR intervals

    print(f"Reading {len(usable)} records (this takes a minute)...\n")

    for rec in usable:
        beats = label_beats_in_record(rec)
        counts = Counter(b['cls'] for b in beats if b['cls'])
        per_record_counts[rec] = counts

        # ---- RR intervals and successive differences -------------------
        rrs = []
        for i in range(1, len(beats)):
            rr = (beats[i]['sample'] - beats[i - 1]['sample']) / FS
            rrs.append(rr if 0.2 < rr < 3.0 else np.nan)
        rr_series[rec] = rrs

        for i in range(2, len(beats)):
            cls = beats[i]['cls']
            if cls is None:
                continue
            rr_now, rr_prev = rrs[i - 1], rrs[i - 2]
            if np.isnan(rr_now) or np.isnan(rr_prev):
                continue
            succ_diffs[cls].append(abs(rr_now - rr_prev))

        # ---- beat windows for the average-shape figure -----------------
        needed = [c for c in TARGET_CLASSES
                  if len(beat_windows[c]) < MAX_BEATS_PER_CLASS
                  and counts.get(c, 0) > 0]
        if not needed:
            continue

        record = wfdb.rdrecord(os.path.join(DATA_DIR, rec))
        sig = record.p_signal[:, 0]     # channel 0 = MLII on most records

        taken = Counter()
        for b in beats:
            c = b['cls']
            if c not in needed or taken[c] >= 60:
                continue
            s = b['sample']
            if s - BEFORE < 0 or s + AFTER >= len(sig):
                continue
            if len(beat_windows[c]) >= MAX_BEATS_PER_CLASS:
                continue
            beat_windows[c].append(znorm(sig[s - BEFORE:s + AFTER]))
            taken[c] += 1

    return beat_windows, per_record_counts, succ_diffs, rr_series


def best_record_for(per_record_counts, cls):
    """Record containing the most beats of this class."""
    return max(per_record_counts, key=lambda r: per_record_counts[r].get(cls, 0))


# ---------------------------------------------------------------------------
# CORRECTED STATISTICS
# ---------------------------------------------------------------------------
def print_corrected_stats(succ_diffs):
    print("=" * 72)
    print("CORRECTED RHYTHM STATISTIC: successive RR differences")
    print("=" * 72)
    print("Plain standard deviation was the wrong tool - it mixes together")
    print("differences BETWEEN patients (resting heart rates differ) with")
    print("beat-to-beat jitter WITHIN a patient. AFib is only the second kind.")
    print()
    print("So instead we measure, for every beat: how different is this RR")
    print("interval from the one immediately before it? Sustained beat-to-beat")
    print("irregularity is exactly what atrial fibrillation is.")
    print()
    print(f"{'Class':<8}{'mean |dRR|':>13}{'median':>10}{'90th pct':>11}{'n':>10}")
    print("-" * 72)
    for c in TARGET_CLASSES:
        d = np.array(succ_diffs[c])
        if len(d) == 0:
            continue
        print(f"{c:<8}{d.mean():>13.4f}{np.median(d):>10.4f}"
              f"{np.percentile(d, 90):>11.4f}{len(d):>10,}")
    print()
    print("Expected reading: AFIB clearly highest. PVC also raised, because a")
    print("premature beat arrives early and breaks the spacing - but PVC")
    print("irregularity is occasional, while AFib's is continuous. That")
    print("difference is why the model gets the beat shape AND the timing.")


# ---------------------------------------------------------------------------
# FIGURE 3 - average beat shape, each class overlaid on normal
# ---------------------------------------------------------------------------
def plot_mean_beats(beat_windows):
    normal = np.array(beat_windows['NOR'])
    nor_mean = normal.mean(axis=0)

    others = [c for c in TARGET_CLASSES if c != 'NOR']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    t = (np.arange(-BEFORE, AFTER)) / FS

    for ax, cls in zip(axes.ravel(), others):
        arr = np.array(beat_windows[cls])
        m, sd = arr.mean(axis=0), arr.std(axis=0)

        # grey reference = normal
        ax.plot(t, nor_mean, color='0.45', linewidth=2.0,
                linestyle='--', label=f'NORMAL (n={len(normal)})')
        # coloured = this class, with a shaded band showing beat-to-beat spread
        ax.plot(t, m, color='tab:red', linewidth=2.0,
                label=f'{cls} (n={len(arr)})')
        ax.fill_between(t, m - sd, m + sd, color='tab:red', alpha=0.18)

        ax.axvline(0, color='k', linewidth=0.7, alpha=0.4)
        ax.set_title(f"{cls} vs NORMAL", loc='left', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("seconds from R-peak")
    for ax in axes[:, 0]:
        ax.set_ylabel("amplitude (z-normalised)")

    fig.suptitle("Figure 3 - Average beat shape: each class drawn over NORMAL\n"
                 "(amplitude normalised, so you are comparing SHAPE only)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "fig3_mean_beats_vs_normal.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


# ---------------------------------------------------------------------------
# FIGURE 4 - 10 second strips, all five classes stacked
# ---------------------------------------------------------------------------
def plot_all_strips(per_record_counts):
    fig, axes = plt.subplots(len(TARGET_CLASSES), 1, figsize=(13, 12))
    window = 10 * FS

    for ax, cls in zip(axes, TARGET_CLASSES):
        rec = best_record_for(per_record_counts, cls)
        record = wfdb.rdrecord(os.path.join(DATA_DIR, rec))
        sig = record.p_signal[:, 0]
        beats = [b for b in label_beats_in_record(rec) if b['cls'] == cls]

        start = beats[len(beats) // 2]['sample']
        start = min(start, len(sig) - window - 1)
        seg = sig[start:start + window]
        t = np.arange(len(seg)) / FS

        colour = 'tab:blue' if cls == 'NOR' else 'tab:red'
        ax.plot(t, seg, linewidth=0.9, color=colour)

        pts = [(b['sample'] - start) / FS for b in beats
               if start <= b['sample'] < start + window]
        for p in pts:
            ax.plot(p, sig[start + int(p * FS)], 'o', color='k', markersize=3)

        ax.set_title(f"{cls}   (record {rec})", loc='left', fontsize=10)
        ax.set_ylabel("mV")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("seconds")
    fig.suptitle("Figure 4 - Ten seconds of each class, NORMAL at the top "
                 "for comparison", fontsize=12)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "fig4_strips_all_classes.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# FIGURE 5 - tachogram
# ---------------------------------------------------------------------------
def plot_tachogram(per_record_counts, rr_series):
    """
    A tachogram plots the RR interval (gap between beats) against beat number.
    Forget the ECG waveform for a moment - this is PURE TIMING.

    Normal sinus rhythm: a fairly flat line, gently drifting with breathing.
    AFib: a jagged mess, every beat a different distance from the last.

    If you only look at one picture to understand AFib, look at this one.
    """
    nor_rec = best_record_for(per_record_counts, 'NOR')
    afib_rec = best_record_for(per_record_counts, 'AFIB')

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharey=True)
    n_beats = 120

    for ax, (rec, name, colour) in zip(
            axes, [(nor_rec, 'NORMAL sinus rhythm', 'tab:blue'),
                   (afib_rec, 'ATRIAL FIBRILLATION', 'tab:red')]):
        beats = label_beats_in_record(rec)
        target = 'NOR' if 'NORMAL' in name else 'AFIB'
        idx = [i for i, b in enumerate(beats) if b['cls'] == target]
        start = idx[len(idx) // 2]

        rrs = rr_series[rec][start:start + n_beats]
        ax.plot(range(len(rrs)), rrs, 'o-', markersize=3,
                linewidth=1.0, color=colour)
        ax.set_title(f"{name}   (record {rec})", loc='left', fontsize=11)
        ax.set_ylabel("RR interval (s)")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("beat number")
    fig.suptitle("Figure 5 - Tachogram: the gap between beats, beat after beat\n"
                 "Flat-ish = regular rhythm.  Jagged = atrial fibrillation.",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "fig5_tachogram.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# FIGURE 6 - Poincare plot
# ---------------------------------------------------------------------------
def plot_poincare(per_record_counts, rr_series):
    """
    The Poincare plot is the classic cardiology picture for rhythm.
    Each dot is one beat: x = the gap before it, y = the gap after it.

    Regular rhythm -> every gap is similar to the next -> a tight little cloud.
    AFib -> gaps are unrelated to each other -> a big diffuse smear.

    This is a picture of the SAME information the RR features give the model.
    """
    nor_rec = best_record_for(per_record_counts, 'NOR')
    afib_rec = best_record_for(per_record_counts, 'AFIB')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)

    for ax, (rec, name, colour) in zip(
            axes, [(nor_rec, 'NORMAL sinus rhythm', 'tab:blue'),
                   (afib_rec, 'ATRIAL FIBRILLATION', 'tab:red')]):
        rrs = np.array(rr_series[rec], dtype=float)
        x, y = rrs[:-1], rrs[1:]
        ok = ~(np.isnan(x) | np.isnan(y))
        ax.scatter(x[ok], y[ok], s=6, alpha=0.35, color=colour)
        ax.set_title(f"{name}   (record {rec})", fontsize=11)
        ax.set_xlabel("RR interval n (s)")
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    axes[0].set_ylabel("RR interval n+1 (s)")
    fig.suptitle("Figure 6 - Poincare plot: tight cluster = regular, "
                 "wide smear = atrial fibrillation", fontsize=12)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "fig6_poincare.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    beat_windows, per_record_counts, succ_diffs, rr_series = collect()

    print_corrected_stats(succ_diffs)

    plot_mean_beats(beat_windows)
    plot_all_strips(per_record_counts)
    plot_tachogram(per_record_counts, rr_series)
    plot_poincare(per_record_counts, rr_series)

    print("\nDone. Four figures saved in the 'plots' folder.")
    plt.show()


if __name__ == "__main__":
    main()