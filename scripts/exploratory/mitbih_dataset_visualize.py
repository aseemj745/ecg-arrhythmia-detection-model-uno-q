"""
=============================================================================
PHASE 1 - MIT-BIH DATA EXPLORATION
=============================================================================
Goal of this script (it does NOT train anything):

  1. Download the MIT-BIH Arrhythmia Database (~100 MB, one time only)
  2. Count how many beats we have for each of our 5 target classes
  3. Count how many PATIENTS contribute to each class  <-- the number that
     actually decides whether we have "enough data"
  4. Show, numerically, why AFib needs RR-interval timing and not beat shape
  5. Plot examples so you can see the classes with your own eyes

Run it, then paste me the printed output and describe the two plots.
=============================================================================
"""

import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import wfdb

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "data/mitdb"     # where the database gets downloaded
PLOT_DIR = "plots"          # where the figures get saved
FS = 360                    # MIT-BIH sampling rate: 360 samples per second

# The 48 standard MIT-BIH records.
ALL_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221', '222', '223',
    '228', '230', '231', '232', '233', '234',
]

# Records 102, 104, 107 and 217 contain PACED beats (from a pacemaker).
# Standard practice in every serious ECG paper is to exclude them: the
# pacemaker changes the beat shape entirely, and "paced" is not one of our
# classes. Excluding them is a convention you should follow, not a choice.
PACED_RECORDS = {'102', '104', '107', '217'}

# In the annotation files, each annotation has a SYMBOL. Some symbols mark a
# real heartbeat; others mark events (rhythm changes, noise, etc.). This is
# the set of symbols that mark an actual beat.
BEAT_SYMBOLS = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
    'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?',
}

# Our 5 target classes, and how we decide which one a beat belongs to.
#   NOR  = normal beat, during normal sinus rhythm
#   LBBB = beat symbol 'L'
#   RBBB = beat symbol 'R'
#   PVC  = beat symbol 'V'
#   AFIB = ANY beat that occurs during an atrial-fibrillation episode
#
# Read that last line again, because it is the whole AFib lesson:
# AFib is decided by the RHYTHM the beat sits inside, not by the beat's shape.
TARGET_CLASSES = ['NOR', 'LBBB', 'RBBB', 'PVC', 'AFIB']


# ---------------------------------------------------------------------------
# STEP 1 - DOWNLOAD
# ---------------------------------------------------------------------------
def download_database():
    """Download MIT-BIH from PhysioNet, but only the first time."""
    if os.path.isdir(DATA_DIR) and len(os.listdir(DATA_DIR)) > 100:
        print(f"Database already present in '{DATA_DIR}' - skipping download.\n")
        return

    print("Downloading MIT-BIH Arrhythmia Database (~100 MB).")
    print("This takes a few minutes on first run only...\n")
    os.makedirs(DATA_DIR, exist_ok=True)
    wfdb.dl_database('mitdb', DATA_DIR)
    print("Download complete.\n")


# ---------------------------------------------------------------------------
# STEP 2 - READ ONE RECORD AND LABEL EVERY BEAT
# ---------------------------------------------------------------------------
def label_beats_in_record(record_name):
    """
    Read one record's annotations and return a list of labelled beats.

    Each returned beat is a dict:
        sample : position of the R-peak, in samples
        symbol : the raw MIT-BIH beat symbol ('N', 'V', 'L', ...)
        rhythm : the rhythm the beat sits inside ('(N', '(AFIB', ...)
        cls    : our class name, or None if the beat is not one of our 5

    HOW THIS WORKS
    --------------
    The annotation file is a time-ordered list. Two kinds of entries matter:
      - beat annotations: symbol is a beat symbol, sample = R-peak location
      - rhythm annotations: symbol is '+', and aux_note says which rhythm
        STARTS at that point (e.g. '(AFIB') and continues until the next one.
    So we walk through in time order, remembering the current rhythm, and
    tag every beat with the rhythm that was active when it happened.
    """
    path = os.path.join(DATA_DIR, record_name)
    ann = wfdb.rdann(path, 'atr')

    beats = []
    current_rhythm = None

    for sample, symbol, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        # Rhythm-change marker: update what rhythm we are currently in.
        # The aux_note often has trailing null characters, so we clean it.
        aux_clean = aux.strip().strip('\x00').strip()
        if aux_clean.startswith('('):
            current_rhythm = aux_clean

        # Not a real beat -> nothing more to do with this entry.
        if symbol not in BEAT_SYMBOLS:
            continue

        # --- decide our class for this beat ---
        cls = None
        if current_rhythm == '(AFIB':
            # Inside an AFib episode. Note we take the beat regardless of its
            # shape symbol - that is exactly the point about AFib.
            cls = 'AFIB'
        elif symbol == 'L':
            cls = 'LBBB'
        elif symbol == 'R':
            cls = 'RBBB'
        elif symbol == 'V':
            cls = 'PVC'
        elif symbol == 'N' and current_rhythm in ('(N', None):
            # Normal beat during normal sinus rhythm only. We deliberately do
            # NOT take normal-looking beats from other abnormal rhythms
            # (bigeminy, flutter, etc.) - that would pollute the NOR class.
            cls = 'NOR'

        beats.append({
            'sample': int(sample),
            'symbol': symbol,
            'rhythm': current_rhythm,
            'cls': cls,
        })

    return beats


# ---------------------------------------------------------------------------
# STEP 3 - SCAN EVERY RECORD AND BUILD THE INVENTORY
# ---------------------------------------------------------------------------
def scan_all_records():
    """Go through every usable record and collect counts."""
    usable = [r for r in ALL_RECORDS if r not in PACED_RECORDS]

    class_counts = Counter()                    # class -> total beats
    class_patients = defaultdict(set)           # class -> set of record IDs
    per_record = {}                             # record -> Counter of classes
    all_symbols = Counter()                     # every raw symbol seen
    all_rhythms = Counter()                     # every rhythm type seen
    rr_by_class = defaultdict(list)             # class -> list of RR intervals

    print(f"Scanning {len(usable)} records "
          f"({len(PACED_RECORDS)} paced records excluded)...\n")

    for rec in usable:
        beats = label_beats_in_record(rec)
        rec_counts = Counter()

        for i, b in enumerate(beats):
            all_symbols[b['symbol']] += 1
            if b['rhythm']:
                all_rhythms[b['rhythm']] += 1

            if b['cls'] is None:
                continue

            class_counts[b['cls']] += 1
            class_patients[b['cls']].add(rec)
            rec_counts[b['cls']] += 1

            # RR interval = time from the previous beat to this one, in seconds.
            # This is THE feature that exposes AFib.
            if i > 0:
                rr = (b['sample'] - beats[i - 1]['sample']) / FS
                if 0.2 < rr < 3.0:          # ignore impossible values
                    rr_by_class[b['cls']].append(rr)

        per_record[rec] = rec_counts

    return class_counts, class_patients, per_record, all_symbols, all_rhythms, rr_by_class


# ---------------------------------------------------------------------------
# STEP 4 - PRINT THE REPORT
# ---------------------------------------------------------------------------
def print_report(class_counts, class_patients, per_record,
                 all_symbols, all_rhythms, rr_by_class):

    total = sum(class_counts[c] for c in TARGET_CLASSES)

    print("=" * 70)
    print("CLASS INVENTORY  (the answer to 'do we have enough data?')")
    print("=" * 70)
    print(f"{'Class':<8}{'Beats':>10}{'% of set':>11}{'Patients':>11}   Records")
    print("-" * 70)
    for c in TARGET_CLASSES:
        n = class_counts[c]
        pats = sorted(class_patients[c])
        pct = 100 * n / total if total else 0
        shown = ','.join(pats[:8]) + (' ...' if len(pats) > 8 else '')
        print(f"{c:<8}{n:>10,}{pct:>10.1f}%{len(pats):>11}   {shown}")
    print("-" * 70)
    print(f"{'TOTAL':<8}{total:>10,}")

    # Imbalance ratio: how many times more common is the biggest class than
    # the smallest? This number decides how hard we must fight imbalance.
    counts = [class_counts[c] for c in TARGET_CLASSES if class_counts[c] > 0]
    if counts:
        print(f"\nImbalance ratio (largest / smallest class): "
              f"{max(counts) / min(counts):.1f}x")

    print("\n" + "=" * 70)
    print("PATIENT DIVERSITY PER CLASS")
    print("=" * 70)
    print("Beat count is the easy number. Patient count is the honest one:")
    print("a class held by very few patients cannot be split into train and")
    print("test without either starving training or having no real test.\n")
    for c in TARGET_CLASSES:
        pats = sorted(class_patients[c])
        flag = "  <-- THIN" if len(pats) <= 3 else ""
        print(f"  {c:<6} {len(pats):>3} patients{flag}")
        if len(pats) <= 10:
            print(f"         {', '.join(pats)}")

    print("\n" + "=" * 70)
    print("RR-INTERVAL STATISTICS  (why AFib needs timing, not shape)")
    print("=" * 70)
    print("RR interval = seconds between one beat and the next.")
    print("Watch the STD DEV column: AFib should be visibly more erratic.\n")
    print(f"{'Class':<8}{'mean RR':>10}{'std dev':>10}{'min':>8}{'max':>8}")
    print("-" * 46)
    for c in TARGET_CLASSES:
        rr = np.array(rr_by_class[c])
        if len(rr) == 0:
            continue
        print(f"{c:<8}{rr.mean():>10.3f}{rr.std():>10.3f}"
              f"{rr.min():>8.3f}{rr.max():>8.3f}")

    print("\n" + "=" * 70)
    print("ALL RHYTHM TYPES FOUND IN THE DATABASE")
    print("=" * 70)
    for r, n in all_rhythms.most_common():
        print(f"  {r:<10} {n:>8,} beats")

    print("\n" + "=" * 70)
    print("ALL BEAT SYMBOLS FOUND (including ones we are not using)")
    print("=" * 70)
    for s, n in all_symbols.most_common():
        print(f"  '{s}'  {n:>8,}")


# ---------------------------------------------------------------------------
# STEP 5 - PLOTS
# ---------------------------------------------------------------------------
def find_example(per_record, cls):
    """Pick the record with the most beats of this class."""
    best, best_n = None, 0
    for rec, counts in per_record.items():
        if counts[cls] > best_n:
            best, best_n = rec, counts[cls]
    return best


def plot_beat_shapes(per_record):
    """
    FIGURE 1 - one beat per class, ~1.4 seconds around the R-peak.
    This is the MORPHOLOGY view: what the model's CNN part will look at.
    You should be able to see LBBB/RBBB/PVC look wrong. AFib will look
    fairly normal - which is exactly the lesson.
    """
    fig, axes = plt.subplots(len(TARGET_CLASSES), 1,
                             figsize=(9, 11), sharex=True)

    half = int(0.7 * FS)   # 0.7 s on each side of the R-peak

    for ax, cls in zip(axes, TARGET_CLASSES):
        rec = find_example(per_record, cls)
        if rec is None:
            ax.set_title(f"{cls}: no examples found")
            continue

        record = wfdb.rdrecord(os.path.join(DATA_DIR, rec))
        sig = record.p_signal[:, 0]          # lead 0 = MLII on most records
        beats = [b for b in label_beats_in_record(rec) if b['cls'] == cls]

        # take a beat from the middle of the record (start/end can be messy)
        b = beats[len(beats) // 2]
        s = b['sample']
        if s - half < 0 or s + half >= len(sig):
            b = beats[len(beats) // 2 + 5]
            s = b['sample']

        seg = sig[s - half:s + half]
        t = np.arange(-half, half) / FS

        ax.plot(t, seg, linewidth=1.2)
        ax.axvline(0, linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel("mV")
        ax.set_title(f"{cls}   (record {rec}, symbol '{b['symbol']}', "
                     f"rhythm {b['rhythm']})", loc='left', fontsize=10)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("seconds from R-peak")
    fig.suptitle("Figure 1 - Beat morphology by class", fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "fig1_beat_shapes.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


def plot_rhythm_strips(per_record):
    """
    FIGURE 2 - 10-second strips: normal sinus rhythm vs AFib, R-peaks marked.
    This is the RHYTHM view. Look at the SPACING between the red dots.
    Normal = evenly spaced. AFib = visibly irregular.
    This picture is the entire justification for feeding RR intervals
    into the model alongside the waveform.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    window = 10 * FS

    for ax, cls in zip(axes, ['NOR', 'AFIB']):
        rec = find_example(per_record, cls)
        if rec is None:
            ax.set_title(f"{cls}: no examples found")
            continue

        record = wfdb.rdrecord(os.path.join(DATA_DIR, rec))
        sig = record.p_signal[:, 0]
        beats = [b for b in label_beats_in_record(rec) if b['cls'] == cls]

        start = beats[len(beats) // 2]['sample']
        end = start + window
        seg = sig[start:end]
        t = np.arange(len(seg)) / FS

        peaks = [(b['sample'] - start) / FS for b in beats
                 if start <= b['sample'] < end]

        ax.plot(t, seg, linewidth=1.0)
        for p in peaks:
            ax.plot(p, sig[start + int(p * FS)], 'ro', markersize=4)

        rrs = np.diff([p for p in peaks])
        label = (f"{cls}  (record {rec})   "
                 f"RR std dev in this strip: {rrs.std():.3f} s"
                 if len(rrs) > 1 else f"{cls}  (record {rec})")
        ax.set_title(label, loc='left', fontsize=10)
        ax.set_ylabel("mV")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("seconds")
    fig.suptitle("Figure 2 - Rhythm view: look at the SPACING of the red dots",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "fig2_rhythm_strips.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    download_database()
    results = scan_all_records()
    print_report(*results)

    per_record = results[2]
    plot_beat_shapes(per_record)
    plot_rhythm_strips(per_record)

    print("\nDone. Both figures are in the 'plots' folder.")
    plt.show()


if __name__ == "__main__":
    main()