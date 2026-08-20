"""
=============================================================================
PHASE 1c - LEAD COMPARISON AND ANNOTATION INSPECTION
=============================================================================
Answers three questions with data instead of explanation:

  1. Why does our LBBB not look like the textbook picture?
     -> Figure 7 shows the SAME beats in BOTH recorded leads.
        A condition can look completely different depending on which
        electrode view you use.

  2. Which lead is actually in channel 0 of each record?
     -> Lead audit table. Most records use MLII in channel 0, but not all,
        and blindly taking channel 0 in Phase 2 would silently mix leads.

  3. What is that unlabelled beat in record 215 around 3.4 seconds?
     -> Annotation dump prints every beat in a chosen time window with its
        symbol, class and RR interval.
=============================================================================
"""

import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import wfdb

DATA_DIR = "data/mitdb"
PLOT_DIR = "plots"
FS = 360
BEFORE = int(0.30 * FS)
AFTER = int(0.50 * FS)

BEAT_SYMBOLS = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
    'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?',
}

ALL_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221', '222', '223',
    '228', '230', '231', '232', '233', '234',
]

# Human-readable names for the beat symbols, so the dump is readable.
SYMBOL_NAMES = {
    'N': 'Normal beat',
    'L': 'Left bundle branch block beat',
    'R': 'Right bundle branch block beat',
    'V': 'Premature ventricular contraction (PVC)',
    'A': 'Atrial premature beat',
    'a': 'Aberrated atrial premature beat',
    'J': 'Nodal (junctional) premature beat',
    'S': 'Supraventricular premature beat',
    'F': 'Fusion of ventricular and normal beat',
    'e': 'Atrial escape beat',
    'j': 'Nodal (junctional) escape beat',
    'E': 'Ventricular escape beat',
    'Q': 'Unclassifiable beat',
    '/': 'Paced beat',
    'f': 'Fusion of paced and normal beat',
}


def label_beats_in_record(record_name):
    ann = wfdb.rdann(os.path.join(DATA_DIR, record_name), 'atr')
    beats, current_rhythm = [], None
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


# ---------------------------------------------------------------------------
# 1. LEAD AUDIT
# ---------------------------------------------------------------------------
def lead_audit():
    """
    Print which physical lead sits in each channel of each record.

    WHY THIS MATTERS FOR PHASE 2:
    if some records put MLII in channel 0 and others put a chest lead there,
    then taking channel 0 everywhere silently mixes two different views of
    the heart into one training set. The model then has to learn two
    different-looking versions of every class for no reason. The fix is to
    select the channel BY NAME, not by index.
    """
    print("=" * 72)
    print("LEAD AUDIT - which signal is in which channel")
    print("=" * 72)

    layouts = Counter()
    exceptions = []

    for rec in ALL_RECORDS:
        header = wfdb.rdheader(os.path.join(DATA_DIR, rec))
        names = tuple(header.sig_name)
        layouts[names] += 1
        if names[0] != 'MLII':
            exceptions.append((rec, names))

    print("\nChannel layouts found across all 48 records:")
    for names, n in layouts.most_common():
        print(f"  {str(names):<28} {n:>3} records")

    if exceptions:
        print("\nRecords where channel 0 is NOT MLII:")
        for rec, names in exceptions:
            print(f"  record {rec}:  channel0={names[0]}, channel1={names[1]}")
        print("\n  ^ In Phase 2 we will select the channel by NAME so these")
        print("    records cannot silently contaminate the dataset.")
    else:
        print("\nEvery record has MLII in channel 0.")


# ---------------------------------------------------------------------------
# 2. ANNOTATION DUMP
# ---------------------------------------------------------------------------
def inspect_window(record_name, t_start, t_end):
    """
    Print every beat between t_start and t_end seconds, with its symbol,
    our class label, and the gap since the previous beat.

    Use this any time a plot shows something you do not understand -
    the annotation file always has the answer.
    """
    print("\n" + "=" * 72)
    print(f"ANNOTATION DUMP - record {record_name}, "
          f"{t_start:.1f}s to {t_end:.1f}s")
    print("=" * 72)

    beats = label_beats_in_record(record_name)
    print(f"{'time (s)':>9}{'symbol':>9}{'class':>8}{'RR (s)':>9}   meaning")
    print("-" * 72)

    for i, b in enumerate(beats):
        t = b['sample'] / FS
        if not (t_start <= t <= t_end):
            continue
        rr = (b['sample'] - beats[i - 1]['sample']) / FS if i > 0 else np.nan
        cls = b['cls'] or '-'
        flag = ""
        if not np.isnan(rr) and rr < 0.5:
            flag = "   <-- EARLY"
        elif not np.isnan(rr) and rr > 1.0:
            flag = "   <-- PAUSE"
        print(f"{t:>9.3f}{b['symbol']:>9}{cls:>8}{rr:>9.3f}   "
              f"{SYMBOL_NAMES.get(b['symbol'], '?')}{flag}")


# ---------------------------------------------------------------------------
# 3. FIGURE 7 - same condition seen through two different leads
# ---------------------------------------------------------------------------
def mean_beat(record_name, cls, channel, max_beats=200):
    record = wfdb.rdrecord(os.path.join(DATA_DIR, record_name))
    sig = record.p_signal[:, channel]
    beats = [b for b in label_beats_in_record(record_name) if b['cls'] == cls]

    windows = []
    for b in beats[:max_beats * 2]:
        s = b['sample']
        if s - BEFORE < 0 or s + AFTER >= len(sig):
            continue
        windows.append(sig[s - BEFORE:s + AFTER])
        if len(windows) >= max_beats:
            break

    if not windows:
        return None, record.sig_name[channel]
    return np.array(windows).mean(axis=0), record.sig_name[channel]


def plot_lead_comparison():
    """
    Three conditions, two leads each, raw millivolts (NOT normalised) so you
    can see real amplitude and real width.

    Read it ROW BY ROW to compare conditions in the same lead,
    and COLUMN BY COLUMN to see how much the lead choice changes appearance.
    """
    rows = [
        ('100', 'NOR', 'NORMAL sinus rhythm'),
        ('109', 'LBBB', 'LEFT bundle branch block'),
        ('118', 'RBBB', 'RIGHT bundle branch block'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    t = np.arange(-BEFORE, AFTER) / FS

    for r, (rec, cls, title) in enumerate(rows):
        for c in (0, 1):
            ax = axes[r, c]
            m, lead_name = mean_beat(rec, cls, c)
            if m is None:
                ax.set_title(f"{title}: no beats")
                continue

            colour = 'tab:blue' if cls == 'NOR' else 'tab:red'
            ax.plot(t, m, color=colour, linewidth=2.0)
            ax.axvline(0, color='k', linewidth=0.7, alpha=0.4)
            ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
            ax.set_title(f"{title}\nrecord {rec}, lead {lead_name}",
                         loc='left', fontsize=10)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("mV")

    for ax in axes[2]:
        ax.set_xlabel("seconds from R-peak")

    fig.suptitle("Figure 7 - The SAME condition through TWO different leads\n"
                 "Left column = limb lead view.  Right column = chest lead view.\n"
                 "This is why the textbook picture and our plot disagree.",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "fig7_lead_comparison.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


# ---------------------------------------------------------------------------
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    lead_audit()

    # The mystery beat you spotted in Figure 4, top panel.
    inspect_window('215', 2.5, 5.0)

    plot_lead_comparison()

    print("\nDone.")
    plt.show()


if __name__ == "__main__":
    main()
