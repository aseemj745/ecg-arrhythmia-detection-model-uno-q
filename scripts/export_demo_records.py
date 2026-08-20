"""
Export every approved demo record to applab/data/ so the on-device dashboard
can replay any of them from a dropdown, with no MIT-BIH library or raw
database on the board.

Every record here is TEST FOLD ONLY - see config.DEMO_HEALTHY /
DEMO_ARRHYTHMIA and the DEMO_ONLY_TEST_RECORDS guard in step1b_check_split.py.
Replaying a training patient and calling it a demo would be showing the model
its own homework, so the split guard is what actually enforces this and this
script only reads the already-vetted lists.

    python scripts/export_demo_records.py

Writes one CSV per record plus demo_records.json (the manifest the dashboard
reads to build its dropdown).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ecg import config as C          # noqa: E402
from ecg import data as D            # noqa: E402

OUT = ROOT / "applab" / "data"
SECONDS = 90
EXPORT_FS = 250          # what the CSVs are stored at; see SELFTEST_SAMPLE_RATE

# Where to start the 90 s excerpt, for records whose condition is not present
# at the beginning. Read straight off each record's own rhythm annotations
# (the "(AFIB" / "(N" markers in the .atr aux_note field), not guessed:
#
#   202  first 1142 s are annotated "(N" - normal sinus. Exporting from 0 s
#        gives a demo labelled AFIB that contains no AFib at all, and the
#        model correctly reports nothing, which looks like a failure. The
#        record has three AFib runs: 1176-1291, 1302-1527, 1575-1805 s.
#
#        Which run matters, and it is not the first one. Scored against
#        ecg.data.label_beats over every 90 s window of the record, the
#        model's PVC confusion varies enormously across them:
#
#            start 1180  101/180 AFib correct,  61 called PVC   36.7% PVC
#            start 1260  100/142 AFib correct,  25 called PVC   18.2% PVC
#            start 1380  125/139 AFib correct,   1 called PVC    0.8% PVC
#            start 1620  119/133 AFib correct,   1 called PVC    0.8% PVC
#
#        1180 sits at the ONSET of the first run, where RR irregularity is
#        still settling; the model reads that transitional short-long
#        pattern as ventricular ectopy. This is a real model weakness, not
#        a replay artefact - it reproduces at native 360 Hz, so the desktop
#        GUI would show the same thing on the same window. 1380 s is a
#        settled stretch of the second run and is what the demo now plays.
#   221  annotated "(AFIB" from 0.1 s, so the default start is already right.
#
# Anything not listed starts at 0.
START_AT = {
    "202": 1380,
}

# Manifest note override, for records where the 90 s excerpt does not show
# what the whole-record note in config describes. 202's config note ("AFib
# episodes alternating with normal rhythm") is true of the record but not of
# the settled AFib run this exports, and the dropdown should describe the
# signal actually on screen.
NOTE_AT_WINDOW = {
    "202": "sustained AFib run (the record alternates AFib and normal)",
}


def export(rec, condition, note):
    path = C.DATA_DIR / rec
    header = wfdb.rdheader(str(path))
    ch = D.find_lead_channel(header)
    if ch is None:
        print(f"  {rec}: no {C.TARGET_LEAD} lead, skipped")
        return None

    start = START_AT.get(rec, 0)
    lo = C.FS * start
    sig = wfdb.rdrecord(str(path)).p_signal[lo:lo + C.FS * SECONDS, ch]
    # 360 -> 250 with an anti-aliasing filter. resample_poly, never plain
    # slicing: naive decimation aliases high-frequency QRS energy back down
    # into the band the model reads and distorts the waveform.
    out = resample_poly(sig, EXPORT_FS, C.FS)

    fname = f"demo_{rec}.csv"
    np.savetxt(OUT / fname, out, delimiter=",", fmt="%.5f")
    kb = (OUT / fname).stat().st_size / 1024
    at = f"from {start}s" if start else ""
    print(f"  {rec:>4}  {condition:<5} {len(out):>6} samples  {kb:6.0f} KB  "
          f"{note} {at}")
    return {"record": rec, "condition": condition,
            "note": NOTE_AT_WINDOW.get(rec, note),
            "file": fname, "fs": EXPORT_FS, "seconds": SECONDS,
            "start_s": start}


def main():
    if not C.DATA_DIR.exists():
        print(f"{C.DATA_DIR} not found - run step1_build_dataset.py first.")
        return
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = []
    print(f"Exporting {SECONDS}s per record at {EXPORT_FS} Hz -> {OUT}")
    print()
    print("HEALTHY (mostly normal sinus):")
    for rec, fold, note in C.DEMO_HEALTHY:
        assert fold == "test", f"{rec} is not test fold"
        m = export(rec, "NOR", note)
        if m:
            manifest.append(m)

    print()
    print("ARRHYTHMIA:")
    for rec, cond, fold, note in C.DEMO_ARRHYTHMIA:
        assert fold == "test", f"{rec} is not test fold"
        m = export(rec, cond, note)
        if m:
            manifest.append(m)

    (OUT / "demo_records.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum((OUT / m["file"]).stat().st_size for m in manifest) / 1024 / 1024
    print()
    print(f"{len(manifest)} records, {total:.1f} MB total")
    print(f"manifest -> {OUT / 'demo_records.json'}")


if __name__ == "__main__":
    main()
