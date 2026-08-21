"""
Export every approved demo record to applab/data/ so the on-device dashboard
can replay any of them from a dropdown, with no MIT-BIH library or raw
database on the board.

Every record here is test fold only. This script just reads the vetted lists in
config.DEMO_HEALTHY / DEMO_ARRHYTHMIA; the DEMO_ONLY_TEST_RECORDS guard in
step1b_check_split.py is what actually enforces the rule.

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

# Where to start the 90 s excerpt, for records whose condition isn't there at
# the beginning. Read off each record's own rhythm annotations rather than
# guessed. Anything not listed here starts at 0.
#
#   202  the first 1142 s are annotated "(N", so exporting from 0 gives a demo
#        labelled AFIB with no AFib in it. Of its three AFib runs, the start
#        point matters a lot for PVC confusion:
#            start 1180  101/180 AFib correct,  61 called PVC
#            start 1260  100/142 AFib correct,  25 called PVC
#            start 1380  125/139 AFib correct,   1 called PVC
#            start 1620  119/133 AFib correct,   1 called PVC
#        1180 is the onset of the first run, where RR irregularity is still
#        settling and the model reads the transitional short-long pattern as
#        ectopy. That's a real weakness, not a replay artefact, it reproduces
#        at 360 Hz too. 1380 is a settled stretch and is what we export.
#   221  annotated "(AFIB" from 0.1 s, so the default start is fine.
START_AT = {
    "202": 1380,
}

# Note override for records where the 90 s excerpt doesn't match the
# whole-record note in config. 202's note is true of the record but not of the
# settled run we export, and the dropdown should describe what's on screen.
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
