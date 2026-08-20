"""
STEP 1 - Build the labelled beat dataset from MIT-BIH.

Reads every usable record, extracts R-peak-centred MLII windows plus causal
rhythm features, and writes one npz to artifacts/dataset/beats.npz along with
a per-record class-count table used to design the patient-level split.

    python step1_build_dataset.py
"""
import json
import time

import numpy as np

from ecg import config as C
from ecg import data as D


def main():
    t0 = time.time()
    print("=" * 74)
    print("STEP 1  -  BUILD BEAT DATASET")
    print("=" * 74)
    print(f"Source            : {C.DATA_DIR}")
    print(f"Lead              : {C.TARGET_LEAD} (selected by name, not index)")
    print(f"Window            : -{C.BEFORE} / +{C.AFTER} samples "
          f"= {C.WINDOW} @ {C.FS} Hz")
    print(f"Bandpass          : {C.BANDPASS} Hz")
    print(f"Excluded (paced)  : {', '.join(C.EXCLUDED_RECORDS)}")
    print(f"Records to read   : {len(C.USABLE_RECORDS)}")
    print()

    parts = []
    per_record = {}

    for rec in C.USABLE_RECORDS:
        out = D.extract_record(rec, verbose=True)
        if out is None:
            continue
        parts.append(out)
        per_record[rec] = {c: int((out["y"] == i).sum())
                           for i, c in enumerate(C.CLASSES)}

    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    out_path = C.DATASET_DIR / "beats.npz"
    np.savez_compressed(out_path, **merged)

    with open(C.DATASET_DIR / "per_record_counts.json", "w") as f:
        json.dump(per_record, f, indent=2)

    # ---------------------------------------------------------------- summary
    y = merged["y"]
    recs = merged["record"]
    print()
    print("-" * 74)
    print(f"{'class':<8}{'beats':>10}{'patients':>11}{'share':>9}")
    print("-" * 74)
    for i, c in enumerate(C.CLASSES):
        n = int((y == i).sum())
        pats = sorted(set(recs[y == i].tolist()))
        print(f"{c:<8}{n:>10,}{len(pats):>11}{100 * n / len(y):>8.1f}%")
        print(f"        patients: {', '.join(str(p) for p in pats)}")
    print("-" * 74)
    counts = np.bincount(y, minlength=C.N_CLASSES)
    print(f"TOTAL   {len(y):>10,} beats from {len(set(recs.tolist()))} records")
    print(f"Imbalance ratio (largest/smallest class): "
          f"{counts.max() / counts.min():.1f}x")
    print()
    print(f"Saved  {out_path}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved  {C.DATASET_DIR / 'per_record_counts.json'}")
    print(f"Elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
