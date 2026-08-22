"""
STEP 1b - Validate the patient-level split.

Prints the beat count of every class in every fold and hard-fails if a class
is missing from a fold or if any patient leaks across folds. Run this after
any change to TEST_RECORDS / VAL_RECORDS in ecg/config.py.

    python step1b_check_split.py
"""
import sys

import numpy as np

from ecg import config as C
from ecg import data as D


def main():
    d = D.load_dataset()
    y, recs = d["y"], d["record"]

    folds = {
        "train": C.TRAIN_RECORDS,
        "val": C.VAL_RECORDS,
        "test": C.TEST_RECORDS,
    }

    print("=" * 74)
    print("STEP 1b  -  PATIENT-LEVEL SPLIT CHECK")
    print("=" * 74)

    # ------------------------------------------------------------- leak check
    problems = []
    seen = {}
    for name, records in folds.items():
        for r in records:
            if r in seen:
                problems.append(f"record {r} is in both {seen[r]} and {name}")
            seen[r] = name

    missing = set(C.USABLE_RECORDS) - set(seen)
    if missing:
        problems.append(f"records assigned to no fold: {sorted(missing)}")

    # ------------------------------------------------------------ count table
    header = f"{'fold':<7}{'records':>9}{'beats':>10}"
    for c in C.CLASSES:
        header += f"{c:>9}"
    print(header)
    print("-" * 74)

    for name, records in folds.items():
        m = D.mask_for(recs, records)
        counts = np.bincount(y[m], minlength=C.N_CLASSES)
        row = f"{name:<7}{len(records):>9}{int(m.sum()):>10,}"
        for i in range(C.N_CLASSES):
            row += f"{counts[i]:>9,}"
        print(row)

        for i, c in enumerate(C.CLASSES):
            if counts[i] == 0:
                problems.append(f"class {c} has ZERO beats in {name}")

    print("-" * 74)

    # ------------------------------------------------- patients per class/fold
    print(f"\n{'class':<7}{'train pts':>11}{'val pts':>9}{'test pts':>10}"
          f"   test patient ids")
    print("-" * 74)
    for i, c in enumerate(C.CLASSES):
        cells = []
        for name, records in folds.items():
            m = D.mask_for(recs, records) & (y == i)
            cells.append(sorted(set(recs[m].tolist())))
        tr, va, te = cells
        print(f"{c:<7}{len(tr):>11}{len(va):>9}{len(te):>10}   "
              f"{', '.join(str(x) for x in te)}")
        if len(te) == 1:
            print(f"        NOTE: test {c} comes from ONE patient - the score "
                  f"for this class is a single-patient estimate.")

    # ------------------------------------------------- demo record 
    print("\n\nDEMO RECORDS OFFERED BY THE GUI")
    print("-" * 74)
    print(f"{'record':>8}{'role':>12}{'fold':>8}   status")
    print("-" * 74)
    train_set = set(C.TRAIN_RECORDS)
    demo = ([(r, 'healthy', fold) for r, fold, _ in C.DEMO_HEALTHY]
            + [(r, cls, fold) for r, cls, fold, _ in C.DEMO_ARRHYTHMIA])
    for r, role, fold in demo:
        if r in train_set:
            status = "*** TRAINED ON ***"
            problems.append(f"demo record {r} ({role}) is in TRAIN_RECORDS")
        elif fold == 'test':
            status = "never seen by the model at all"
        elif C.DEMO_ONLY_TEST_RECORDS:
            status = "*** VALIDATION RECORD - excluded by policy ***"
            problems.append(
                f"demo record {r} ({role}) is a validation record, but "
                f"DEMO_ONLY_TEST_RECORDS is True")
        else:
            status = "not trained on; used only to pick the checkpoint"
        print(f"{r:>8}{role:>12}{fold:>8}   {status}")
    print("-" * 74)
    n_test = sum(1 for _, _, f in demo if f == 'test')
    if n_test == len(demo):
        print(f"  All {len(demo)} demo recordings come from patients the "
              f"model has never seen in any capacity.")
    else:
        print(f"  {n_test} of {len(demo)} demo recordings are fully held out; "
              f"{len(demo) - n_test} are validation records.")

    # ---------------------------------------------------------------- verdict
    print()
    if problems:
        print("SPLIT IS INVALID:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("Split is valid: no patient leakage, every class present in every "
          "fold.")


if __name__ == "__main__":
    main()
