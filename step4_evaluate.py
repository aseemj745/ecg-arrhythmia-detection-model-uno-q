"""
STEP 4 - Full evaluation of the deployed INT8 model.

    python step4_evaluate.py

Reports, on the held-out patients only:
  1. per-class precision / recall / F1 and the confusion matrix
  2. per-record accuracy, since that's what a per-patient demo shows
  3. normal-vs-abnormal screening performance, which is what drives the GUI's
     flagging behaviour
  4. single-beat inference latency, for the real-time claim
  5. the CNN-only vs CNN+RR ablation, if that run exists
"""
from __future__ import annotations

import json
import time

import numpy as np
import onnxruntime as ort

from ecg import config as C
from ecg import data as D
from ecg import metrics as M


def predict(path, X, F, batch=512):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(X), batch):
        out.append(sess.run(["logits"], {
            "beat": X[i:i + batch][:, None, :].astype(np.float32),
            "rr_features": F[i:i + batch].astype(np.float32)})[0])
    return np.concatenate(out)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def benchmark(path, n=300):
    """Single-beat latency - the live path processes one beat at a time."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.zeros((1, 1, C.WINDOW), dtype=np.float32)
    f = np.zeros((1, C.N_FEATURES), dtype=np.float32)
    for _ in range(30):
        sess.run(["logits"], {"beat": x, "rr_features": f})
    times = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(["logits"], {"beat": x, "rr_features": f})
        times.append((time.perf_counter() - t) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    print("=" * 74)
    print("STEP 4  -  EVALUATION  (INT8 ONNX, held-out patients)")
    print("=" * 74)

    d = D.load_dataset()
    te = D.mask_for(d["record"], C.TEST_RECORDS)
    X, F, y, rec = d["X"][te], d["F"][te], d["y"][te], d["record"][te]

    int8 = C.MODEL_DIR / "model_int8.onnx"
    logits = predict(int8, X, F)
    prob = softmax(logits)
    pred = logits.argmax(1)

    rep = M.report(y, pred)
    M.print_report(rep, "1. PER-CLASS  (5-way, INT8)")

    # ------------------------------------------------------- 2. per record
    print("\n\n2. PER-RECORD  (each row is one unseen patient)")
    print("-" * 74)
    print(f"{'record':>8}{'beats':>9}{'accuracy':>11}   dominant true class"
          f"   most common error")
    print("-" * 74)
    per_record = {}
    for r in sorted(set(rec.tolist())):
        m = rec == r
        acc = float((pred[m] == y[m]).mean())
        true_major = C.CLASSES[np.bincount(y[m],
                                           minlength=C.N_CLASSES).argmax()]
        wrong = pred[m][pred[m] != y[m]]
        err = (C.CLASSES[np.bincount(wrong, minlength=C.N_CLASSES).argmax()]
               if len(wrong) else "-")
        print(f"{r:>8}{int(m.sum()):>9,}{acc:>11.3f}   {true_major:<20}"
              f"   {err}")
        per_record[int(r)] = {"beats": int(m.sum()), "accuracy": acc,
                              "dominant_class": true_major}
    print("-" * 74)

    # -------------------------------------------- 3. normal vs abnormal
    # This is what the GUI screening behaviour rests on: the demo marks a
    # window as "something is wrong here", it does not have to name the
    # arrhythmia correctly to be useful. A PVC called LBBB is still a
    # correctly flagged abnormal beat.
    nor = C.CLASS_TO_IDX["NOR"]
    yb = (y != nor).astype(int)
    pb = (pred != nor).astype(int)
    tp = int(((yb == 1) & (pb == 1)).sum())
    tn = int(((yb == 0) & (pb == 0)).sum())
    fp = int(((yb == 0) & (pb == 1)).sum())
    fn = int(((yb == 1) & (pb == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)

    print("\n\n3. NORMAL vs ABNORMAL  (the screening question the GUI asks)")
    print("-" * 74)
    print(f"  sensitivity (abnormal beats caught) : {sens:.4f}   "
          f"{tp:,} of {tp + fn:,}")
    print(f"  specificity (normal beats left alone): {spec:.4f}   "
          f"{tn:,} of {tn + fp:,}")
    print(f"  precision   (flags that were right)  : {ppv:.4f}")
    print(f"  balanced accuracy                    : {(sens + spec) / 2:.4f}")
    print("-" * 74)
    print("  Note: this is a LOWER bar than the 5-way task above and is")
    print("  reported separately, not as a substitute for it.")

    # -------------------------------------------------- 4. latency
    mean_ms, p95_ms = benchmark(int8)
    print("\n\n4. INFERENCE LATENCY  (single beat, INT8, CPU)")
    print("-" * 74)
    print(f"  mean {mean_ms:.3f} ms   p95 {p95_ms:.3f} ms   "
          f"on this dev PC's CPU")
    print(f"  A beat arrives at most ~3x/second, so the budget is ~300 ms.")
    print(f"  The UNO Q's Cortex-A53 is far slower than this CPU, but the")
    print(f"  headroom here is {300 / max(mean_ms, 1e-6):.0f}x, so real-time")
    print(f"  on-device inference is not in doubt. Measure on the actual")
    print(f"  board before quoting a number in the submission.")

    # -------------------------------------------------- 5. ablation
    print("\n\n5. ABLATION  -  what do the RR features actually buy?")
    print("-" * 74)
    rows = []
    for tag, label in [("model", "CNN + RR fusion"),
                       ("model_cnn_only", "CNN only (no RR)")]:
        p = C.REPORT_DIR / f"{tag}_training.json"
        if not p.exists():
            print(f"  [{label}] not run yet - "
                  f"`python step2_train.py --no-rr --tag model_cnn_only`")
            continue
        with open(p) as fh:
            j = json.load(fh)
        rows.append((label, j["test"]))

    if len(rows) == 2:
        print(f"{'variant':<20}{'macro F1':>10}" +
              "".join(f"{c:>9}" for c in C.CLASSES))
        print("-" * 74)
        for label, r in rows:
            print(f"{label:<20}{r['macro_f1']:>10.3f}" +
                  "".join(f"{r['per_class'][c]['f1']:>9.3f}"
                          for c in C.CLASSES))
        print("-" * 74)
        fused, cnn = rows[0][1], rows[1][1]
        da = (fused["per_class"]["AFIB"]["f1"] -
              cnn["per_class"]["AFIB"]["f1"])
        dm = fused["macro_f1"] - cnn["macro_f1"]
        print(f"  RR features change macro F1 by {dm:+.3f} and AFIB F1 by "
              f"{da:+.3f}.")
        print("  AFIB is the row to read: it is a timing diagnosis, so it is")
        print("  where the fusion has to earn its place.")

    out = {"per_class_int8": rep, "per_record": per_record,
           "binary_screening": {"sensitivity": sens, "specificity": spec,
                                "precision": ppv, "tp": tp, "tn": tn,
                                "fp": fp, "fn": fn},
           "latency_ms": {"mean": mean_ms, "p95": p95_ms,
                          "note": "dev PC CPU, not the UNO Q"}}
    with open(C.REPORT_DIR / "evaluation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved  {C.REPORT_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
