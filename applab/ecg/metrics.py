"""
Evaluation helpers.

Overall accuracy is deliberately never reported on its own: with a ~10x
class imbalance, a model that answered "NOR" to everything would score 65%
and be completely useless. Everything here is per-class.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score)

from . import config as C


def report(y_true, y_pred):
    """Return a dict of per-class and aggregate metrics."""
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(C.N_CLASSES), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(C.N_CLASSES))
    return {
        "per_class": {
            C.CLASSES[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(sup[i]),
            } for i in range(C.N_CLASSES)
        },
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=sup)) if sup.sum() else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }


def print_report(rep, title=""):
    if title:
        print(f"\n{title}")
    print("-" * 62)
    print(f"{'class':<8}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>11}")
    print("-" * 62)
    for c, m in rep["per_class"].items():
        print(f"{c:<8}{m['precision']:>11.3f}{m['recall']:>9.3f}"
              f"{m['f1']:>9.3f}{m['support']:>11,}")
    print("-" * 62)
    print(f"{'macro F1':<8}{rep['macro_f1']:>11.3f}"
          f"      (accuracy {rep['accuracy']:.3f} - shown for completeness "
          f"only)")

    cm = np.array(rep["confusion_matrix"])
    print("\nconfusion matrix   rows = true, cols = predicted")
    print(f"{'':<8}" + "".join(f"{c:>9}" for c in C.CLASSES))
    for i, c in enumerate(C.CLASSES):
        print(f"{c:<8}" + "".join(f"{v:>9,}" for v in cm[i]))


def plot_confusion(rep, path, title):
    """Row-normalised confusion matrix heatmap, counts annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(rep["confusion_matrix"], dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(C.N_CLASSES), C.CLASSES)
    ax.set_yticks(range(C.N_CLASSES), C.CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\nmacro F1 = {rep['macro_f1']:.3f}", fontsize=11)

    for i in range(C.N_CLASSES):
        for j in range(C.N_CLASSES):
            ax.text(j, i, f"{norm[i, j]:.2f}\n{int(cm[i, j]):,}",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
