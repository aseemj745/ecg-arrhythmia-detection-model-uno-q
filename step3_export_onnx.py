"""
STEP 3 - Export to ONNX and quantise to INT8.

    python step3_export_onnx.py                 # exports artifacts/models/model_best.pt
    python step3_export_onnx.py --tag model_cnn_only

Produces, in artifacts/models/:
    <tag>_fp32.onnx     portable float model
    <tag>_int8.onnx     statically quantised, this is what ships to the UNO Q
and verifies numerically that the ONNX graphs agree with PyTorch before
declaring success.

Static (calibrated) quantisation is used rather than dynamic: dynamic
quantisation leaves Conv layers in float, and Conv is essentially the whole
model here, so it would save almost nothing.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import onnxruntime as ort
import torch

from ecg import config as C
from ecg import data as D
from ecg import metrics as M
from ecg.model import ECGNet


def load_checkpoint(tag, device="cpu"):
    path = C.MODEL_DIR / f"{tag}_best.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ECGNet(use_rr=ck["use_rr"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def export_fp32(model, path):
    """Dynamic batch axis so the same file serves single-beat live inference
    and batched offline evaluation."""
    dummy_x = torch.zeros(1, 1, C.WINDOW)
    dummy_f = torch.zeros(1, C.N_FEATURES)
    torch.onnx.export(
        model, (dummy_x, dummy_f), str(path),
        input_names=["beat", "rr_features"], output_names=["logits"],
        dynamic_axes={"beat": {0: "batch"}, "rr_features": {0: "batch"},
                      "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    return path


class Calibrator:
    """Feeds real training beats to the quantiser so it can pick per-tensor
    scales from the actual activation ranges rather than guesses."""

    def __init__(self, X, F, n=512):
        from onnxruntime.quantization import CalibrationDataReader  # noqa
        self.data = [{"beat": X[i:i + 1][:, None, :].astype(np.float32),
                      "rr_features": F[i:i + 1].astype(np.float32)}
                     for i in range(min(n, len(X)))]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        item = self.data[self.i]
        self.i += 1
        return item

    def rewind(self):
        self.i = 0


def run_onnx(path, X, F, batch=512):
    sess = ort.InferenceSession(str(path),
                                providers=["CPUExecutionProvider"])
    outs = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch][:, None, :].astype(np.float32)
        fb = F[i:i + batch].astype(np.float32)
        outs.append(sess.run(["logits"], {"beat": xb, "rr_features": fb})[0])
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="model")
    ap.add_argument("--calib", type=int, default=512,
                    help="beats used to calibrate INT8 activation ranges")
    args = ap.parse_args()

    print("=" * 74)
    print(f"STEP 3  -  ONNX EXPORT + INT8 QUANTISATION  ({args.tag})")
    print("=" * 74)

    model, ck = load_checkpoint(args.tag)
    d = D.load_dataset()

    tr = D.mask_for(d["record"], C.TRAIN_RECORDS)
    te = D.mask_for(d["record"], C.TEST_RECORDS)
    Xte, Fte, yte = d["X"][te], d["F"][te], d["y"][te]

    fp32_path = C.MODEL_DIR / f"{args.tag}_fp32.onnx"
    int8_path = C.MODEL_DIR / f"{args.tag}_int8.onnx"

    # ---------------------------------------------------------------- FP32
    export_fp32(model, fp32_path)
    print(f"exported {fp32_path.name}  "
          f"({fp32_path.stat().st_size / 1024:.0f} KB)")

    # ---------------------------------------------------------------- INT8
    from onnxruntime.quantization import (quantize_static, QuantType,
                                          QuantFormat, CalibrationMethod,
                                          CalibrationDataReader)

    # Calibrator first in the MRO, so its concrete get_next() satisfies the
    # abstract method rather than being shadowed by it.
    class Reader(Calibrator, CalibrationDataReader):
        pass

    reader = Reader(d["X"][tr], d["F"][tr], n=args.calib)
    quantize_static(
        str(fp32_path), str(int8_path), reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )
    print(f"exported {int8_path.name}  "
          f"({int8_path.stat().st_size / 1024:.0f} KB)  "
          f"calibrated on {args.calib} training beats")

    # ------------------------------------------------------------- verify
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(Xte).unsqueeze(1),
                             torch.from_numpy(Fte)).numpy()
    torch_pred = torch_logits.argmax(1)

    fp32_pred = run_onnx(fp32_path, Xte, Fte).argmax(1)
    int8_pred = run_onnx(int8_path, Xte, Fte).argmax(1)

    agree_fp32 = float((fp32_pred == torch_pred).mean())
    agree_int8 = float((int8_pred == torch_pred).mean())

    reps = {
        "pytorch_fp32": M.report(yte, torch_pred),
        "onnx_fp32": M.report(yte, fp32_pred),
        "onnx_int8": M.report(yte, int8_pred),
    }

    print()
    print("-" * 74)
    print(f"{'variant':<16}{'macro F1':>11}{'accuracy':>11}"
          f"{'agrees with torch':>20}{'size':>10}")
    print("-" * 74)
    sizes = {"pytorch_fp32": (C.MODEL_DIR / f"{args.tag}_best.pt").stat().st_size,
             "onnx_fp32": fp32_path.stat().st_size,
             "onnx_int8": int8_path.stat().st_size}
    agrees = {"pytorch_fp32": 1.0, "onnx_fp32": agree_fp32,
              "onnx_int8": agree_int8}
    for k, rep in reps.items():
        print(f"{k:<16}{rep['macro_f1']:>11.4f}{rep['accuracy']:>11.4f}"
              f"{agrees[k]:>19.2%}{sizes[k] / 1024:>9.0f}K")
    print("-" * 74)

    delta = reps["onnx_int8"]["macro_f1"] - reps["pytorch_fp32"]["macro_f1"]
    print(f"INT8 macro-F1 change vs PyTorch FP32: {delta:+.4f}")
    if delta < -0.02:
        print("  WARNING: quantisation cost more than 0.02 macro F1. "
              "Consider per-channel weights or a larger calibration set.")
    else:
        print("  Within tolerance - INT8 is safe to deploy.")

    M.print_report(reps["onnx_int8"], "INT8 ONNX on held-out patients")
    M.plot_confusion(reps["onnx_int8"],
                     C.REPORT_DIR / f"{args.tag}_confusion_int8.png",
                     f"ONNX INT8 - {args.tag} - held-out patients")

    with open(C.REPORT_DIR / f"{args.tag}_export.json", "w") as fh:
        json.dump({"tag": args.tag, "agreement_fp32": agree_fp32,
                   "agreement_int8": agree_int8,
                   "sizes_bytes": sizes, "reports": reps}, fh, indent=2)

    print(f"\nSaved  {fp32_path}")
    print(f"Saved  {int8_path}")
    print(f"Saved  {C.REPORT_DIR / f'{args.tag}_export.json'}")


if __name__ == "__main__":
    main()
