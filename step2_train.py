"""
STEP 2 - Train the multi-input beat classifier.

    python step2_train.py                 # full CNN + RR fusion model
    python step2_train.py --no-rr         # CNN-only ablation
    python step2_train.py --epochs 10     # quick smoke run

Everything lands on disk as it goes: best checkpoint, last checkpoint,
training history json and a loss/F1 curve. Nothing lives only in RAM.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ecg import config as C
from ecg import data as D
from ecg import metrics as M
from ecg.model import ECGNet, describe


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class BeatDataset(Dataset):
    """
    Augmentation is only applied to the training fold, and every transform
    models a distortion the deployed sensor will actually produce:

      time shift  - the STM32 R-peak detector will not land on exactly the
                    same sample offset the MIT-BIH annotator chose
      time warp   - the SAME condition is wider or narrower from patient to
                    patient; with only 4 LBBB and 6 RBBB patients in the
                    whole database this is the augmentation that actually
                    buys held-out generalisation
      noise       - EMG / mains pickup on a dry-electrode chest lead
      baseline    - residual wander that survives the 0.5 Hz highpass
      gain        - electrode contact quality varies between sessions
      rr jitter   - R-peak timing error propagates into the RR features
    """

    def __init__(self, X, F, y, augment=False):
        self.X, self.F, self.y = X, F, y
        self.augment = augment
        self.rng = np.random.default_rng(C.SEED)

    def __len__(self):
        return len(self.y)

    def _warp(self, x):
        """Resample the window by a random factor, keeping the R-peak pinned
        at its original index, then crop/pad back to WINDOW samples."""
        factor = self.rng.uniform(0.88, 1.12)
        n = len(x)
        src = np.arange(n, dtype=np.float32)
        # position sampled from the original, measured from the R-peak
        q = C.BEFORE + (src - C.BEFORE) * factor
        return np.interp(q, src, x).astype(np.float32)

    def __getitem__(self, i):
        x = self.X[i].copy()
        f = self.F[i].copy()

        if self.augment:
            x = self._warp(x)

            shift = int(self.rng.integers(-10, 11))
            if shift:
                x = np.roll(x, shift)
                if shift > 0:
                    x[:shift] = x[shift]
                else:
                    x[shift:] = x[shift - 1]

            x = x * self.rng.uniform(0.85, 1.15)
            x = x + self.rng.normal(0, 0.05, size=x.shape).astype(np.float32)

            t = np.linspace(0, 1, len(x), dtype=np.float32)
            x = x + self.rng.uniform(-0.15, 0.15) * t

            f = f * self.rng.normal(1.0, 0.02, size=f.shape).astype(np.float32)

        return (torch.from_numpy(x.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(f.astype(np.float32)),
                int(self.y[i]))


def make_fold(d, records, augment=False):
    m = D.mask_for(d["record"], records)
    return BeatDataset(d["X"][m], d["F"][m], d["y"][m], augment=augment)


# --------------------------------------------------------------------------
# Weight EMA
# --------------------------------------------------------------------------
class EMA:
    """
    Exponential moving average of the weights.

    Without this, held-out macro F1 swung between 0.47 and 0.73 from one
    epoch to the next and "best checkpoint" was a lottery: two runs that
    differed only in which epoch spiked gave LBBB F1 of 0.77 and 0.56. Only
    5 validation patients exist, so the validation signal is genuinely noisy
    - averaging the weights over roughly the last four epochs turns that
    noise into something a checkpoint can actually be selected on.

    Buffers (BatchNorm running stats, the RR feature scaler) are averaged
    too, so the EMA copy is directly usable with no recalibration pass.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if self.shadow[k].dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


# --------------------------------------------------------------------------
# Train / eval loops
# --------------------------------------------------------------------------
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None,
              ema=None):
    train = optimizer is not None
    model.train(train)

    total_loss, n = 0.0, 0
    preds, trues = [], []

    for x, f, y in loader:
        x, f, y = x.to(device, non_blocking=True), f.to(device), y.to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=scaler is not None):
                out = model(x, f)
                loss = criterion(out, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if ema is not None:
                    ema.update(model)

        total_loss += loss.item() * len(y)
        n += len(y)
        preds.append(out.argmax(1).detach().cpu().numpy())
        trues.append(y.cpu().numpy())

    return (total_loss / n,
            np.concatenate(trues), np.concatenate(preds))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rr", action="store_true",
                    help="CNN-only ablation: drop the RR feature branch")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--tag", default=None,
                    help="checkpoint name; defaults to model / model_cnn_only")
    ap.add_argument("--include-val", action="store_true",
                    help="fold the validation patients into training and keep "
                         "the FINAL EMA weights instead of the best-val ones. "
                         "The rare classes live in so few patients that the 5 "
                         "val records are worth more as training diversity "
                         "(LBBB 2->3 patients, RBBB 3->4, AFIB 4->5) than as "
                         "a noisy selection signal. TEST stays untouched, so "
                         "the reported test numbers remain honest; only the "
                         "epoch count is inherited from the val-selected runs.")
    args = ap.parse_args()

    use_rr = not args.no_rr
    tag = args.tag or ("model" if use_rr else "model_cnn_only")

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 74)
    print(f"STEP 2  -  TRAIN  ({'CNN + RR fusion' if use_rr else 'CNN only'})")
    print("=" * 74)

    d = D.load_dataset()
    train_recs = (C.TRAIN_RECORDS + C.VAL_RECORDS if args.include_val
                  else C.TRAIN_RECORDS)
    train_ds = make_fold(d, train_recs, augment=C.AUGMENT)
    val_ds = make_fold(d, C.VAL_RECORDS)
    test_ds = make_fold(d, C.TEST_RECORDS)
    if args.include_val:
        print("NOTE: validation patients folded into TRAINING. The 'val'"
              "\n      column below is therefore in-sample and is printed for"
              "\n      monitoring only - it is NOT used to select the "
              "checkpoint.")

    print(f"device            : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"train / val / test: {len(train_ds):,} / {len(val_ds):,} / "
          f"{len(test_ds):,} beats")

    # --- feature scaler from TRAIN ONLY -----------------------------------
    f_mean = train_ds.F.mean(axis=0)
    f_std = train_ds.F.std(axis=0)
    print("rr feature scaler :")
    for name, mu, sd in zip(C.FEATURE_NAMES, f_mean, f_std):
        print(f"    {name:<16} mean {mu:+.4f}  std {sd:.4f}")

    # --- class weights ----------------------------------------------------
    counts = np.bincount(train_ds.y, minlength=C.N_CLASSES).astype(np.float64)
    weights = (counts.sum() / (C.N_CLASSES * counts)) ** C.CLASS_WEIGHT_POWER
    print(f"class weights     : (inverse freq ^ {C.CLASS_WEIGHT_POWER})  "
          + "  ".join(f"{c}={w:.2f}" for c, w in zip(C.CLASSES, weights)))

    # --- model ------------------------------------------------------------
    model = ECGNet(use_rr=use_rr).to(device)
    model.set_feature_scaler(f_mean, f_std)
    print(describe(model))
    print()

    loaders = {
        "train": DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=(device == "cuda"),
                            drop_last=True),
        "val": DataLoader(val_ds, batch_size=512, shuffle=False),
        "test": DataLoader(test_ds, batch_size=512, shuffle=False),
    }

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=C.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR,
                                  weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    ema = EMA(model, decay=C.EMA_DECAY)
    ema_model = ECGNet(use_rr=use_rr).to(device)

    best_f1, best_epoch = -1.0, -1
    history = []
    ckpt_best = C.MODEL_DIR / f"{tag}_best.pt"
    t0 = time.time()

    print(f"{'epoch':>6}{'train loss':>12}{'val loss':>11}{'val F1':>9}"
          f"{'ema F1':>9}{'lr':>10}   ")
    print("-" * 74)

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, loaders["train"], device, criterion,
                                  optimizer, scaler, ema=ema)
        va_loss, yv, pv = run_epoch(model, loaders["val"], device, criterion)
        raw_f1 = M.report(yv, pv)["macro_f1"]

        ema.copy_to(ema_model)
        _, yv2, pv2 = run_epoch(ema_model, loaders["val"], device, criterion)
        f1 = M.report(yv2, pv2)["macro_f1"]

        lr = optimizer.param_groups[0]["lr"]
        sched.step()

        # With --include-val there is no honest held-out signal left, so we
        # simply keep the latest EMA weights and stop at --epochs.
        flag = ""
        if args.include_val or f1 > best_f1:
            if not args.include_val:
                best_f1, best_epoch = f1, epoch
                flag = "  <- best, saved"
            else:
                best_f1, best_epoch = f1, epoch
            torch.save({"state_dict": ema_model.state_dict(),
                        "use_rr": use_rr,
                        "feat_mean": f_mean, "feat_std": f_std,
                        "epoch": epoch, "val_macro_f1": f1,
                        "trained_on": train_recs},
                       ckpt_best)

        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_macro_f1": f1,
                        "val_macro_f1_raw": raw_f1, "lr": lr})
        print(f"{epoch:>6}{tr_loss:>12.4f}{va_loss:>11.4f}{raw_f1:>9.4f}"
              f"{f1:>9.4f}{lr:>10.2e}{flag}")

    print("-" * 74)
    print(f"best val macro F1 (EMA weights) {best_f1:.4f} at epoch "
          f"{best_epoch}   ({time.time() - t0:.0f}s total)")

    torch.save({"state_dict": model.state_dict(), "use_rr": use_rr,
                "feat_mean": f_mean, "feat_std": f_std},
               C.MODEL_DIR / f"{tag}_last.pt")

    # --- final numbers on the held-out patients ---------------------------
    ck = torch.load(ckpt_best, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])

    _, yt, pt = run_epoch(model, loaders["test"], device, criterion)
    test_rep = M.report(yt, pt)
    M.print_report(test_rep, f"TEST SET - {len(C.TEST_RECORDS)} unseen "
                             f"patients (best checkpoint, epoch "
                             f"{ck['epoch']})")

    out = {"tag": tag, "use_rr": use_rr, "epochs": args.epochs,
           "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
           "test": test_rep, "history": history,
           "train_records": C.TRAIN_RECORDS, "val_records": C.VAL_RECORDS,
           "test_records": C.TEST_RECORDS}
    with open(C.REPORT_DIR / f"{tag}_training.json", "w") as fh:
        json.dump(out, fh, indent=2)

    M.plot_confusion(test_rep, C.REPORT_DIR / f"{tag}_confusion_fp32.png",
                     f"PyTorch FP32 - {'CNN+RR' if use_rr else 'CNN only'} "
                     f"- held-out patients")
    plot_history(history, C.REPORT_DIR / f"{tag}_curves.png", tag)

    print(f"\nSaved  {ckpt_best}")
    print(f"Saved  {C.REPORT_DIR / f'{tag}_training.json'}")
    print(f"Saved  {C.REPORT_DIR / f'{tag}_confusion_fp32.png'}")
    print(f"Saved  {C.REPORT_DIR / f'{tag}_curves.png'}")


def plot_history(history, path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["val_loss"] for h in history], label="val")
    a1.set_xlabel("epoch"); a1.set_ylabel("weighted CE loss")
    a1.legend(); a1.grid(alpha=0.3); a1.set_title("loss")

    a2.plot(ep, [h["val_macro_f1"] for h in history], color="tab:green")
    a2.set_xlabel("epoch"); a2.set_ylabel("macro F1")
    a2.grid(alpha=0.3); a2.set_title("validation macro F1")

    fig.suptitle(f"{tag} - training history")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
