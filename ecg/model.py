"""
The multi-input beat classifier.

Two branches:
  * a 1D-CNN over the 288-sample beat window  -> learns SHAPE
      (this is what separates LBBB / RBBB / PVC / normal)
  * a tiny MLP over three RR-timing features  -> learns RHYTHM
      (this is the only thing that can find AFib, whose beats look normal)
concatenated before the classifier head.

No LSTM by design. On a single segmented beat there is almost no long-range
temporal structure for a recurrent layer to exploit, and recurrence is
sequential, so it parallelises badly on the UNO Q's Cortex-A53 cores.

The RR feature standardisation is stored INSIDE the model as buffers, so the
exported ONNX consumes raw seconds and the deployment target needs no
separate constants file to stay in sync.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


def conv_block(cin, cout, k, pool=True):
    layers = [
        nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2, bias=False),
        nn.BatchNorm1d(cout),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class ECGNet(nn.Module):
    """
    use_rr=False builds the CNN-only ablation. The forward signature is
    unchanged so the same export and evaluation code drives both variants.
    """

    def __init__(self, n_classes=C.N_CLASSES, n_features=C.N_FEATURES,
                 use_rr=True):
        super().__init__()
        self.use_rr = use_rr
        self.n_features = n_features

        self.cnn = nn.Sequential(
            conv_block(1, 32, 7),     # 288 -> 144
            conv_block(32, 64, 5),    # 144 -> 72
            conv_block(64, 64, 3),    # 72  -> 36
            conv_block(64, 128, 3, pool=False),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.mx = nn.AdaptiveMaxPool1d(1)
        cnn_out = 128 * 2

        # Standardisation constants for the RR features, filled in by
        # set_feature_scaler() from TRAIN-set statistics only.
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

        if use_rr:
            self.rr = nn.Sequential(
                nn.Linear(n_features, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 32), nn.ReLU(inplace=True),
            )
            fuse_in = cnn_out + 32
        else:
            self.rr = None
            fuse_in = cnn_out

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fuse_in, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def set_feature_scaler(self, mean, std):
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(std, dtype=torch.float32)
                            .clamp_min(1e-6))

    def forward(self, x, f):
        """
        x : (B, 1, 288) z-normalised beat window
        f : (B, 3)      raw RR features in seconds / ratios
        """
        h = self.cnn(x)
        h = torch.cat([self.avg(h), self.mx(h)], dim=1).flatten(1)

        if self.use_rr:
            fs = (f - self.feat_mean) / self.feat_std
            h = torch.cat([h, self.rr(fs)], dim=1)

        return self.head(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model):
    total = count_params(model)
    cnn = sum(p.numel() for p in model.cnn.parameters())
    rr = sum(p.numel() for p in model.rr.parameters()) if model.rr else 0
    head = sum(p.numel() for p in model.head.parameters())
    return (f"parameters: {total:,} total  "
            f"(cnn {cnn:,} / rr {rr:,} / head {head:,})")
