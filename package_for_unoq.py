"""
Build the deployment bundle for the Arduino UNO Q.

    python package_for_unoq.py

Writes artifacts/unoq_deploy.tar.gz containing ONLY what the board needs:
the INT8 model, the ecg package, the headless monitor, and a requirements
file. No PyTorch, no matplotlib, no wfdb, no dataset - about 100 KB total
instead of the multi-gigabyte training environment.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

from ecg import config as C

ROOT = Path(__file__).resolve().parent

INCLUDE = [
    "ecg/__init__.py",
    "ecg/config.py",
    "ecg/data.py",
    "ecg/model.py",       # not used at inference, but keeps the package whole
    "ecg/qrs.py",
    "ecg/pipeline.py",
    "ecg/sources.py",
    "deploy/uno_q_monitor.py",
    "deploy/requirements-unoq.txt",
    "artifacts/models/model_int8.onnx",
]


def main():
    out = C.ARTIFACTS / "unoq_deploy.tar.gz"
    missing = [p for p in INCLUDE if not (ROOT / p).exists()]
    if missing:
        print("Missing files - run step3_export_onnx.py first:")
        for m in missing:
            print(f"  {m}")
        return

    with tarfile.open(out, "w:gz") as tar:
        for rel in INCLUDE:
            tar.add(ROOT / rel, arcname=f"ecg_unoq/{rel}")

    size = out.stat().st_size
    print(f"Built {out}  ({size / 1024:.0f} KB)")
    print()
    print("Copy it to the board and unpack:")
    print(f"  scp {out.name} <user>@<board-ip>:~/")
    print("  ssh <user>@<board-ip>")
    print("  tar xzf unoq_deploy.tar.gz && cd ecg_unoq")
    print("  python3 -m pip install -r deploy/requirements-unoq.txt")
    print("  python3 deploy/uno_q_monitor.py --serial /dev/ttyACM0 --fs 250")


if __name__ == "__main__":
    main()
