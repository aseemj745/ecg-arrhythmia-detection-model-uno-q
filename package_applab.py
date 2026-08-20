"""
Build the Arduino App Lab importable .zip.

    python package_applab.py

Produces artifacts/ecg_arrhythmia_unoq.zip, which App Lab's
"Import an App -> Import from computer" accepts.

IMPORTANT - the manifest is a best guess.
App Lab 0.8.0's exact app.yaml schema has not been verified against this
board, because apps live on the UNO Q itself and not on the PC. If the
import is rejected, do NOT fight it: run
    python package_applab.py --show-manifest-help
and follow the instructions to dump the manifest of an existing working app
(e.g. "ECG Monitor"), then paste it back so this file can be corrected.

The self-test does not depend on App Lab at all - the same folder can be
copied to the board and run directly with python3, which is the fallback if
importing proves fussy.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APPDIR = ROOT / "applab"
APP_NAME = "ecg_arrhythmia_unoq"

# Fields (name/icon/description/version/bricks) and their layout confirmed
# against real app.yaml files in arduino/app-bricks-examples (e.g.
# system-resources-logger, 08-web-ui-basics/02-data-transmission) - not a
# guess. `bricks: - arduino:web_ui` is what actually gets the dashboard's
# port published from the container to the host; nothing else on this
# platform does that (searched all 156 example app.yaml files for a real
# `ports:` value - zero hits).
MANIFEST = """\
name: ECG Arrhythmia Classifier
icon: "\U0001F493"
description: >-
  Per-beat arrhythmia classification (Normal / LBBB / RBBB / PVC / AFib) from
  a single ECG lead, running an INT8 ONNX CNN on the QRB2210 Linux side. Runs
  a no-hardware self-test on a held-out MIT-BIH patient by default, and
  serves a live dashboard once connected to the BioAmp EXG Pill via the MCU
  Bridge.
version: "1.0.0"
bricks:
  - arduino:web_ui
"""

HELP = """
HOW TO GET THE REAL MANIFEST FORMAT
===================================
App Lab apps live on the UNO Q, not on your PC. Open the App Lab terminal
(the >_ button at the bottom of the App Lab window, next to the board name)
and run:

    ls /app
    find / -maxdepth 4 -name "app.yaml" 2>/dev/null | head
    cat <path to an existing app>/app.yaml

Paste the output back and the manifest in package_applab.py can be made to
match exactly instead of guessed.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-manifest-help", action="store_true")
    ap.add_argument("--no-wheels", action="store_true",
                    help="exclude the bundled offline wheels (~17 MB). Use "
                         "once the board has onnxruntime installed, or if "
                         "App Lab rejects the larger zip.")
    args = ap.parse_args()
    if args.show_manifest_help:
        print(HELP)
        return

    if not (APPDIR / "models" / "model_int8.onnx").exists():
        print("applab/models/model_int8.onnx missing - "
              "run step3_export_onnx.py, then rebuild the applab folder.")
        return

    # keep the shipped copy of the library in step with the trained one
    shutil.rmtree(APPDIR / "ecg", ignore_errors=True)
    shutil.copytree(ROOT / "ecg", APPDIR / "ecg",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "artifacts" / "models" / "model_int8.onnx",
                 APPDIR / "models" / "model_int8.onnx")
    (APPDIR / "app.yaml").write_text(MANIFEST, encoding="utf-8")
    # MUST be python/requirements.txt, not an app-root file - confirmed
    # against arduino/app-bricks-examples (system-resources-logger,
    # mascot-jump-game): the App Lab container auto-installs from this exact
    # path on start, picked up by convention, no app.yaml reference needed.
    # An app-root requirements.txt is silently ignored, which is the bug
    # that made the container crash with ModuleNotFoundError: onnxruntime.
    #
    # Pinned to the EXACT versions verified on the desktop (numpy 2.5.1,
    # scipy 1.18.0, onnxruntime 1.28.0 - see test.py's original output).
    # This container is isolated per-app, unlike the shared host Python, so
    # there is no apt-shadowing risk here (that only matters when installing
    # directly on the board's own Python, outside this container).
    (APPDIR / "python" / "requirements.txt").write_text(
        "numpy==2.5.1\nscipy==1.18.0\nonnxruntime==1.28.0\n", encoding="utf-8")

    out = ROOT / "artifacts" / f"{APP_NAME}.zip"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        out.unlink()

    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(APPDIR.rglob("*")):
            if p.is_dir() or "__pycache__" in p.parts:
                continue
            if p.name == "detections.csv":
                continue
            if args.no_wheels and "wheels" in p.parts:
                continue
            z.write(p, arcname=f"{APP_NAME}/{p.relative_to(APPDIR)}")
            n += 1

    mb = out.stat().st_size / 1024 / 1024
    print(f"Built {out}")
    print(f"  {n} files, {mb:.1f} MB"
          + ("" if args.no_wheels else "  (includes offline aarch64 wheels)"))
    print()
    if not args.no_wheels:
        print("onnxruntime ships inside the app, pinned to the version the")
        print("model was verified against. On the board, before first run:")
        print("            sh install_deps.sh")
        print("(safe even if a different onnxruntime is already installed -")
        print(" the script force-reinstalls the pinned version with --no-deps)")
        print()
    print("Import it:  App Lab -> Create new app + -> Import an App ->")
    print("            Import from computer -> pick this .zip")
    print()
    print("Then run the self-test (no sensor needed):")
    print("            python3 python/main.py")
    print()
    print("If the import is rejected, run:")
    print("            python package_applab.py --show-manifest-help")


if __name__ == "__main__":
    main()
