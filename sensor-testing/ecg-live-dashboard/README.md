# ECG UNO Q — Live P-QRS-T Dashboard

Reads the BioAmp EXG Pill (wired to A0) at 250 Hz on the STM32 side,
streams it over the Bridge to Linux, and serves a live browser dashboard
with P-QRS-T detection and clinical interval extraction.

Part of the sensor calibration and output verification work behind the
[ECG arrhythmia classifier](../../README.md) this repo is built around.
An earlier, simpler predecessor to [`ecg-monitor`](../ecg-monitor) — no
motion sensor, otherwise the same idea.

**Known issue:** `app.yaml` has `ports: []`, so unlike `ecg-monitor` (which
declares `ports: [5000]`) the dashboard's port may not actually be published
to the host. Not yet confirmed working end to end for that reason — verify
before relying on it. The classifier's own App Lab app hit the same class of
problem and resolved it with the `arduino:web_ui` Brick instead of `ports:`;
see [`applab/README.md`](../../applab/README.md) for the details.

The block diagram and circuit schematic that used to sit in this folder
describe the classifier's system, not this app, and have moved to
[`docs/images/`](../../docs/images/).

## Contents

- [Project structure](#project-structure)
- [How the pieces link up](#how-the-pieces-link-up)
- [Swapping in your own analysis](#swapping-in-your-own-analysis)
- [Deploy](#deploy)

## Project structure

```
ecg_uno_q_project/
├── app.yaml                     # App Lab manifest (ties python + sketch together)
├── README.md                    # this file
├── python/                      # runs on the MPU (Linux side), launched by arduino-app-cli
│   ├── main.py                  # ENTRY POINT: Bridge receiver + Flask dashboard, in one process
│   ├── ecg_pqrst_detector.py    # core P-QRS-T detection algorithm (filtering, Pan-Tompkins, intervals)
│   ├── pqrst_adapter.py         # wraps the detector as analyze(signal, fs) for main.py to call
│   ├── user_module.py           # optional aid file: minimal analyze() template if you want to
│   │                             #   write your own analysis instead of the P-QRS-T detector
│   └── requirements.txt         # numpy, pandas, scipy, matplotlib, flask
└── sketch/                      # runs on the MCU (STM32 side), flashed by arduino-app-cli
    ├── sketch.ino                # samples A0, batches 25 samples, Bridge.notify("ecg_batch", ...)
    └── sketch.yaml                # board FQBN + Arduino_RouterBridge library pin
```

## How the pieces link up

```
 BioAmp EXG Pill
        │  analog signal
        ▼
   sketch/sketch.ino  (STM32, 250 Hz)
        │  Bridge.notify("ecg_batch", 25 ints)   <-- every 100 ms
        ▼
   python/main.py      (Linux, the ONLY process that owns the Bridge)
        │  Bridge.provide("ecg_batch", on_ecg_batch)
        │  → SensorBuffer (in-memory rolling buffer, 10 s @ 250 Hz)
        │  → Flask dashboard (same process, background thread), served on :5000
        │
        │  "Run Module Now" in the browser calls:
        ▼
   python/pqrst_adapter.py :: analyze(signal, fs)
        │  imports and drives
        ▼
   python/ecg_pqrst_detector.py
        (bandpass+notch filter → Pan-Tompkins R-peaks → P/Q/S/T search →
         HR / PR / QRS / QT extraction, writes ecg_parameters.csv)
        │
        ▼
   result dict → JSON → browser chart + results panel
```

The `"ecg_batch"` string is the only thing linking the sketch to `main.py` —
it must match exactly on both sides. Everything downstream of the buffer
(the adapter, the detector, the dashboard) is plain Python with no Bridge
dependency, so it's easy to test with `sample_ecg.csv` offline if you want
to sanity-check `ecg_pqrst_detector.py` before touching hardware:

```bash
python3 ecg_pqrst_detector.py sample_ecg.csv --fs 250 --plot
```

**Why one process instead of separate scripts:** `arduino.app_utils`
(the `Bridge`/`App` objects) is only importable inside the container
`arduino-app-cli` builds for `python/main.py` specifically. A script
launched any other way — `python3 somescript.py`, or spawned as a
subprocess of another script — sits outside that container and can't
import it. So the Bridge receiver and the dashboard have to be the same
process; that's exactly what `main.py` does (Flask runs in a background
thread, `App.run()` owns the main thread and keeps the Bridge alive).

## Swapping in your own analysis

Point the dashboard's "Analysis module" field at `user_module.py` (or a
copy of it) instead of `pqrst_adapter.py` if you want to try a different
detection approach — it follows the exact same `analyze(signal, fs) -> dict`
contract, just with a simpler scipy `find_peaks`-based R-peak detector
instead of the full P-QRS-T pipeline.

## Deploy

From your computer:
```bash
scp -r * arduino@<UNO_Q_IP_ADDRESS>:~/ArduinoApps/ecg_uno_q_project/
```

On the board (SSH):
```bash
arduino-app-cli app start ~/ArduinoApps/ecg_uno_q_project
arduino-app-cli app logs  ~/ArduinoApps/ecg_uno_q_project   # watch startup + Bridge status
```

Then open `http://<UNO_Q_IP_ADDRESS>:5000` from any device on the same
network. The chart starts moving as soon as the sketch is sending batches
— no separate "start streaming" step needed, since the Bridge connection
is live for as long as the app is running.

Stop it with:
```bash
arduino-app-cli app stop ~/ArduinoApps/ecg_uno_q_project
```
