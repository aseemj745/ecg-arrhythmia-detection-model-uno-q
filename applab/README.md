# App Lab app

This folder is the app that runs on the UNO Q's Linux side, imported through
Arduino App Lab. It loads `model_int8.onnx`, classifies each heartbeat as it
arrives, and serves a browser dashboard with the waveform and the results.

See [`../DEPLOY_UNO_Q.md`](../DEPLOY_UNO_Q.md) for how to build and import it
as a zip.

## Layout

```
python/main.py              entry point (selftest + live modes)
ecg/                        copy of the shared library, kept in sync with the root ecg/
models/model_int8.onnx      the deployed model
sketch/sketch.ino           MCU-side (STM32) sketch: samples the BioAmp EXG Pill at
                             125Hz, bandpass-filters it, pushes batches over the
                             Bridge, and handles MPU6050 motion gating. Started from
                             a teammate's working app rather than written from scratch.
data/sample_ecg_250hz.csv   90s of MIT-BIH record 119 @ 250Hz, for the selftest
data/demo_*.csv             8 test-fold records (one per class plus a couple of hard
                             cases) for the dashboard's Demo Replay dropdown
wheels/                     offline aarch64 wheels, for a board with no internet
install_deps.sh             offline dependency install
```

## How it works

Samples in (any rate) -> resample to 360 Hz -> 0.5-40 Hz bandpass ->
Pan-Tompkins R-peak detection -> 288-sample window per beat -> z-normalise ->
polarity-normalise -> 10 RR rhythm features -> ONNX INT8 -> class per beat ->
consecutive same-label beats grouped into episodes.

The rhythm features need about 12 beats of history before the first
classification, so there is a warmup of roughly 10 seconds at startup. That is
expected.

Run the selftest with no board and no sensor:

```bash
python3 python/main.py --mode selftest
```

Expect somewhere between 15 and 25 episodes, all PVC (record 119 is a bigeminy
patient), and a `SELF-TEST PASSED` line at the end.

## App Lab notes

A few things that were not obvious from the App Lab docs:

**Each app runs in its own Docker container**, not as a plain process on the
board. So running `python3 python/main.py --mode live` from a normal terminal
always fails with `ModuleNotFoundError: No module named 'arduino'`, because the
`arduino` package only exists inside the container. Starting the app through
App Lab's own Start button is the only way to reach the Bridge.

**Dependencies have to be at `python/requirements.txt`**, not at the app root.
The container only picks them up from there.

**`apps_start` cannot pass command-line arguments** to the container, so
whatever `main.py`'s default mode is is what runs when someone presses Start.
That is why the default is `--mode live` and not `--mode selftest`.

**The Bridge is push-based.** The MCU calls `Bridge.notify("ecg_batch", batch)`
and the Linux side registers a callback with
`Bridge.provide("ecg_batch", callback)`. There is no way to poll for new
samples from the Linux side.

## The dashboard

The dashboard is served through App Lab's `arduino:web_ui` Brick
(`bricks: [arduino:web_ui]` in `app.yaml`). A plain `http.server` looked like it
worked when tested directly on the board but was not reachable from outside the
container, because each app gets a private Docker network with nothing
published to the host. The `web_ui` Brick is what Arduino's own examples use to
get a port published, and it works here: the page loads, the waveform updates,
and the episode table fills in as beats are classified.

There are three controls: **Live Sensor**, **Demo Replay** and **Stop**. Demo
Replay feeds one of the 8 bundled test-fold recordings, picked from a dropdown,
into the same code path a real sensor push would take. So it is a real test of
the classifier, just not of the sensor. While it runs, the dashboard labels it
`DEMO REPLAY - NOT a live sensor` so it cannot be mistaken for a real reading.

There is also a motion log. The MCU sketch sends a `motion_state` Bridge event
when its MPU6050 sees a sudden movement, and the dashboard marks those spans on
the waveform instead of showing whatever the electrodes picked up while the
patient was moving.

## Limitations

See the root [`README.md`](../README.md#limitations) for the model's accuracy
limitations and the current live-sensor status.
