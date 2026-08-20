# App Lab app

This folder is the actual app that runs on the UNO Q's Linux side, imported
through Arduino App Lab. It loads `model_int8.onnx` and classifies each
heartbeat live, with a browser dashboard showing the waveform and results.

See [`../DEPLOY_UNO_Q.md`](../DEPLOY_UNO_Q.md) for how to build and import
this as a zip.

## Layout

```
python/main.py              entry point (self-test + live modes)
ecg/                        copy of the shared library (kept in sync with the root ecg/)
models/model_int8.onnx      the deployed model
data/sample_ecg_250hz.csv   90s of MIT-BIH record 119 @ 250Hz, for the no-hardware self-test
wheels/                     offline aarch64 wheels, in case the board has no internet
install_deps.sh             offline dependency install
```

## How it works

Samples in (any rate) → resampled to 360 Hz → 0.5–40 Hz bandpass →
Pan-Tompkins R-peak detection → 288-sample window per beat → z-normalise →
polarity-normalise → 10 RR-based rhythm features → ONNX INT8 → per-beat
class → consecutive same-label beats grouped into episodes.

Rhythm features need about 12 beats of history before the first
classification, so there's a real ~10s warmup at startup. That's expected,
not a bug.

Run the self-test with no board and no sensor:

```bash
python3 python/main.py --mode selftest
```

Expect roughly 15–25 episodes, all PVC (record 119 is a bigeminy patient),
and a `SELF-TEST PASSED` line.

## A few things about App Lab that weren't obvious from the docs

**Each app runs in its own Docker container**, not as a bare process on the
board. So `python3 python/main.py --mode live` run from a plain terminal
will always fail with `ModuleNotFoundError: No module named 'arduino'` — the
`arduino` package only exists inside the container. Running the app through
App Lab's own Start button (`apps_start`) is the only way to actually reach
the Bridge.

**Dependencies have to live at `python/requirements.txt`**, not the app
root — the container only picks them up from there.

**`apps_start` can't pass command-line arguments** to the container, so
whatever `main.py`'s default mode is is what actually runs when someone
presses Start. That's why the default is `--mode live` rather than
`--mode selftest`.

**The Bridge is push-based.** The MCU calls
`Bridge.notify("ecg_batch", batch)`, and the Linux side registers a callback
with `Bridge.provide("ecg_batch", callback)` — there's no way to poll for
new samples from the Linux side.

## The live dashboard

The dashboard is served through App Lab's `arduino:web_ui` Brick
(`bricks: [arduino:web_ui]` in `app.yaml`). A plain `http.server` looked
like it worked when tested directly on the board, but wasn't actually
reachable from outside the container — each app gets its own private Docker
network with nothing published to the host. The `web_ui` Brick is what
Arduino's own examples use to get a port actually published, and it's
confirmed working: page loads, live waveform updates, episode table fills
in as beats are classified.

There's also a **"Start Demo Replay" button** on the dashboard. Since
`apps_start` always launches live mode, and there's no point showing an
empty dashboard while waiting for a sensor, this button feeds the bundled
MIT-BIH recording into the exact same code path a real sensor would use.
While it's running, the dashboard clearly labels it `DEMO REPLAY — NOT a
live sensor`, so it's never confused with an actual reading.

## Known limitations

See the root [`README.md`](../README.md#honest-limitations) for the model's
accuracy limitations and the current sample-rate / live-sensor status.
