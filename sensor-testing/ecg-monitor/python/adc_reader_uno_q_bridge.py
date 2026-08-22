"""
adc_reader_uno_q_bridge.py
============================
REAL hardware data source for the Uno Q ("tally"), pairing with
sketch/sketch.ino.

The MCU sketch reads the BioAmp EXG Pill on A0 at a fixed 250 Hz and pushes
batches of 25 raw ADC samples (100 ms of data) over the Bridge via:

    Bridge.notify("ecg_batch", ecg_batch)   // ecg_batch: std::vector<int>

The sketch also reads an MPU6050 IMU and, whenever it detects a big
displacement (the sensor/patient moved), pauses ECG collection on the MCU
side and notifies the state change:

    Bridge.notify("motion_state", 1)   // 1 = motion started, 0 = stopped

This script runs on the Linux (MPU) side, registers matching handlers with
Bridge.provide(...) for both events. ECG samples are printed one number per
line, flushed immediately, in the exact order they were sampled. Motion
state changes are printed as a "MOTION:<0|1>" sentinel line on the SAME
stdout stream -- live_stream.py's LiveReader recognizes that prefix and
updates a motion flag instead of treating it as a sample, so no separate
channel/socket is needed.

That output format is intentionally identical to adc_reader_template.py's
simulated output, so this script is a drop-in replacement data source:
anywhere you'd run

    python3 adc_reader_template.py --fs 250

instead run

    python3 adc_reader_uno_q_bridge.py

as the "Data source command" in ecg_gui_live.py / ecg_web_dashboard.py
(this is what python/main.py does by default in "live" and "web" modes
when --source real is selected).

REQUIREMENTS:
  - sketch/sketch.ino must be uploaded to the Uno Q's MCU and
    running (App Lab / arduino-app-cli handles this as part of the app).
  - The `arduino` Python package (arduino.app_utils) must be installed
    on the Linux side. It ships with Arduino App Lab; if you're running
    outside App Lab in your own venv, install it per Arduino's docs.

Run standalone to sanity-check the hardware link:
    python3 adc_reader_uno_q_bridge.py
    (prints one number per line as real samples arrive -- Ctrl+C to stop)
"""

import sys

from arduino.app_utils import App, Bridge


def ecg_batch(values):
    """
    Called by the Bridge every time the MCU sketch notifies a new batch.

    `values` is a list/vector of raw ADC ints (BATCH_SIZE = 25 samples,
    i.e. 100 ms of data at 250 Hz), in acquisition order. Emit them one
    per line so downstream consumers (LiveReader, the GUIs, the web
    dashboard) see exactly the same stream shape they'd get from
    adc_reader_template.py.
    """
    for value in values:
        print(value, flush=True)


def motion_state(value):
    """
    Called by the Bridge whenever the MCU's motion detector flips
    state (see MOTION_THRESHOLD_G / update_motion_state() in
    sketch.ino). Only fires on state changes, not continuously.

    Printed as a "MOTION:<0|1>" sentinel line (not a numeric ECG
    sample) on the same stdout stream ecg_batch() already writes to.
    live_stream.py's LiveReader parses that prefix separately from
    numeric samples and exposes it via status()["motion_detected"],
    which the dashboards use to pause/warn the user.
    """
    print(f"MOTION:{int(value)}", flush=True)


def main():
    Bridge.provide("ecg_batch", ecg_batch)
    Bridge.provide("motion_state", motion_state)
    try:
        App.run()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
