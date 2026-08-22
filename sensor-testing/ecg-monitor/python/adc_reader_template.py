"""
adc_reader_template.py
=======================
This is the script the live GUI runs as a subprocess to get ECG samples
in real time. It must print ONE numeric sample per line to stdout,
flushed immediately, at roughly your sampling rate, forever (until
killed by the GUI).

Right now this file only SIMULATES a signal (a fake but ECG-shaped
waveform) so you can test the live-plotting pipeline before wiring up
real hardware. Replace the marked section below with your actual
Uno Q ADC/GPIO read call.

Test it standalone first:
    python3 adc_reader_template.py --fs 250
    (prints one number per line, forever — Ctrl+C to stop)

Optionally simulate the MPU6050 "motion / be still" behavior (see
sketch.ino + adc_reader_uno_q_bridge.py) without any hardware attached,
to try out the dashboard's "please be still" popup:
    python3 adc_reader_template.py --fs 250 --simulate-motion
    (periodically emits a few seconds of simulated displacement)
"""

import sys
import time
import argparse
import math
import random


def read_real_adc_sample():
    """
    TODO: Replace this with your actual Uno Q ADC/GPIO read.

    Whatever API you use to read the BioAmp EXG Pill on the Uno Q
    (e.g. a sysfs ADC path, a GPIO/analog library, a bridge to the
    onboard microcontroller, etc.) — call it here and `return` a
    single float sample. This function is called once per sample.

    Example shape (pseudocode — swap in your real read call):

        import my_uno_q_adc_library as adc
        channel = adc.open("A0")
        ...
        def read_real_adc_sample():
            return channel.read_voltage()
    """
    raise NotImplementedError(
        "Wire this up to your Uno Q's ADC/GPIO read call, then run with --source real"
    )


def simulate_sample(t, fs):
    """Fake ECG-ish waveform: a beat-shaped pulse ~75 bpm plus a little noise."""
    beat_period = 0.8  # seconds -> 75 bpm
    phase = (t % beat_period) / beat_period
    # crude PQRST-ish shape from a couple of gaussians
    r_wave = 1.6 * math.exp(-((phase - 0.35) ** 2) / (2 * 0.004))
    t_wave = 0.35 * math.exp(-((phase - 0.55) ** 2) / (2 * 0.02))
    p_wave = 0.15 * math.exp(-((phase - 0.15) ** 2) / (2 * 0.01))
    noise = random.gauss(0, 0.02)
    return p_wave + r_wave + t_wave + noise


def main():
    parser = argparse.ArgumentParser(description="ECG sample source for the live GUI")
    parser.add_argument("--fs", type=float, default=250.0, help="Samples per second to emit")
    parser.add_argument(
        "--source", choices=["simulate", "real"], default="simulate",
        help="Use 'simulate' for a fake test signal, 'real' once read_real_adc_sample() is implemented",
    )
    parser.add_argument(
        "--simulate-motion", action="store_true",
        help="Also emit fake MOTION:0/MOTION:1 sentinel lines (see live_stream.py) "
             "so you can try the dashboard's 'please be still' popup with no MPU6050 attached.",
    )
    args = parser.parse_args()

    period = 1.0 / args.fs
    start = time.time()
    n = 0

    # Simple demo schedule for --simulate-motion: still for a while, a
    # few seconds of "motion", repeat. Mirrors the real sketch's
    # edge-triggered MOTION:<0|1> sentinel lines exactly.
    MOTION_STILL_S = 12.0
    MOTION_ACTIVE_S = 4.0
    MOTION_CYCLE_S = MOTION_STILL_S + MOTION_ACTIVE_S
    motion_state = False

    try:
        while True:
            target_t = start + n * period
            now = time.time()
            if target_t > now:
                time.sleep(target_t - now)

            if args.simulate_motion:
                elapsed = n * period
                should_move = (elapsed % MOTION_CYCLE_S) >= MOTION_STILL_S
                if should_move != motion_state:
                    motion_state = should_move
                    print(f"MOTION:{int(motion_state)}", flush=True)

            if motion_state and args.simulate_motion:
                # Data collection is "paused" while moving, same as the
                # real sketch -- don't emit an ECG sample this tick.
                n += 1
                continue

            if args.source == "simulate":
                value = simulate_sample(n * period, args.fs)
            else:
                value = read_real_adc_sample()

            print(value, flush=True)
            n += 1
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
