"""
main.py
=======
Single entry point for the ECG Monitor app. This is the file App Lab /
arduino-app-cli launches on the Linux (MPU) side, as declared in app.yaml.

It just wires together the modules already in this folder:

  live_stream.py              -> LiveReader (subprocess + rolling buffer)
  adc_reader_template.py      -> simulated data source (no hardware needed)
  adc_reader_uno_q_bridge.py  -> real data source (BioAmp EXG Pill via the
                                  MCU sketch in ../sketch/sketch.ino)
  ecg_web_dashboard.py        -> browser dashboard (default: no monitor needed)
  ecg_gui_live.py             -> Tkinter live desktop GUI (needs a display)
  ecg_gui.py                  -> Tkinter offline file-viewer GUI
  ecg_pqrst_detector.py       -> the actual PQRST detection pipeline (CLI)
  pqrst_adapter.py            -> wraps ecg_pqrst_detector.py as analyze(signal, fs)
  user_module.py              -> a simpler example analyze(signal, fs)

Usage:
    python3 main.py                          # web dashboard, simulated data (default)
    python3 main.py --mode web --source real  # web dashboard, real BioAmp Pill
    python3 main.py --mode live --source real # desktop live GUI, real hardware
    python3 main.py --mode gui                # offline file-viewer GUI
    python3 main.py --mode cli sample_ecg.csv --fs 250   # headless CLI analysis

Modes:
  web   (default) Starts ecg_web_dashboard.py's Flask server on --port
                   (5000 by default). View from any browser on the network at
                   http://<tally-ip>:<port>. No X11/monitor required on tally.
  live             Launches ecg_gui_live.py's Tkinter live-streaming desktop
                   GUI. Requires a monitor attached to tally (or forwarded X11).
  gui              Launches ecg_gui.py's Tkinter offline viewer, for loading
                   and analyzing an already-recorded CSV/TXT file.
  cli              Runs ecg_pqrst_detector.py directly against a CSV file,
                   no GUI/server involved -- prints the summary and writes
                   ecg_parameters.csv. Pass the CSV path as a positional arg.

--source controls which acquisition script feeds the live modes (web/live):
  simulate (default) adc_reader_template.py  -- fake ECG-shaped waveform,
                       no hardware required, good for testing the pipeline.
  real                adc_reader_uno_q_bridge.py -- real samples from the
                       BioAmp EXG Pill via the Bridge, pushed by
                       sketch/sketch.ino running on the MCU.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SIMULATE_CMD = f"python3 {os.path.join(_HERE, 'adc_reader_template.py')} --fs {{fs}}"
REAL_CMD = f"python3 {os.path.join(_HERE, 'adc_reader_uno_q_bridge.py')}"

DEFAULT_MODULE = os.path.join(_HERE, "pqrst_adapter.py")


def _source_command(source: str, fs: float) -> str:
    if source == "real":
        # adc_reader_uno_q_bridge.py takes its sample rate from the MCU
        # sketch (fixed at 250 Hz), so no --fs flag is passed to it.
        return REAL_CMD
    return SIMULATE_CMD.format(fs=fs)


def run_web(args):
    import ecg_web_dashboard as web

    # Pre-fill the dashboard's defaults (used by /api/start when the page
    # posts with no override) so it starts with the right data source
    # command already selected, without editing ecg_web_dashboard.py itself.
    command = _source_command(args.source, args.fs)
    web.DEFAULT_COMMAND = command
    web.DEFAULT_FS = args.fs
    web.DEFAULT_WINDOW_S = args.window

    # Also patch the initial HTML so the visible input fields match --source
    # / --fs / the analysis module on first load, not just the JS defaults.
    web.PAGE = (
        web.PAGE
        .replace(
            'value="python3 adc_reader_template.py --fs 250"',
            f'value="{command}"',
        )
        .replace('id="fs" value="250"', f'id="fs" value="{args.fs:g}"')
        .replace('id="win" value="10"', f'id="win" value="{args.window:g}"')
        .replace(
            'value="pqrst_adapter.py"',
            f'value="{DEFAULT_MODULE}"',
        )
    )

    os.environ["PORT"] = str(args.port)
    web.main()


def run_live(args):
    import tkinter as tk
    from ecg_gui_live import ECGLiveApp

    root = tk.Tk()
    app = ECGLiveApp(root)
    app.cmd_var.set(_source_command(args.source, args.fs))
    app.fs_var.set(str(args.fs))
    app.window_var.set(str(args.window))
    app.module_var.set(DEFAULT_MODULE)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


def run_gui(args):
    import tkinter as tk
    from ecg_gui import ECGViewerApp

    root = tk.Tk()
    app = ECGViewerApp(root)
    app.module_path_var.set(DEFAULT_MODULE)
    if args.csv:
        app.ecg_path_var.set(args.csv)
        app.fs_var.set(str(args.fs))
        app.load_file()
    root.mainloop()


def _load_last_numeric_column(path):
    """
    Load a CSV as raw samples, robust to both:
      - headerless raw ADC dumps (ecg_pqrst_detector.py's original assumption)
      - a labeled column, e.g. sample_ecg.csv's single "ecg" header
    Returns the last column as a 1D float array (amplitude, if two columns
    are time+amplitude).
    """
    import pandas as pd

    with open(path) as f:
        first_line = f.readline()
    last_token = first_line.strip().split(",")[-1]
    try:
        float(last_token)
        header = None  # first row is already numeric data
    except ValueError:
        header = 0  # first row is a column label

    data = pd.read_csv(path, header=header)
    return data.iloc[:, -1].to_numpy(dtype=float)


def run_cli(args):
    if not args.csv:
        sys.exit("cli mode requires a CSV path, e.g.: python3 main.py --mode cli sample_ecg.csv --fs 250")
    import ecg_pqrst_detector as pqrst

    raw_signal = _load_last_numeric_column(args.csv)

    filtered = pqrst.preprocess(raw_signal, args.fs, powerline_freq=args.powerline)
    r_peaks = pqrst.pan_tompkins_r_peaks(filtered, args.fs)
    print(f"Detected {len(r_peaks)} R-peaks")

    complexes = pqrst.detect_pqrst_complex(filtered, args.fs, r_peaks)
    df, summary = pqrst.extract_parameters(complexes, args.fs)

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    out_csv = os.path.join(_HERE, "ecg_parameters.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nPer-beat parameters saved to {out_csv}")

    if args.plot:
        pqrst.plot_pqrst(filtered, args.fs, complexes)


def main():
    parser = argparse.ArgumentParser(description="ECG Monitor app entry point")
    parser.add_argument("csv", nargs="?", default=None,
                         help="CSV path (only used by --mode gui or --mode cli)")
    parser.add_argument("--mode", choices=["web", "live", "gui", "cli"], default="web",
                         help="Which interface to launch (default: web)")
    parser.add_argument("--source", choices=["simulate", "real"], default="simulate",
                         help="Data source for web/live modes (default: simulate)")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling rate in Hz (default: 250)")
    parser.add_argument("--window", type=float, default=10.0, help="Live rolling window, seconds (default: 10)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)),
                         help="Web dashboard port (default: 5000)")
    parser.add_argument("--powerline", type=float, default=50.0,
                         help="Powerline freq for --mode cli: 50 or 60 Hz (default: 50)")
    parser.add_argument("--plot", action="store_true", help="For --mode cli: also save a PQRST annotated plot")
    args = parser.parse_args()

    {
        "web": run_web,
        "live": run_live,
        "gui": run_gui,
        "cli": run_cli,
    }[args.mode](args)


if __name__ == "__main__":
    main()
