"""
Where samples come from.

One interface, three implementations, so the GUI never needs to know whether
it is replaying a 1975 tape recording or reading a BioAmp on someone's chest:

    DatasetSource   replay a held-out MIT-BIH record at wall-clock speed
    SerialSource    live BioAmp EXG Pill via the UNO Q USB serial bridge
    MotionGate      wraps a source and rejects motion-corrupted windows

Every source yields samples in wall-clock time, so playback speed and live
capture behave the same way from the GUI's point of view.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from . import config as C
from . import data as D


class SignalSource:
    """Interface. read() returns however many samples have become available
    since the last call - possibly an empty array."""

    fs = C.FS
    name = "source"

    def start(self):
        pass

    def read(self):
        raise NotImplementedError

    def prefetch(self, n):
        """
        Return the next `n` samples immediately, ignoring wall-clock pacing.

        Recorded sources can do this, so the GUI can push the rhythm-feature
        warmup through before the user sees anything. A live sensor cannot -
        the future has not happened yet - so it returns nothing and the GUI
        shows a warmup countdown instead.
        """
        return np.zeros(0)

    def stop(self):
        pass

    @property
    def status(self):
        return "ready"


# --------------------------------------------------------------------------
class DatasetSource(SignalSource):
    """
    Replays one MIT-BIH record in real time.

    `loop=True` restarts at the end so a demo can be left running.
    Reports the annotated truth alongside, purely so the GUI can show
    "annotated: LBBB" next to "predicted: LBBB" - the truth is NEVER fed to
    the model.
    """

    def __init__(self, record, speed=1.0, loop=True, start_s=0.0):
        self.record = str(record)
        self.speed = speed
        self.loop = loop

        import wfdb          # dataset replay only; not needed on the board
        header = wfdb.rdheader(str(C.DATA_DIR / self.record))
        ch = D.find_lead_channel(header)
        if ch is None:
            raise ValueError(f"record {self.record} has no {C.TARGET_LEAD} "
                             f"lead (channels: {header.sig_name})")
        rec = wfdb.rdrecord(str(C.DATA_DIR / self.record))
        self.signal = rec.p_signal[:, ch].astype(np.float64)
        self.fs = int(rec.fs)
        self.lead = header.sig_name[ch]

        self.truth = {b["sample"]: b for b in D.label_beats(self.record)}

        self.pos = int(start_s * self.fs)
        self._t0 = None
        self.name = f"MIT-BIH {self.record}"

    def start(self):
        self._t0 = time.perf_counter()
        self._start_pos = self.pos

    def read(self):
        if self._t0 is None:
            self.start()
        elapsed = (time.perf_counter() - self._t0) * self.speed
        target = self._start_pos + int(elapsed * self.fs)
        target = min(target, len(self.signal))

        if target <= self.pos:
            return np.zeros(0)

        out = self.signal[self.pos:target]
        self.pos = target

        if self.pos >= len(self.signal):
            if self.loop:
                self.pos = 0
                self._start_pos = 0
                self._t0 = time.perf_counter()
            else:
                self._t0 = None
        return out

    def prefetch(self, n):
        """Hand over the next n samples at once and advance the playhead, so
        the caller can prime the analyser before real-time playback begins."""
        n = int(min(n, len(self.signal) - self.pos))
        if n <= 0:
            return np.zeros(0)
        out = self.signal[self.pos:self.pos + n]
        self.pos += n
        self._start_pos = self.pos
        self._t0 = time.perf_counter()
        return out

    def truth_near(self, sample, tol=None):
        """Annotated class within `tol` samples of `sample`, or None."""
        tol = tol or int(0.075 * self.fs)
        for s in range(sample - tol, sample + tol + 1):
            b = self.truth.get(s)
            if b is not None:
                return b["cls"]
        return None

    @property
    def duration_s(self):
        return len(self.signal) / self.fs

    @property
    def status(self):
        return (f"replay {self.record} lead {self.lead} @ {self.fs} Hz "
                f"x{self.speed:g}")


# --------------------------------------------------------------------------
class SerialSource(SignalSource):
    """
    Live BioAmp EXG Pill through the UNO Q USB serial bridge.

    The QRB2210 Linux side runs the teammate's bridge, which forwards samples
    from the STM32U585 that owns the analog front end. This class only has to
    turn a stream of text lines into floats.

    Line format is deliberately forgiving because the exact framing on the
    board is not pinned down yet: any line whose first whitespace/comma
    separated token parses as a number is treated as one sample, and
    everything else (banners, JSON status, blank lines) is ignored. When the
    real bridge output is known, tighten `parse_line` and nothing else needs
    to change.
    """

    def __init__(self, port, baud=115200, fs=125, timeout=0.05,
                 adc_bits=None, vref=None):
        self.port = port
        self.baud = baud
        self.fs = fs
        self.timeout = timeout
        # If the bridge sends raw ADC counts rather than volts, set these and
        # samples get scaled; the model z-normalises anyway, so this only
        # affects what the axis says.
        self.adc_bits = adc_bits
        self.vref = vref

        self._ser = None
        self._partial = ""
        self._error = None

    def start(self):
        try:
            import serial  # pyserial
        except ImportError as e:
            raise RuntimeError(
                "pyserial is not installed. Run:  pip install pyserial"
            ) from e
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        self._partial = ""
        self._error = None

    @staticmethod
    def parse_line(line):
        """Return a float sample, or None if the line is not a sample."""
        line = line.strip()
        if not line:
            return None
        token = line.replace(",", " ").split()[0]
        try:
            return float(token)
        except ValueError:
            return None

    def read(self):
        if self._ser is None:
            return np.zeros(0)
        try:
            chunk = self._ser.read(4096).decode("utf-8", errors="ignore")
        except Exception as e:                      # cable yanked mid-demo
            self._error = str(e)
            return np.zeros(0)

        if not chunk:
            return np.zeros(0)

        self._partial += chunk
        *lines, self._partial = self._partial.split("\n")

        out = []
        for ln in lines:
            v = self.parse_line(ln)
            if v is not None:
                out.append(v)

        arr = np.asarray(out, dtype=np.float64)
        if len(arr) and self.adc_bits and self.vref:
            arr = arr * (self.vref / (2 ** self.adc_bits))
        return arr

    def stop(self):
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None

    @property
    def status(self):
        if self._error:
            return f"serial error: {self._error}"
        if self._ser is None:
            return "not connected"
        return f"live {self.port} @ {self.baud} baud, {self.fs} Hz"


def list_serial_ports():
    """Ports the GUI can offer in its dropdown. Empty if pyserial is absent."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [p.device for p in list_ports.comports()]


# --------------------------------------------------------------------------
class MotionGate:
    """
    Rejects windows corrupted by movement, using an accelerometer channel.

    NOT yet wired to real hardware - the MPU6050 data path on the UNO Q is
    still undecided. It is built now, against an injectable motion signal, so
    the logic is testable and demoable, and so that connecting the real IMU
    later is a matter of calling `push_motion()` from whatever ends up
    producing the data.

    Scope note, from the project brief: motion gating exists to stop
    classifying GARBAGE, and to let heart-rate be read in context ("130 bpm
    is fine while walking"). It is NOT an activity-aware arrhythmia
    classifier - MIT-BIH has no accelerometer data, so no such claim can be
    supported.
    """

    def __init__(self, threshold_g=0.25, hold_s=2.0, fs=C.FS):
        self.threshold_g = threshold_g
        self.hold_s = hold_s
        self.fs = fs
        self._recent = deque(maxlen=64)
        self._blocked_until = 0.0
        self.last_magnitude = 0.0

    def push_motion(self, accel_xyz, now=None):
        """
        accel_xyz : (ax, ay, az) in g. Feed this from the MPU6050 once its
        data path exists - from the STM32 stream or straight off I2C.
        """
        now = now if now is not None else time.perf_counter()
        ax, ay, az = accel_xyz
        # Deviation from 1g resting gravity: what is left is real movement.
        mag = abs(float(np.sqrt(ax * ax + ay * ay + az * az)) - 1.0)
        self.last_magnitude = mag
        self._recent.append(mag)
        if mag > self.threshold_g:
            self._blocked_until = now + self.hold_s
        return mag

    def is_blocked(self, now=None):
        now = now if now is not None else time.perf_counter()
        return now < self._blocked_until

    @property
    def status(self):
        if not self._recent:
            return "no IMU data (motion gating idle)"
        state = "REJECTING" if self.is_blocked() else "clean"
        return f"motion {self.last_magnitude:.2f} g - {state}"
