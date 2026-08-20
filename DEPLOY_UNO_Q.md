# Running the model on the Arduino UNO Q

Written for someone who has not deployed to this board before. Every step
says what to type, what you should see, and what to do when you do not see
it.

Some specifics (exact device paths, the App Lab menu wording, the default
username) depend on your board image and are flagged with **VERIFY** — check
them on your actual hardware rather than trusting this document.

> **Known issue — sample rate.** Every `--fs 250` / "250 Hz" reference below
> assumes the MCU sketch samples at 250 Hz. That was true for two of the
> reference sketches used during development, but the most advanced one
> actually samples at **125 Hz** (check its `FS_HZ` constant — don't trust
> comments, they're stale). **Confirm the real rate from your own sketch
> before trusting any `--fs` value here** — a wrong rate doesn't crash
> anything, it just silently scales every heart rate and RR interval.

---

## 0. What is going where, and why

The UNO Q has two processors:

| | chip | runs | your job |
|---|---|---|---|
| **Linux side** | Qualcomm Dragonwing QRB2210, 4× Cortex-A53 | Debian Linux | **the ONNX model — this is what we deploy** |
| **MCU side** | STM32U585, Cortex-M33 | bare metal / Arduino sketch | analog front end, samples the BioAmp |

You are deploying to the **Linux side**. It is an ordinary ARM64 Debian
machine: `python3`, `pip`, `ssh`, and `systemd` all behave normally.

The deployment bundle is **92 KB** and needs numpy, scipy and onnxruntime.
No PyTorch, no matplotlib, no wfdb, no dataset.

**Performance is already a non-issue.** A full analysis pass over a 20-second
buffer measures 2.68 ms on the dev PC and runs once every 500 ms — a 0.5%
duty cycle. Even assuming the A53 is 20× slower, that is ~11% of one core out
of four. The constraints on this board are RAM and dependencies, not compute,
which is why the work below is about packaging rather than optimisation.

---

## Route A — Arduino App Lab (recommended, matches your workflow)

App Lab imports apps as `.zip`. Build one with:

```bash
python package_applab.py
```

That writes `artifacts/ecg_arrhythmia_unoq.zip` (~154 KB), containing the
INT8 model, the `ecg` package, `python/main.py`, and a **90-second recording
of held-out MIT-BIH record 119 resampled to 250 Hz** — the same rate the MCU
sketch samples A0 at. So the app can prove itself with no sensor and no
electrodes attached.

1. App Lab → **Create new app +** → **Import an App** → **Import from
   computer** → pick the zip.
2. Open the app and run the self-test:

   ```bash
   python3 python/main.py
   ```

3. Expect roughly 15–25 **PVC-only** episodes and a `SELF-TEST PASSED` line.
   Record 119 is a bigeminy patient the model has never seen, so PVCs are
   the correct answer and any AFIB/LBBB/RBBB would be a red flag.

**The manifest (`app.yaml`) is an unverified guess** — App Lab apps live on
the board, not on the PC, so its exact 0.8.0 schema could not be checked. If
the import is rejected, do not fight it; run
`python package_applab.py --show-manifest-help` and dump the manifest of a
working app instead. Route B below does not depend on the manifest at all.

## Route B — plain files over the terminal (always works)

Use this if the import is rejected, or to check quickly. Everything in the
zip is ordinary Python; App Lab is not required to run it.

## 1. Build the bundle (on your PC)

```bash
python package_for_unoq.py
```

You get `artifacts/unoq_deploy.tar.gz` (~92 KB) containing the INT8 model,
the `ecg` package, and `deploy/uno_q_monitor.py`.

---

## 2. Get a shell on the board

Three ways, easiest first.

**A — Arduino App Lab terminal.** Connect the board over USB-C, open App Lab,
and use its terminal/console panel for the Linux side. **VERIFY** the exact
menu name on your version.

**B — SSH over USB or Wi-Fi.** Once the board is on your network:

```bash
ssh <user>@<board-ip>
```

To find the IP, check your router, or on the board's own console run
`hostname -I`. **VERIFY** the default username for your image.

**C — Serial console.** The board appears as a serial device on your PC
(`COMx` on Windows). Connect with PuTTY at 115200 baud.

Confirm you are on the Linux side and not the MCU:

```bash
uname -a && python3 --version && nproc
```

You should see `aarch64`, Python 3.9+, and `4`.

---

## 3. Install the dependencies (on the board)

```bash
sudo apt update
sudo apt install -y python3-pip python3-numpy python3-scipy
pip3 install --break-system-packages onnxruntime pyserial
```

### Why `--break-system-packages` is needed here

The UNO Q image ships Python 3.13 on a Debian that enforces **PEP 668**, so a
plain `pip3 install` refuses with `externally-managed-environment`. The flag
sounds alarming; on this board it is the right call, for two reasons:

* `onnxruntime` and `pyserial` are **leaf** packages — nothing in Debian
  depends on them, so there is no system package for pip to shadow or
  downgrade. numpy and scipy, the two that *would* be risky to shadow, come
  from `apt` and are already present (numpy 2.2.4, scipy 1.15.3).
* App Lab runs `python/main.py` with the **system** interpreter. Anything
  installed into a venv is invisible to App Lab's Run button.

If you would rather not touch the system interpreter, use a venv that
**reuses** the apt-installed numpy and scipy — without
`--system-site-packages` pip would try to build scipy from source, which
takes hours on a Cortex-A53 and can exhaust the 2 GB board:

```bash
sudo apt install -y python3-venv
python3 -m venv --system-site-packages ~/ecgvenv
~/ecgvenv/bin/pip install onnxruntime pyserial
~/ecgvenv/bin/python python/main.py      # must use this python, not python3
```

Installing numpy and scipy through `apt` rather than `pip` is deliberate —
the Debian packages are prebuilt for ARM, whereas pip may try to compile
scipy from source, which takes a very long time on a Cortex-A53 and can run
the 2 GB board out of memory.

Check onnxruntime imported and picked up the ARM build:

```bash
python3 -c "import onnxruntime; print(onnxruntime.__version__, onnxruntime.get_available_providers())"
```

Expect a version number and `['CPUExecutionProvider']`.

> If `pip3 install onnxruntime` fails to find a wheel, your Python version is
> probably outside the range with prebuilt aarch64 wheels. Check
> `python3 --version` and try `pip3 install "onnxruntime==1.16.3"`. Do **not**
> try to build onnxruntime from source on this board.

---

## 4. Copy and unpack

From your PC:

```bash
scp artifacts/unoq_deploy.tar.gz <user>@<board-ip>:~/
```

On the board:

```bash
tar xzf unoq_deploy.tar.gz
cd ecg_unoq
```

---

## 5. Prove it works with no sensor attached

Do this before touching hardware — it separates "the model runs on the board"
from "the sensor is wired correctly", so when something breaks you know which
half to look at.

On your PC, create a test capture and copy it over:

```bash
python -c "import wfdb,numpy as np;from scipy.signal import resample_poly;from ecg import config as C,data as D;h=wfdb.rdheader(str(C.DATA_DIR/'119'));ch=D.find_lead_channel(h);s=wfdb.rdrecord(str(C.DATA_DIR/'119')).p_signal[:C.FS*120,ch];np.savetxt('capture.csv',resample_poly(s,25,36),delimiter=',',fmt='%.5f')"
```

On the board:

```bash
python3 deploy/uno_q_monitor.py --csv capture.csv --fs 250 --out detections.csv
```

You should see PVC detections scrolling past and a periodic
`# ... processed, N beats, M episodes, 1.00x real time` line. **If the
real-time factor is well below 1.00, the board cannot keep up** — that would
be the first genuine sign optimisation is needed. On record 119 expect
roughly 24 episodes over the 120 s capture.

---

## 6. Run it against the live sensor

Once the bridge is streaming, pick whichever matches your setup:

```bash
# the bridge exposes a serial device
python3 deploy/uno_q_monitor.py --serial /dev/ttyACM0 --fs 250 --out detections.csv

# the bridge prints samples to stdout
python3 /app/python/adc_reader_uno_q_bridge.py | python3 deploy/uno_q_monitor.py --stdin --fs 250
```

**VERIFY** two things against the real bridge:

1. **The device path.** `ls /dev/tty*` before and after the STM32 starts
   streaming; the new entry is yours. It may be `/dev/ttyACM0`, `/dev/ttyUSB0`
   or `/dev/ttySx`.
2. **The sample rate.** `--fs` must match what the bridge actually sends —
   your friend's dashboard showed 250 Hz. If this is wrong, everything still
   *runs*, but every heart rate and duration will be scaled by the ratio, and
   the rhythm features will be silently wrong. This is the single most
   likely thing to get wrong.

The parser accepts one number per line and ignores anything it cannot parse,
so banner text and status lines are harmless. If the bridge emits CSV with
several columns, the first column is used.

---

## 7. Start automatically on boot (optional, good for the demo)

```bash
sudo tee /etc/systemd/system/ecg-monitor.service > /dev/null <<'EOF'
[Unit]
Description=ECG arrhythmia monitor
After=multi-user.target

[Service]
Type=simple
User=%i
WorkingDirectory=/home/<user>/ecg_unoq
ExecStart=/usr/bin/python3 deploy/uno_q_monitor.py --serial /dev/ttyACM0 --fs 250 --out /home/<user>/detections.csv
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ecg-monitor
journalctl -u ecg-monitor -f      # watch detections live
```

Replace `<user>` with your actual username.

---

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: ecg` | run from the wrong directory | `cd ~/ecg_unoq` first |
| No detections ever | warmup, or wrong `--fs` | needs 12 beats of rhythm history; then check `--fs` |
| Detections but absurd heart rates | `--fs` does not match the bridge | set `--fs` to the real rate |
| `Permission denied: /dev/ttyACM0` | user not in `dialout` | `sudo usermod -aG dialout $USER`, then log out and back in |
| Real-time factor < 1.0 | board genuinely overloaded | `--threads 3`, or raise `reanalyse_s` in `StreamAnalyzer` |
| Killed / out of memory | 2 GB variant under pressure | close App Lab; check with `free -h` |

---

## 9. Numbers worth measuring on the board

For the submission, measure these **on the hardware** rather than quoting the
dev-PC figures:

```bash
python3 - <<'EOF'
import time, numpy as np, onnxruntime as ort
s = ort.InferenceSession("artifacts/models/model_int8.onnx",
                         providers=["CPUExecutionProvider"])
x = np.zeros((1,1,288), np.float32); f = np.zeros((1,10), np.float32)
for _ in range(20): s.run(["logits"], {"beat":x, "rr_features":f})
t=time.perf_counter()
for _ in range(200): s.run(["logits"], {"beat":x, "rr_features":f})
print(f"single-beat inference: {(time.perf_counter()-t)/200*1000:.3f} ms")
EOF
```

Also worth recording: `free -h` while running, and the real-time factor from
the monitor's own status lines.
