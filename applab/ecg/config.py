"""
All the fixed project decisions in one place, so the scripts stay readable and
there is only one spot to change any choice.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mitdb"
ARTIFACTS = ROOT / "artifacts"
DATASET_DIR = ARTIFACTS / "dataset"
MODEL_DIR = ARTIFACTS / "models"
REPORT_DIR = ARTIFACTS / "reports"

for _d in (DATASET_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Signal / windowing
# --------------------------------------------------------------------------
FS = 360                      # MIT-BIH native sampling rate
BEFORE = int(0.30 * FS)       # 108 samples before R-peak
AFTER = int(0.50 * FS)        # 180 samples after R-peak
WINDOW = BEFORE + AFTER       # 288 samples per beat

# Pick the lead by name, not by channel index. Record 114 has MLII in channel 1.
TARGET_LEAD = "MLII"

# Matched to the BioAmp EXG Pill's own ~0.5-48 Hz analog passband, so the
# training data looks like what the sensor gives us. Set to None and retrain
# to compare.
BANDPASS = (0.5, 40.0)        # Hz, 4th-order Butterworth, zero-phase
BANDPASS_ORDER = 4

# Flip each beat so the dominant QRS deflection points up. MIT-BIH has the same
# condition in both polarities (LBBB in 207 is inverted vs 109/111), which on
# its own made held-out LBBB unlearnable.
POLARITY_NORMALISE = True
POLARITY_WINDOW_MS = 45       # search +/- this around the R-peak

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------
CLASSES = ["NOR", "LBBB", "RBBB", "PVC", "AFIB"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

CLASS_DESCRIPTIONS = {
    "NOR": "Normal sinus beat",
    "LBBB": "Left bundle branch block",
    "RBBB": "Right bundle branch block",
    "PVC": "Premature ventricular contraction",
    "AFIB": "Beat during atrial fibrillation",
}

# Symbols that mark an actual heartbeat, as opposed to a rhythm or signal
# quality note. Used to build the RR series.
BEAT_SYMBOLS = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
    'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?',
}

# A beat inside an "(AFIB" rhythm annotation is AFIB whatever its own symbol
# says, because most AFib beats are marked 'N'. False lets the L/R/V morphology
# symbols win instead.
AFIB_OVERRIDES_ALL_SYMBOLS = True

# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
# Paced records. Pacing changes QRS shape completely and paced is not one of
# our classes. Standard exclusion in the literature.
EXCLUDED_RECORDS = ["102", "104", "107", "217"]

ALL_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221', '222', '223',
    '228', '230', '231', '232', '233', '234',
]

USABLE_RECORDS = [r for r in ALL_RECORDS if r not in EXCLUDED_RECORDS]

# --------------------------------------------------------------------------
# Patient-level split  (run step1b_check_split.py to see the resulting table)
# --------------------------------------------------------------------------
# Never a random beat-level split. The same patient in both train and test lets
# the model memorise that person's heartbeat instead of learning the class.
#
# Built rare-class-first, since so few patients carry the rare classes:
#   LBBB is in only 4 patients (109, 111, 207, 214) -> 214 test, 109 val,
#        111+207 train. So test LBBB rests on one patient.
#   RBBB is in 6 (118, 124, 207, 212, 231, 232) -> 231+232 test, 118 val.
#   AFIB is in 7 (201, 202, 203, 210, 219, 221, 222) -> 202+221 test, 219 val.
#   NOR and PVC are spread over 30+ patients so they fill in freely.
#
# 114 is in test on purpose: it is the one record with MLII in channel 1, so a
# regression back to lead-by-index would break the test score visibly.
TEST_RECORDS = ['114', '119', '202', '214', '221', '231', '232', '233', '234']

# Checkpoint selection only. Never seen by the optimiser.
VAL_RECORDS = ['109', '118', '205', '208', '219']

TRAIN_RECORDS = [r for r in USABLE_RECORDS
                 if r not in TEST_RECORDS and r not in VAL_RECORDS]

# --------------------------------------------------------------------------
# GUI demo clips
# --------------------------------------------------------------------------
# Nothing in these lists may come from TRAIN_RECORDS. step1b_check_split.py
# enforces that and fails the build otherwise. Demoing a record the model was
# fitted on would make the whole thing meaningless.
DEMO_ONLY_TEST_RECORDS = True

DEMO_HEALTHY = [
    # record, fold, note
    ('234', 'test', '2689 normal vs 3 ectopic - essentially clean'),
    ('114', 'test', '97.7% normal; also the MLII-in-channel-1 record'),
    # Val records are never trained on, but they did pick the checkpoint, so
    # they are left out to keep every demo waveform completely unseen. Re-add
    # them (and set DEMO_ONLY_TEST_RECORDS = False) for more variety:
    #   ('205', 'val', '97.3% normal, faster sinus rhythm'),
]
DEMO_HEALTHY_RECORD = DEMO_HEALTHY[0][0]

DEMO_ARRHYTHMIA = [
    # record, headline class, fold, note
    ('214', 'LBBB', 'test', '1992 LBBB + 255 PVC'),
    ('231', 'RBBB', 'test', '1242 RBBB, sustained throughout'),
    # 232 dropped from the demo list: most of its beats are atrial ('A'/'a'),
    # a class we don't have, so the dashboard just labels them as whichever of
    # the five they resemble. Still in TEST_RECORDS, so it still counts in the
    # reported metrics.
    ('119', 'PVC',  'test', '442 PVC in bigeminy - clean, regular pattern'),
    ('233', 'PVC',  'test', '826 PVC, 28% burden - the hard case'),
    ('221', 'AFIB', 'test', '2346 AFib beats, sustained'),
    ('202', 'AFIB', 'test', 'AFib episodes alternating with normal rhythm'),
    # Val records, left out for the same reason as above:
    #   ('109', 'LBBB', 'val', '2480 LBBB, textbook morphology'),
    #   ('118', 'RBBB', 'val', '2155 RBBB'),
    #   ('208', 'PVC',  'val', '986 PVC, frequent and irregular'),
    #   ('219', 'AFIB', 'val', '1791 AFib + normal stretches'),
]
DEMO_ARRHYTHMIA_RECORDS = {cls: rec for rec, cls, fold, _ in DEMO_ARRHYTHMIA
                           if fold == 'test'}

# Seconds fed through the analyser before the display starts, so a replay is
# already past the rhythm-feature warmup when the user sees it. Live capture
# can't do this, so there the warmup is shown as a countdown instead.
DEMO_PRIME_S = 20.0

# --------------------------------------------------------------------------
# Numeric (rhythm) features
# --------------------------------------------------------------------------
# All causal: current and previous beats only, never a future one, so the same
# code runs live on the UNO Q off a 10-entry ring buffer.
#
# First three are this beat's timing (what a PVC disturbs), the rest describe
# irregularity over the last 10 beats (what AFib is). Per-beat features alone
# gave AFIB F1 0.46, since an AFib beat in isolation looks normal.
FEATURE_NAMES = [
    "rr_prev",         # this RR interval, seconds
    "rr_delta",        # change from the previous RR interval
    "rr_ratio_local",  # this RR / mean of previous 10, scale-free
    "rmssd",           # RMS of successive RR differences over the window
    "cv_rr",           # std/mean of the window
    "pnn50",           # fraction of successive diffs above 50 ms
    "rr_range",        # (max-min)/mean of the window
    "mad_median",      # mean abs deviation from window median / median
    "rmssd_clean",     # rmssd after ectopic intervals are median-replaced
    "cv_rr_clean",     # cv    after ectopic intervals are median-replaced
]
N_FEATURES = len(FEATURE_NAMES)
RR_LOCAL_WINDOW = 10          # beats in the irregularity window
RR_CLIP = (0.20, 3.00)        # seconds; outside this is annotation noise
ECTOPIC_TOL = 0.25            # RR more than 25% off the median counts as ectopic

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
SEED = 1337
BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
AUGMENT = True                # see BeatDataset in step2_train.py

# Class-weight exponent for the loss. Dropped from 1.0 (full inverse frequency)
# to 0.5, because at 1.0 the model fired AFIB at 2000+ normal beats to buy
# recall and wrecked NOR precision.
CLASS_WEIGHT_POWER = 0.5

# Weight-EMA decay. About 4 epochs of averaging at 243 steps/epoch.
EMA_DECAY = 0.999

# --------------------------------------------------------------------------
# Episode detection (GUI highlighting + Excel logging)
# --------------------------------------------------------------------------
# Consecutive beats of one class needed before it counts as a finding.
# Per-class on purpose: a PVC is a single ectopic beat, so requiring three
# meant never reporting one (record 119's 17 PVCs vanished from the list).
# Sustained rhythms are the other way round, one stray LBBB call inside normal
# sinus is just a misclassification.
MIN_EPISODE_BEATS = {
    "PVC": 1,
    "LBBB": 3,
    "RBBB": 3,
    "AFIB": 3,
}
DEFAULT_MIN_EPISODE_BEATS = 3

# Beats below this are greyed in the GUI and marked low confidence in the log.
LOW_CONFIDENCE = 0.50

# Minimum mean confidence before an episode is logged. Measured on the
# validation patients, never on test, and per-class because the model is
# badly calibrated in class-specific ways:
#   PVC   gating helps: precision 0.272 -> 0.594 at 0.50, recall still 0.834
#   LBBB  gating destroys it: recall 0.675 -> 0.025, most correct calls are
#         below 0.45
#   AFIB  gating runs backwards: precision 0.564 -> 0.409 as the threshold
#         rises, since the most confident AFIB calls come from extreme RR
#         irregularity, which is what a PVC-heavy record looks like
#   RBBB  already 0.996 precision ungated, a gate only costs recall
MIN_EPISODE_CONFIDENCE = {
    "PVC": 0.50,
    "LBBB": 0.0,
    "RBBB": 0.0,
    "AFIB": 0.0,
}

# One stricter gate for every class when the input is a live sensor. The map
# above was measured on MIT-BIH, where mean confidence is 0.71; on live BioAmp
# input it's about 0.40 against a 0.20 chance level, because real electrode
# noise makes beat morphology ~2.5x more variable (std 0.28 vs 0.11). Ungated,
# a healthy resting subject produced 5 false AFIB/PVC episodes.
#
# Measured on a 23.6s BioAmp capture vs record 119 (24 real PVC episodes):
#   gate 0.00 -> 5 false live episodes, 24/24 PVC kept
#   gate 0.55 -> 0 false live episodes, 24/24 PVC kept
#   gate 0.65 -> 0 false live episodes, 23/24 PVC kept
# So 0.55. This doesn't make the model better on noisy input, it just makes it
# decline to guess.
LIVE_MIN_EPISODE_CONFIDENCE = 0.55

# Cap on episode length. Without it a patient who is in RBBB or AFib for the
# whole recording produces one episode that never ends, so it never gets
# finalised, logged or shaded.
MAX_EPISODE_S = 30.0

# Same thing for the on-device demo replay, but shorter. At 30s a 90s replay of
# a sustained rhythm gives about three rows, the first only after 30s of
# watching, which reads as "RBBB isn't logging". 10s gives ~9 rows and the
# first within ten seconds. Display granularity only, it doesn't change which
# beats are detected.
DEMO_MAX_EPISODE_S = 10.0
