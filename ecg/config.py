"""Project settings in one place, so there's only one spot to change anything."""
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
# Patient-level split (step1b_check_split.py prints the table)
# --------------------------------------------------------------------------
# Split by patient, not by beat, so the model can't just memorise one person's
# heartbeat. Assigned the rare classes first: LBBB is in only 4 patients
# (109/111/207/214), so test LBBB rests on 214 alone. RBBB has 6, AFIB 7.
# 114 is in test because it's the odd record with MLII in channel 1.
TEST_RECORDS = ['114', '119', '202', '214', '221', '231', '232', '233', '234']

# Checkpoint selection only. Never seen by the optimiser.
VAL_RECORDS = ['109', '118', '205', '208', '219']

TRAIN_RECORDS = [r for r in USABLE_RECORDS
                 if r not in TEST_RECORDS and r not in VAL_RECORDS]

# --------------------------------------------------------------------------
# GUI demo clips
# --------------------------------------------------------------------------
# Test-fold records only. step1b_check_split.py fails the build if a training
# record ends up in here.
DEMO_ONLY_TEST_RECORDS = True

DEMO_HEALTHY = [
    # record, fold, note
    ('234', 'test', '2689 normal vs 3 ectopic - essentially clean'),
    ('114', 'test', '97.7% normal; also the MLII-in-channel-1 record'),
    # Val records picked the checkpoint, so they're left out. Re-add with
    # DEMO_ONLY_TEST_RECORDS = False if you want more variety:
    #   ('205', 'val', '97.3% normal, faster sinus rhythm'),
]
DEMO_HEALTHY_RECORD = DEMO_HEALTHY[0][0]

DEMO_ARRHYTHMIA = [
    # record, headline class, fold, note
    ('214', 'LBBB', 'test', '1992 LBBB + 255 PVC'),
    ('231', 'RBBB', 'test', '1242 RBBB, sustained throughout'),
    # 232 dropped from the demo list - mostly atrial beats ('A'/'a'), which we
    # don't model. Still in TEST_RECORDS, so it counts in the metrics.
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

# Fed through the analyser before the display starts so a replay skips the
# warmup. Live capture can't do that, it shows a countdown instead.
DEMO_PRIME_S = 20.0

# --------------------------------------------------------------------------
# Numeric (rhythm) features
# --------------------------------------------------------------------------
# Causal only - this beat and earlier ones, never a future one, so the same
# code runs live off a 10-entry ring buffer. First three are this beat's
# timing (PVC), the rest are irregularity over 10 beats (AFib). Without the
# window features AFIB F1 was 0.46.
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

# Dropped from 1.0 to 0.5. At 1.0 the model called AFIB on 2000+ normal beats
# chasing recall and wrecked NOR precision.
CLASS_WEIGHT_POWER = 0.5

# Weight-EMA decay. About 4 epochs of averaging at 243 steps/epoch.
EMA_DECAY = 0.999

# --------------------------------------------------------------------------
# Episode detection (GUI highlighting + Excel logging)
# --------------------------------------------------------------------------
# Consecutive beats needed before it counts as a finding. Per-class, because a
# PVC is a single beat - requiring 3 lost all 17 of record 119's PVCs. A lone
# LBBB call inside normal sinus is usually just a misclassification, so the
# sustained rhythms need 3.
MIN_EPISODE_BEATS = {
    "PVC": 1,
    "LBBB": 3,
    "RBBB": 3,
    "AFIB": 3,
}
DEFAULT_MIN_EPISODE_BEATS = 3

# Beats below this are greyed in the GUI and marked low confidence in the log.
LOW_CONFIDENCE = 0.50

# Minimum mean confidence before an episode is logged. Per-class, measured on
# the validation patients:
#   PVC   helps: precision 0.272 -> 0.594 at 0.50
#   LBBB  gating kills it: recall 0.675 -> 0.025
#   AFIB  goes backwards: precision 0.564 -> 0.409 as the gate rises
#   RBBB  already 0.996 precision, gating only costs recall
MIN_EPISODE_CONFIDENCE = {
    "PVC": 0.50,
    "LBBB": 0.0,
    "RBBB": 0.0,
    "AFIB": 0.0,
}

# Stricter gate for live sensor input. The map above came from MIT-BIH, where
# mean confidence is 0.71; on live BioAmp it's around 0.40 against a 0.20
# chance level, and ungated a healthy subject threw 5 false AFIB/PVC episodes.
# Tested on a 23.6s capture: 0.55 dropped those to 0 and still kept all 24
# real PVCs from record 119. 0.65 started losing real ones.
LIVE_MIN_EPISODE_CONFIDENCE = 0.55

# Cap on episode length. Without it a patient in AFib for the whole recording
# gives one episode that never ends, so it never gets logged.
MAX_EPISODE_S = 30.0

# Shorter for the on-device demo. At 30s a 90s replay only gives ~3 rows and
# the first takes 30s to appear, which looks like nothing is logging. Display
# granularity only, doesn't change what gets detected.
DEMO_MAX_EPISODE_S = 10.0
