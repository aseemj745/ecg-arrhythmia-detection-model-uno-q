"""
Central configuration. Every locked project decision lives here so that the
scripts read like a recipe and there is exactly one place to change a choice.
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
# Signal / windowing  (locked)
# --------------------------------------------------------------------------
FS = 360                      # MIT-BIH native sampling rate
BEFORE = int(0.30 * FS)       # 108 samples before R-peak
AFTER = int(0.50 * FS)        # 180 samples after R-peak
WINDOW = BEFORE + AFTER       # 288 samples per beat

# Lead selection is BY NAME, never by channel index. Record 114 puts MLII in
# channel 1; 102/104 have no MLII at all (they are paced records we exclude).
TARGET_LEAD = "MLII"

# Bandpass applied to the whole record before beat extraction.
# Not in the original brief, but the BioAmp EXG Pill has a ~0.5-48 Hz analog
# passband of its own, so filtering MIT-BIH the same way narrows the gap
# between training data and the deployment sensor. Set BANDPASS = None to
# disable and retrain if you want to compare.
BANDPASS = (0.5, 40.0)        # Hz, 4th-order Butterworth, zero-phase
BANDPASS_ORDER = 4

# Flip each beat so its dominant QRS deflection points UP.
# MIT-BIH records the same condition in opposite polarities depending on
# electrode placement - LBBB in record 207 is inverted relative to 109/111,
# which alone was enough to make held-out LBBB unlearnable. Normalising
# polarity removes a nuisance dimension the tiny LBBB patient count cannot
# cover, and it is equally valid on the deployment side: swapping two BioAmp
# electrodes inverts the trace in exactly the same way.
POLARITY_NORMALISE = True
POLARITY_WINDOW_MS = 45       # search +/- this around the R-peak for the
                              # dominant deflection

# --------------------------------------------------------------------------
# Classes  (locked)
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

# Beat annotation symbols that mark an actual heartbeat (vs a rhythm or a
# signal-quality note). Used to build the RR series.
BEAT_SYMBOLS = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
    'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?',
}

# Locked labelling rule: a beat inside an "(AFIB" rhythm annotation is AFIB
# no matter what its own symbol says, because AFib is a rhythm phenomenon and
# most of its beats carry the symbol 'N'. Flip to False to instead let
# L/R/V morphology symbols win over the rhythm annotation.
AFIB_OVERRIDES_ALL_SYMBOLS = True

# --------------------------------------------------------------------------
# Records  (locked)
# --------------------------------------------------------------------------
# Paced beats change QRS shape completely and "paced" is not one of our
# classes. Standard exclusion in the ECG literature.
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
# NEVER a random beat-level split: beats from one patient in both train and
# test would let the model memorise that patient's heartbeat instead of
# learning the class, and the score would be a lie.
#
# The split is constrained by how few patients carry the rare classes, so it
# was built rare-class-first:
#   LBBB exists in only 4 patients (109, 111, 207, 214) -> 214 test,
#        109 val, 111+207 train. Test LBBB therefore rests on a SINGLE
#        patient. That is a real limitation, stated rather than hidden.
#   RBBB exists in 6 (118, 124, 207, 212, 231, 232) -> 231+232 test,
#        118 val, 124+207+212 train.
#   AFIB exists in 7 (201, 202, 203, 210, 219, 221, 222) -> 202+221 test,
#        219 val, 201+203+210+222 train.
#   NOR and PVC are spread over 30+ patients, so they fill in freely.
#
# Record 114 is deliberately in TEST: it is the one record whose MLII sits in
# channel 1, so if lead-by-name selection ever regressed to lead-by-index,
# the test score would visibly break instead of quietly rotting.

# Held out from training AND validation. Source of the final numbers and of
# the "unseen patient" clips played in the GUI demo.
TEST_RECORDS = ['114', '119', '202', '214', '221', '231', '232', '233', '234']

# Model selection / early stopping only. Also unseen by the optimiser.
VAL_RECORDS = ['109', '118', '205', '208', '219']

TRAIN_RECORDS = [r for r in USABLE_RECORDS
                 if r not in TEST_RECORDS and r not in VAL_RECORDS]

# --------------------------------------------------------------------------
# GUI demo clips - all drawn from TEST_RECORDS, i.e. patients the model has
# never seen. The top pane of the GUI plays the healthy record; the bottom
# pane plays one of the arrhythmia records.
# --------------------------------------------------------------------------
# fold is either 'test' (never seen at all) or 'val' (never trained on, but
# used to pick the checkpoint). Both are legitimate demo material; the GUI
# labels which is which so a viewer is never misled about how unseen a
# recording really is.
# NOTHING IN THESE TWO LISTS MAY COME FROM TRAIN_RECORDS.
# step1b_check_split.py enforces this and fails the build if it is violated.
# Showing the model a recording it was fitted on and calling that a
# demonstration is the exam-with-the-answer-sheet problem; the numbers would
# be meaningless and a reviewer who checked would be right to discount the
# whole result.
DEMO_ONLY_TEST_RECORDS = True     # enforced by step1b_check_split.py

DEMO_HEALTHY = [
    # record, fold, note
    ('234', 'test', '2689 normal vs 3 ectopic - essentially clean'),
    ('114', 'test', '97.7% normal; also the MLII-in-channel-1 record'),
    # Validation records - never trained on, but they did influence which
    # epoch's checkpoint was kept. Excluded from the demo so that every
    # waveform on camera is provably, completely unseen. Re-add them (and set
    # DEMO_ONLY_TEST_RECORDS = False) if you want more variety and are willing
    # to explain the distinction:
    #   ('205', 'val', '97.3% normal, faster sinus rhythm'),
]
DEMO_HEALTHY_RECORD = DEMO_HEALTHY[0][0]

DEMO_ARRHYTHMIA = [
    # record, headline class, fold, note
    ('214', 'LBBB', 'test', '1992 LBBB + 255 PVC'),
    ('231', 'RBBB', 'test', '1242 RBBB, sustained throughout'),
    ('232', 'RBBB', 'test', 'RBBB plus many atrial beats we do not model'),
    ('119', 'PVC',  'test', '442 PVC in bigeminy - clean, regular pattern'),
    ('233', 'PVC',  'test', '826 PVC, 28% burden - the hard case'),
    ('221', 'AFIB', 'test', '2346 AFib beats, sustained'),
    ('202', 'AFIB', 'test', 'AFib episodes alternating with normal rhythm'),
    # Validation records, excluded for the same reason as above:
    #   ('109', 'LBBB', 'val', '2480 LBBB, textbook morphology'),
    #   ('118', 'RBBB', 'val', '2155 RBBB'),
    #   ('208', 'PVC',  'val', '986 PVC, frequent and irregular'),
    #   ('219', 'AFIB', 'val', '1791 AFib + normal stretches'),
]
DEMO_ARRHYTHMIA_RECORDS = {cls: rec for rec, cls, fold, _ in DEMO_ARRHYTHMIA
                           if fold == 'test'}

# Seconds of signal fed through the analyser BEFORE the display starts, so a
# replay is already past the rhythm-feature warmup when the user sees it.
# Live capture cannot do this - the warmup there is real and is shown as a
# countdown instead.
DEMO_PRIME_S = 20.0

# --------------------------------------------------------------------------
# Numeric (rhythm) features  (locked: multi-input fusion model)
# --------------------------------------------------------------------------
# Every feature is CAUSAL - current and previous beats only, never a future
# one - so this exact code runs in real time on the UNO Q from a 10-entry RR
# ring buffer, with no need to wait for the next beat to arrive.
#
# The first three describe THIS beat's timing (what a PVC disturbs); the last
# five describe the IRREGULARITY of the last 10 beats (what AFib is). Three
# per-beat features alone gave AFIB F1 0.46, because an AFib beat in
# isolation genuinely looks and times like a normal one - irregularity only
# exists across a window. This is still one fused model, as decided; only the
# feature vector grew.
FEATURE_NAMES = [
    "rr_prev",         # this RR interval, seconds
    "rr_delta",        # change from the previous RR interval
    "rr_ratio_local",  # this RR / mean of previous 10, scale-free
    "rmssd",           # RMS of successive RR differences over the window
    "cv_rr",           # std/mean of the window - relative dispersion
    "pnn50",           # fraction of successive diffs above 50 ms
    "rr_range",        # (max-min)/mean of the window
    "mad_median",      # mean abs deviation from the window median / median
    "rmssd_clean",     # rmssd after ectopic intervals are median-replaced
    "cv_rr_clean",     # cv    after ectopic intervals are median-replaced
]
N_FEATURES = len(FEATURE_NAMES)
RR_LOCAL_WINDOW = 10          # beats in the irregularity window
RR_CLIP = (0.20, 3.00)        # seconds; anything outside is annotation noise
ECTOPIC_TOL = 0.25            # an RR more than 25% off the window median is
                              # treated as ectopic by the *_clean features

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

# Class-weight exponent for the loss. 1.0 = full inverse frequency, which
# over-corrects here: it made the model fire AFIB at 2,000+ normal beats to
# buy recall. 0.5 (inverse square root) keeps the rare classes trainable
# without wrecking NOR precision.
CLASS_WEIGHT_POWER = 0.5

# Weight-EMA decay. ~4 epochs of effective averaging at 243 steps/epoch.
EMA_DECAY = 0.999

# --------------------------------------------------------------------------
# Episode detection (GUI highlighting + Excel logging)
# --------------------------------------------------------------------------
# How many CONSECUTIVE beats of one class before it counts as a finding.
# This is per-class on purpose. A PVC is a single ectopic beat - demanding a
# run of three would mean never reporting one, which is exactly the bug that
# made record 119's 17 PVCs vanish from the episode list. Sustained rhythms
# are the opposite: one stray LBBB call inside normal sinus is a
# misclassification, three in a row is a finding.
MIN_EPISODE_BEATS = {
    "PVC": 1,
    "LBBB": 3,
    "RBBB": 3,
    "AFIB": 3,
}
DEFAULT_MIN_EPISODE_BEATS = 3

# Beats below this confidence are shown greyed in the GUI and marked
# "low confidence" in the Excel log rather than being silently trusted.
LOW_CONFIDENCE = 0.50

# Minimum mean confidence before an episode is FLAGGED and logged.
#
# This is an alerting operating point, not a change to the model, and the
# thresholds were measured on the VALIDATION patients (never on test).
# They are per-class because the model turns out to be badly calibrated in
# class-specific ways - a single global threshold would be actively harmful:
#
#   PVC   gating works: precision 0.272 -> 0.594 at 0.50, recall still 0.834.
#         PVC is the over-fired class, so this is where a gate earns its
#         keep and it is the reason the log is not drowned in false PVCs.
#   LBBB  gating is destructive: nearly every correct LBBB call sits below
#         0.45 (recall 0.675 -> 0.025). Confident-but-low-probability.
#   AFIB  gating is BACKWARDS: precision 0.564 -> 0.409 as the threshold
#         rises, because the most confident AFIB calls come from extreme RR
#         irregularity, which is exactly what a PVC-heavy normal record
#         looks like.
#   RBBB  already 0.996 precision ungated; a gate only costs recall.
MIN_EPISODE_CONFIDENCE = {
    "PVC": 0.50,
    "LBBB": 0.0,
    "RBBB": 0.0,
    "AFIB": 0.0,
}

# A sustained arrhythmia is chopped into chunks of at most this many seconds.
# Without it, a patient who is in RBBB or AFib for the whole recording
# produces one episode that never ends, so it is never finalised, never
# logged and never shaded - the single most important case silently produces
# nothing. Chunking also keeps the Excel log to a readable number of rows.
MAX_EPISODE_S = 30.0
