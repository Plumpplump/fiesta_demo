# Author: David Li, SISHK
"""Tunable constants for the uniform/glasses demo.

Everything a teacher would want to adjust during the STEM fair lives here, so
nothing has to be dug out of the controllers.

Numbers marked "measured" come from the sibling project
https://github.com/yunguangli/schooluniform (same YOLO11n + MobileCLIP2-B +
YuNet stack, evaluated on public portraits and real crops).
"""

# --- Model files (in src/assets/models) -----------------------------------
YOLO_MODEL_FILE = "yolo11n.onnx"
YUNET_MODEL_FILE = "face_detection_yunet_2023mar.onnx"
CLIP_VISION_MODEL_FILE = "mobileclip2_b_vision.onnx"
CLIP_TEXT_MODEL_FILE = "mobileclip2_b_text.onnx"

# --- Person detection ------------------------------------------------------
PERSON_CLASS_ID = 0  # COCO "person"
DETECT_CONFIDENCE = 0.40
DETECT_IOU = 0.45
DETECT_INPUT_SIZE = 640
MAX_DETECTIONS = 20

# The 15% rule from plan.txt is a *display* filter, not a compute saver: the
# detector has already run by the time a box exists. Real load control is
# DETECT_FPS below plus "classify each person once per track".
MIN_PERSON_AREA_RATIO = 0.15
# Hysteresis for that gate: once somebody is shown, they stay shown until they
# drop below 85% of the limit. Without it, a person standing at the edge of the
# threshold blinks in and out ~6 times a second.
AREA_GATE_HYSTERESIS = 0.85

# --- Tracking --------------------------------------------------------------
TRACK_IOU_THRESHOLD = 0.30
TRACK_MAX_MISSES = 6  # frames a track may be missing before it disappears
# A half-visible person entering the frame gets a partial box; judging that
# box gives a wrong crop and a verdict that then flips. Wait for a third hit
# (~0.5 s at DETECT_FPS) and let the track settle before the first check.
TRACK_CONFIRM_HITS = 3
TRACK_SETTLE_SECONDS = 0.8

# --- Verdict cache ---------------------------------------------------------
# A verdict is only *published* once two samples exist; until then the card says
# "checking...". The published probability is the median of the last
# SMOOTHING_WINDOW samples, so a crop that wobbles around the threshold (the
# reference cosine moves +/-0.03 between checks) cannot flip the card.
WARMUP_SAMPLES = 2
WARMUP_INTERVAL = 0.3  # seconds between the first two checks (kept short: the
# glasses verdict can already be published from a decisive first sample)
SMOOTHING_WINDOW = 5  # samples used for the published (median) probability
RECLASSIFY_INTERVAL = 5.0  # seconds between later refreshes (measured: one
# check ≈0.24 s, so 5 s keeps up to ~6 people in frame comfortably)
RECLASSIFY_MOVE_RATIO = 0.20  # centre moved more than 20% of the box height

# --- Torso / eye crops -----------------------------------------------------
# The shirt is not in the same place for every box (measured): a close-up
# webcam box is head + shoulders, so the shirt sits low (40-92%), while a
# full-body box has it in the upper-middle (12-62%). One fixed band crops the
# face on close-ups, which is the single biggest accuracy killer.
TORSO_CLOSE_UP_RATIO = 0.55  # box height / frame height
TORSO_TOP_CLOSE = 0.40
TORSO_BOTTOM_CLOSE = 0.92
TORSO_TOP_FULL = 0.12
TORSO_BOTTOM_FULL = 0.62
TORSO_SIDE_MARGIN = 0.08  # include the shirt edges, not just the middle

MIN_FACE_CROP_PX = 48  # face view smaller than this: glasses verdict is "unknown"
MIN_DETAIL_CROP_PX = 24  # the eye band is short by nature; below this it is noise

# --- Classification thresholds --------------------------------------------
# Verdicts are probabilities (0..1), not raw cosines: CLIP cosines live in a
# narrow band, so they are mapped with a calibrated sigmoid before deciding.
# Slider range in the view matches these.
UNIFORM_PROB_THRESHOLD = 0.50
UNIFORM_PROB_UNSURE_MARGIN = 0.10
# Slider ranges. The max is exposed to Flutter, which computes the far-right
# value as min + step*divisions and can land one float ulp *above* it
# (0.8500000000000001 > 0.85) - Flet then rejects the value and the app dies.
# The view adds a margin; the handlers clamp back to these bounds.
UNIFORM_THRESHOLD_MIN = 0.30
UNIFORM_THRESHOLD_MAX = 0.85
GLASSES_THRESHOLD_MIN = 0.02
GLASSES_THRESHOLD_MAX = 0.60
# Glasses needs a *high* bar, but not so high that wearers are told "no
# glasses". Default 0.25 sits between a wearer's face crop (0.28-0.56) and the
# worst false positive (0.21). The slider behind the app-bar gear goes down to
# 0.02 for thin frames.
GLASSES_PROB_THRESHOLD = 0.25
GLASSES_PROB_UNSURE_MARGIN = 0.06
# A clearly positive face view needs no second CLIP pass (halves the
# time-to-verdict for obvious wearers). A clearly negative *first sample* is
# also published at once instead of waiting for the median.
GLASSES_SHORTCUT_HIGH = 0.55
GLASSES_SHORTCUT_LOW = 0.05

# Cosine scale -> probability calibration. The uniform signal combines two
# things (both measured on live logs):
#   * the absolute level of the best "school uniform" prompt (real uniform
#     0.32-0.38, ordinary clothing 0.14-0.29),
#   * the gap to the best "casual" prompt (a casual white top scored
#     u=0.30 vs c=0.34, the real PE top u=0.36 vs c=0.30).
# Multiplying them means a crop has to look like a uniform *and* unlike casual
# clothes. Level alone let a white non-uniform top through; the gap alone let
# an ordinary portrait through at 0.998.
UNIFORM_TEXT_COS_MIDPOINT = 0.30
UNIFORM_TEXT_COS_SLOPE = 0.02
# Reference photos are product cut-outs, so a casual white top can reach 0.56
# cosine. They confirm a uniform but must not outvote the prompts: weighted
# mean, text-heavy (measured: this is what fixed the white casual top).
UNIFORM_TEXT_WEIGHT = 0.65
TEXT_SOFTMAX_TEMPERATURE = 100.0
# Measured on worn uniforms from live webcam logs: the best reference cosine is
# 0.41-0.61 for the school PE/shirt but 0.20-0.45 for ordinary clothing, so the
# midpoint sits at 0.50 (self-similarity of the product photos is 1.00).
REFERENCE_COS_THRESHOLD = 0.50  # reference-photo cosine that means "match"
REFERENCE_COS_SLOPE = 0.03

# Colour sanity check for the white shirt + red trim of the school uniform.
UNIFORM_WHITE_RATIO_MIN = 0.35
UNIFORM_RED_RATIO_MIN = 0.002

# --- Camera / preview ------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
PREVIEW_MAX_WIDTH = 960
PREVIEW_JPEG_QUALITY = 70
PREVIEW_MAX_FPS = 12
DETECT_FPS = 6

# --- Labelling (no real names, this is a public demo) ----------------------
PERSON_LABEL_PREFIX = "Person"