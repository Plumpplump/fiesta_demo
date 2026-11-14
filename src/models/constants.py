"""Tunable constants for the uniform/glasses demo.

Everything a teacher would want to adjust during the STEM fair lives here, so
nothing has to be dug out of the controllers.
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

# --- Tracking --------------------------------------------------------------
TRACK_IOU_THRESHOLD = 0.30
TRACK_MAX_MISSES = 6  # frames a track may be missing before it disappears
RECLASSIFY_INTERVAL = 3.0  # seconds; refresh an existing person's labels

# --- Torso / eye crops -----------------------------------------------------
TORSO_TOP = 0.18
TORSO_BOTTOM = 0.72
TORSO_LEFT = 0.18
TORSO_RIGHT = 0.82

MIN_EYE_CROP_PX = 24  # below this the glasses call is "unsure", not a guess

# --- Classification thresholds (deliberately loose: see plan.txt) ----------
UNIFORM_SIM_THRESHOLD = 0.22
UNIFORM_SIM_UNSURE_MARGIN = 0.05
GLASSES_SIM_THRESHOLD = 0.20
GLASSES_SIM_UNSURE_MARGIN = 0.04

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