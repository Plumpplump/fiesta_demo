# Fiesta AI Vision Demo-Detection of Uniform and Eyewear

Author: David Li, SISHK

A Flet demo for the school STEM fair: the laptop webcam watches the room, finds
people, boxes their torso, decides whether they wear the school uniform, and
boxes their eyes to decide whether they wear glasses. Each person is shown as
`Person1`, `Person2`, ... - no names, no recordings, nothing is stored.

Everything runs locally on the MacBook: OpenCV reads the webcam, YOLO11n finds
people, MobileCLIP2 compares torsos with the reference uniform photos, and
OpenCV YuNet locates the eyes.

## Run it

This machine uses the shared virtualenv at
`C:\Users\david\OneDrive\Python Coding\myvenv` (see `pyproject.toml` notes), so
`uv run` would create a *separate* `.venv`. Use the helper scripts instead:

```bat
run.bat          :: desktop window   (myvenv\Scripts\flet.exe run)
run-web.bat      :: browser          (myvenv\Scripts\flet.exe run --web)
test.bat         :: fast test suite  (myvenv\Scripts\python.exe -m pytest -q)
```

On the MacBook (or any machine using uv normally):

```bash
uv sync
uv run flet run          # desktop
uv run flet run --web    # browser
uv run pytest -q         # fast suite
```

On *this* Mac the dependencies live in the shared venv one level up, so the
shortest path is:

```bash
../myvenv/bin/flet run                 # desktop window
../myvenv/bin/flet run --web           # browser
../myvenv/bin/python -m pytest -q      # tests
```

> `flet run --web` runs Python on the **same machine** as the browser; OpenCV
> reads *that* webcam. Opening the page from a phone or another laptop will not
> show a camera image. That is expected, not a bug.

The camera does **not** start by itself. The app loads the models, shows
`Press 'Start camera' to begin`, and waits for the button - no webcam light, no
permission prompt until you ask for it. Press it again to stop.

Layout: app bar with the settings gear, camera preview on the left (60%), the
"Who is in view" cards on the right (40%, full height). Settings slide in under
the preview rather than cluttering the results panel. Each person gets a vivid
colour (never white/light grey) and large status badges.

A new person is **not judged while walking in**: the track has to be confirmed
by three detections and settle for 0.8 s, and the card then publishes the
**median of the last three checks** (two checks are needed before anything is
shown). Until then it says `checking...` rather than flashing a wrong "in
uniform" for a few seconds. Expect a decided card ~2 s after somebody enters.

Each check prints the published value and the raw sample, so a wobbling crop is
visible rather than mysterious:

```
[detect] Person1: uniform=not in uniform p=0.30 (sample 0.59) | glasses=glasses p=0.59 (sample 0.66)
```

Every verdict is printed to the terminal, which is the demo's debug view:

```
[detect] 1 person(s) in view | YOLO scores: 0.91
[detect] Person1: uniform=not in uniform p=0.05 | glasses=glasses p=0.31
           text 0.05 ('a person wearing a school uniform' 0.24 vs ...)
           p=0.31 (0.48, 0.20) prompt match (0.31)
```

The bracketed numbers are the two glasses views (face crop, eye band); the
higher one decides - useful when deciding which slider to move.

## Models

Model files live in `src/assets/models/` and are loaded with onnxruntime:

| File | Purpose | Size |
|---|---|---|
| `yolo11n.onnx` | person detection (COCO class 0) | ~10 MB |
| `face_detection_yunet_2023mar.onnx` | face + 5 landmarks (eye strip) | ~230 KB |
| `mobileclip2_b_vision.onnx` | uniform / glasses image encoder | ~350 MB |
| `mobileclip2_b_vision.json` | that encoder's mean/std/image size | small |
| `mobileclip2_b_text.onnx` | text tower (export only, app never loads it) | ~254 MB |
| `uniform_prompts.npy` + `.json` | pre-computed uniform text embeddings | small |
| `glasses_prompts.npy` + `.json` | pre-computed glasses text embeddings | small |

Regenerate with (needs torch/ultralytics/open_clip - one time only):

```bat
"C:\Users\david\OneDrive\Python Coding\myvenv\Scripts\python.exe" scripts\export_models.py
```

On a fresh MacBook the export toolchain lives in a throwaway environment, so the
app runtime (shared `myvenv`) never gains torch:

```bash
uv venv --python 3.13 .export-venv
VIRTUAL_ENV=.export-venv uv pip install torch torchvision ultralytics open-clip-torch transformers
.export-venv/bin/python scripts/export_models.py   # writes src/assets/models/
rm -rf .export-venv                                # one-off, safe to delete
```

`src/assets/models/` is gitignored (the `.onnx` files are too big for git), so a
fresh clone starts with no models and shows the "model missing" banner. Run the
export once per machine. You can watch it land:

```bash
ls -lh src/assets/models/
```

| File | Status |
|---|---|
| `face_detection_yunet_2023mar.onnx` | downloaded (~230 KB) |
| `yolo11n.onnx` | exported from the YOLO11n weights (~10 MB) |
| `mobileclip2_b_vision.onnx` / `_text.onnx` | exported from MobileCLIP2-B |
| `uniform_prompts.*` / `glasses_prompts.*` | text vectors for both classifiers |

The app degrades instead of crashing when a model is missing: no YOLO means no
people, no CLIP means the colour check still gives a uniform verdict and glasses
stay `unknown`.

## How it works

```
src/
  main.py                      ft.run(main), wires page -> controller
  models/                      no Flet imports, all unit tested
    constants.py               every threshold in one place
    geometry.py                letterbox, IoU, NMS, torso/eye/face boxes
    person_detector.py         YOLO11n ONNX session + decode_yolo
    tracks.py                  IoU tracking, 2-hit confirmation, pose-aware torso band
    uniform.py                 P(uniform) from prompt level + reference photos
    face.py                    YuNet wrapper, face_box(), pick_face()
    glasses.py                 face + eye-band fusion -> P(glasses)
    clip.py                    ONNX CLIP encoder, prompt bank, calibration helpers
    status.py                  tracks -> Person N rows + verdict cache
    frames.py                  JPEG preview + box drawing
  services/
    camera.py                  OpenCV grabber thread (latest frame wins)
    assets.py                  FLET_ASSETS_DIR / src/assets resolution
  views/app_view.py            app bar, 60/40 layout, settings panel
  controllers/app_controller.py camera button + two async loops (12/6 fps)
scripts/
  export_models.py             one-time YOLO/YuNet/MobileCLIP2 export (+ prompts)
  calibrate.py                 score your own photos, prints the best threshold
```

Design rules worth keeping:

- **The camera starts on the button**, never on launch.
- **Detection runs at ~6 fps, not per frame.** Classifiers run once per person
  and are cached for 5 s or until the person moves more than 20% of their box
  height (`RECLASSIFY_*`).
- **The 15% area rule is a display filter, not an optimisation.** The detector
  has already run by the time a box exists. Load control comes from the frame
  rate and from classifying each track once.
- **Classification only happens for tracks that need it** (`status.needs_classification`),
  and only after a track is confirmed by two detections.
- **Boxes are glued to the face.** YuNet runs once per detection frame (6 fps)
  just to refresh the face/eye boxes, so the blue eye box follows a moving head
  instead of sitting where the last verdict was computed. Every box is clipped
  to the face and to the frame - it can never leave the picture.
- **A verdict is the median of recent checks** (`status.smoothed_score`), not a
  single sample. Live logs showed the same person scoring 0.15 / 0.59 / 0.30 as
  the crop shifted; the median settles at 0.30 instead of flipping the card.
- **Glasses never guesses.** Too small / no model / borderline → `unknown` or `unsure`.
- **Glasses answer fast.** A decisive first sample (≥ 0.55 "glasses" or ≤ 0.05
  "no glasses") is published immediately, and a clearly positive face view
  skips the eye-band CLIP pass. The uniform verdict still waits for two samples.

## Tuning at the fair

The gear button in the app bar opens **Settings** underneath the camera video
(the preview shrinks, the status panel keeps its full height). Two sliders,
both re-checking everyone immediately:

- **Uniform match threshold** (default 0.50, range 0.30-0.85): P(uniform) above
  this is "in uniform". Raise it if casual clothes pass, lower it if the school
  shirt does not.
- **Glasses sensitivity** (default 0.25, range 0.02-0.60): lower = finds more
  glasses. Thin/rimless frames need a lower value; watch the `p=` value in the
  terminal while you stand in front of the camera.

`src/models/constants.py` holds the rest: confidence, IoU, area gate, torso
crop ratios, preview size/fps, minimum crop sizes, and `RECLASSIFY_INTERVAL`
(5 s - how often a standing person's verdict is refreshed; moving people are
re-checked sooner).

Reference uniforms are read from any `src/assets/reference*` folder
(`Reference Photos` today). Drop more photos of people **wearing** the uniform
there and restart - flat product cut-outs are the weakest signal in the demo.

Recalibrate on your own photos (no torch needed, ~20 s):

```bash
python scripts/calibrate.py \
    --uniform "src/assets/Reference Photos" --casual ~/Pictures/casual \
    --glasses-with ~/Pictures/glasses --glasses-without ~/Pictures/no-glasses
```

It prints every score, the best separating threshold (Youden's J) and the
numbers the shipped defaults would produce, so a threshold is a measurement
rather than a guess.

## Glasses: what actually works

Measured on this Mac with 13 public portraits **and** live webcam logs (thin
and rimless frames included) - the same stack the sibling project used:

| View | Result |
|---|---|
| tight eye strip only | no better than chance (Youden J 0.00) |
| whole face crop only | separates, but misses thin frames and keeps 1 false positive (a hand/mic near the face, p=0.21) |
| one blurry view + one clear view | the blurry one must not veto: face 0.42 x eye 0.00 fused to 0.04 "no glasses" |

So the classifier scores both views and takes the **higher** probability. On the
spot check that detects 5 of 8 wearers (0.25-0.82) with the worst bare face at
0.21 - below the 0.25 threshold.

Known limits, stated rather than hidden:

- Thin wire/rimless frames can score ~0.00 and are missed (Page, Brin, Wozniak
  in the spot check). Lower the slider to 0.05-0.10 if that is your face.
- Faces smaller than 48 px make the verdict `unknown` (never a guess); the eye
  band needs only 24 px because it is short by nature.
- A face must be found by YuNet first; profile views and masks fail there.
- The eye band gets its own chance to win: it is what catches glasses when the
  face crop is too wide/blurred.

## What we learned from `schooluniform`

https://github.com/yunguangli/schooluniform runs the same models (YOLO11n +
MobileCLIP2-B + YuNet) and documents its measurements. We adopted the
measurements that transfer and replaced the ones that did not:

| Finding there | Here |
|---|---|
| Crop quality beats model size; a fixed torso band crops the face on close-ups | Two bands by pose: close-up 40-92%, full body 12-62%, plus 8% side margin |
| A partial first box produces a wrong crop | 3 hits + 0.8 s settle before the first check (stricter: they use 2 hits) |
| Track every detection, mark far people instead of dropping boxes | Adopted; the 15% gate only controls judging/display, with 15% entry / 12.75% exit hysteresis |
| Largest face inside the person box wins | Adopted (replaces "nearest the head area") |
| Verdict cache: 10 s TTL, or shift **or size change** > 20% of height | Adopted both, TTL 5 s (measured: 0.24 s per check, 6 people comfortable); walking toward the camera also re-checks |
| Face crop (+margin) beats the eye strip for glasses | Both views scored, the higher one wins (a blurred view must not veto) |
| Two explicit Start/Stop buttons | One async toggle (Start camera / Stop camera), disabled while opening, one retry for macOS |
| Publish the first verdict immediately | **Not** adopted: live log flapped 0.15 / 0.59 / 0.30 as the crop shifted; we publish the median of 3 samples |
| Thresholds calibrated on 22 portraits | Recalibrated for this uniform and this Mac; `scripts/calibrate.py` reproduces it |
| CoreML EP for ONNX | YOLO keeps CoreML; MobileCLIP2 runs on CPU (measured 170 ms vs 400 ms with CoreML partitioning) |
| Prompt sets: 8 uniform / 9 casual / 4+3 glasses | Adopted, plus the school-specific red-trim and thin-frame prompts |

Where we deliberately differ: the sibling project scores uniforms with a
temperature-100 softmax over the *gap* between the best uniform and casual
prompt. The gap alone is not enough on our data - an ordinary portrait with
u=0.22, c=0.21 produces a confident 0.73 - and the level alone is not enough
either: a white casual top matched "school PE shirt" at u=0.30, exactly like a
real uniform. We multiply **level x gap** (must look like a uniform *and* unlike
casual clothes), then blend that with the reference-photo cosine **65/35**.

That last number matters: with `max()` a casual white top hit 0.56 against the
product-photo references (p=0.88) and overrode prompts that clearly said
"casual". Replaying a live log through the weighted blend gives the casual top
0.23-0.31 on every sample (correctly "not in uniform") and the real PE uniform
0.94-0.99. Colour (white/red) is reported but does not override the model: the
PE top is not white.

Two things they listed as not-yet-wired are implemented here: the shirt crop is
anchored below the detected chin (instead of guessing from the person box), and
the glasses verdict takes the better of two views.

## Tests

```bat
test.bat                               :: fast suite, no Flutter needed
myvenv\Scripts\python.exe -m pytest tests\test_geometry.py -q
myvenv\Scripts\python.exe -m pytest tests\test_main.py -q   :: needs Flutter, skipped by default
```

`tests/test_main.py` is an `integration` test: it drives a provisioned Flutter
test host, so it is skipped unless run through `flet test` or with
`FLET_RUN_INTEGRATION=1`. Never define your own `flet_app` fixture - the flet
pytest plugin injects it.

## Demo-day checklist

1. macOS: System Settings → Privacy & Security → Camera → allow your terminal.
2. `run-web.bat`, open the local URL **on the same MacBook**.
3. Press **Start camera** (the app never opens the webcam by itself).
4. Check that `src/assets/models/yolo11n.onnx` exists (banner stays empty).
5. Light from the front; avoid a bright window behind the person.
6. Walk in → a `Person N` card appears within about a second, decided ~2 s
   later (the card shows a median of the last three checks).
7. Stand in front of the camera and watch the `p=` values in the terminal; use
   the two sliders if the uniform or glasses verdict is wrong for you.

## If the preview flickers or blinks

The image is a JPEG pushed to an `ft.Image` twelve times a second, so three
things have to hold:

1. `gapless_playback=True` on the preview. With it off, Flutter paints *nothing*
   while the next frame decodes - a visible blink every frame.
2. Only genuinely new frames are pushed. `_preview_loop` compares
   `camera.frame_id`; re-sending a stale frame restarts the image provider.
3. Exactly one thread reads the webcam. Each `start()` gives its thread a fresh
   capture + stop flag, and a leftover thread from a previous run exits instead
   of alternating frames with the new one.

If it still flickers: check the FPS readout in the app (it should say ~30 FPS;
2 FPS means the webcam negotiated a slow format), quit other apps that hold the
camera (Zoom, Photo Booth, Chrome), and grant Camera access to your terminal in
System Settings → Privacy & Security.

## Packaging

```bat
myvenv\Scripts\flet.exe build macos -v
```

macOS apps need `NSCameraUsageDescription` in `Info.plist` (Flet adds it when
the camera permission is requested; for a packaged build add it explicitly).
`flet_camera` is deliberately not used: it raises
`FletUnsupportedPlatformException` on desktop, so OpenCV is the only path that
works on the MacBook.