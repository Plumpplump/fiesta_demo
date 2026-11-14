# Fiesta Demo - School Uniform Check

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

> `flet run --web` runs Python on the **same machine** as the browser; OpenCV
> reads *that* webcam. Opening the page from a phone or another laptop will not
> show a camera image. That is expected, not a bug.

## Models

Model files live in `src/assets/models/` and are loaded with onnxruntime:

| File | Purpose | Size |
|---|---|---|
| `yolo11n.onnx` | person detection (COCO class 0) | ~10 MB |
| `face_detection_yunet_2023mar.onnx` | face + 5 landmarks (eye strip) | ~230 KB |
| `mobileclip2_b_vision.onnx` | uniform / glasses image encoder | export-time |
| `mobileclip2_b_text.onnx` | text tower (export only) | export-time |
| `uniform_prompts.npy` + `.json` | pre-computed uniform text embeddings | small |
| `glasses_prompts.npy` + `.json` | pre-computed glasses text embeddings | small |

Regenerate with (needs torch/ultralytics/open_clip - one time only):

```bat
"C:\Users\david\OneDrive\Python Coding\myvenv\Scripts\python.exe" scripts\export_models.py
```

The app degrades instead of crashing when a model is missing: no YOLO means no
people, no CLIP means the colour check still gives a uniform verdict and glasses
stay `unknown`.

## How it works

```
src/
  main.py                      ft.run(main), wires page -> controller
  models/                      no Flet imports, all unit tested
    constants.py               every threshold in one place
    geometry.py                letterbox, IoU, NMS, torso/eye boxes
    person_detector.py         YOLO11n ONNX session + decode_yolo
    tracks.py                  IoU tracking; new track == new person
    uniform.py                 CLIP similarity + white/red colour check
    face.py                    YuNet wrapper + pick_face
    glasses.py                 glasses verdict from the eye strip
    clip.py                    ONNX CLIP encoder + prompt bank
    status.py                  tracks -> Person N rows
    frames.py                  JPEG preview + box drawing
  services/
    camera.py                  OpenCV grabber thread (latest frame wins)
    assets.py                  FLET_ASSETS_DIR / src/assets resolution
  views/app_view.py            controls and 60/40 layout only
  controllers/app_controller.py two async loops (preview 12 fps, detect 6 fps)
```

Design rules worth keeping:

- **Detection runs at ~6 fps, not per frame.** Classifiers run once per person
  and refresh every 3 s (`RECLASSIFY_INTERVAL`).
- **The 15% area rule is a display filter, not an optimisation.** The detector
  has already run by the time a box exists. Load control comes from the frame
  rate and from classifying each track once.
- **Classification only happens for tracks that need it** (`status.needs_classification`).
- **Glasses never guesses.** Too small / no model / borderline → `unknown` or `unsure`.

## Tuning at the fair

- The slider under "Uniform match threshold" adjusts the CLIP similarity live
  (default 0.22, deliberately loose as requested). Moving it re-checks everyone.
- `src/models/constants.py` holds the rest: confidence, IoU, area gate, torso
  crop ratios, preview size/fps, minimum eye-crop size.
- Reference uniforms are read from any `src/assets/reference*` folder
  (`Reference Photos` today). Drop more photos of people **wearing** the uniform
  there and restart - flat product cut-outs are the weakest signal in the demo.

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
3. Check that `src/assets/models/yolo11n.onnx` exists (banner stays empty).
4. Light from the front; avoid a bright window behind the person.
5. Walk in → a `Person N` card appears within about a second.

## Packaging

```bat
myvenv\Scripts\flet.exe build macos -v
```

macOS apps need `NSCameraUsageDescription` in `Info.plist` (Flet adds it when
the camera permission is requested; for a packaged build add it explicitly).
`flet_camera` is deliberately not used: it raises
`FletUnsupportedPlatformException` on desktop, so OpenCV is the only path that
works on the MacBook.