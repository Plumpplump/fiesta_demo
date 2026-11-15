# Author: David Li, SISHK
"""Calibrate the uniform / glasses thresholds on your own photos.

The demo ships thresholds measured on the sibling project's data
(https://github.com/yunguangli/schooluniform). Uniforms, lighting and cameras
differ per school, so before the fair run this once on *your* photos and copy
the suggested numbers into ``src/models/constants.py``:

    # uniform: reference tops (should score high) vs casual photos (low)
    uv run python scripts/calibrate.py \
        --uniform "src/assets/Reference Photos" \
        --casual ~/Pictures/casual

    # glasses: folders of faces with and without glasses
    uv run python scripts/calibrate.py \
        --glasses-with ~/Pictures/glasses --glasses-without ~/Pictures/no-glasses

No torch needed: this uses the exported ONNX models and the same code paths as
the app, so the numbers are the numbers the demo will show.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import constants as C  # noqa: E402
from models.clip import ClipEncoder, PromptBank, softmax_pair  # noqa: E402
from models.face import FaceDetector  # noqa: E402
from models.glasses import GlassesClassifier  # noqa: E402
from models.uniform import UniformClassifier, load_references, reference_dir  # noqa: E402
from services import assets  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def load_encoder() -> ClipEncoder | None:
    try:
        return ClipEncoder(assets.model_file(C.CLIP_VISION_MODEL_FILE))
    except Exception as exc:  # pragma: no cover - depends on local models
        print(f"CLIP encoder unavailable ({exc}); export the models first.")
        return None


def best_threshold(positive: list[float], negative: list[float]) -> tuple[float, float]:
    """Threshold maximising Youden's J (true-positive rate - false-positive rate)."""
    best = (0.5, -2.0)
    for threshold in np.arange(0.01, 0.995, 0.005):
        tpr = float(np.mean([value >= threshold for value in positive])) if positive else 0.0
        fpr = float(np.mean([value >= threshold for value in negative])) if negative else 0.0
        j = tpr - fpr
        if j > best[1]:
            best = (float(threshold), j)
    return best


def report(name: str, positive: list[float], negative: list[float], default: float) -> None:
    if not positive and not negative:
        return
    print(f"\n{name}")
    print(f"  positives: {[round(v, 3) for v in positive]}")
    print(f"  negatives: {[round(v, 3) for v in negative]}")
    if positive and negative:
        threshold, j = best_threshold(positive, negative)
        print(
            f"  separation: best threshold {threshold:.3f} (Youden J {j:.2f}); "
            f"shipped default {default:.2f}"
        )
        print(
            f"  with {threshold:.2f}: "
            f"{100 * float(np.mean([v >= threshold for v in positive])):.0f}% of positives "
            f"and {100 * float(np.mean([v >= threshold for v in negative])):.0f}% of "
            "negatives above threshold"
        )
    elif positive:
        print(f"  all positives above {default:.2f}: {all(v >= default for v in positive)}")
    else:
        print(f"  all negatives below {default:.2f}: {all(v < default for v in negative)}")


def read_frame(path: Path, max_width: int = 1280):
    """Read an image the way the camera sees it: at most ``max_width`` wide.

    Full-resolution download (3000+ px) makes YuNet and CLIP behave nothing
    like a 1280x720 webcam frame, so calibration must downscale first.
    """
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    if width > max_width:
        image = cv2.resize(
            image,
            (max_width, max(1, int(round(height * max_width / width)))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def uniform_scores(
    encoder: ClipEncoder,
    prompts: PromptBank | None,
    photos: list[Path],
    references: list,
) -> list[float]:
    """P(uniform) exactly as the app computes it (text softmax fused with refs)."""
    from models.geometry import crop

    import cv2

    classifier = UniformClassifier(references, encoder, prompts)
    scores: list[float] = []
    for photo in photos:
        image = read_frame(photo)
        if image is None:
            print(f"  skip unreadable {photo}")
            continue
        height, width = image.shape[:2]
        box = (0.0, 0.0, float(width), float(height))
        verdict = classifier.classify(crop(image, box))
        scores.append(float(verdict.score) if verdict.score is not None else 0.0)
        print(f"  {photo.name:40s} p={verdict.score if verdict.score is None else round(verdict.score, 3)}  {verdict.label}")
    return scores


def glasses_scores(
    encoder: ClipEncoder,
    detector: FaceDetector | None,
    prompts: PromptBank | None,
    photos: list[Path],
    crop_kind: str,
) -> list[float]:
    """P(glasses) per photo, using the same crops/fusion as the app."""
    import cv2

    from models.geometry import crop, eye_box_from_face, face_crop_box

    classifier = GlassesClassifier(encoder, prompts)
    scores: list[float] = []
    for photo in photos:
        image = read_frame(photo)
        if image is None:
            print(f"  skip unreadable {photo}")
            continue
        if detector is None:
            print("  YuNet model missing; cannot locate faces")
            break
        faces = detector.detect(image)
        if not faces:
            print(f"  {photo.name:40s} no face")
            continue
        face = max(faces, key=lambda row: float(row[2]) * float(row[3]))
        fx, fy, fw, fh = (float(v) for v in face[:4])
        face_region = crop(image, face_crop_box((fx, fy, fx + fw, fy + fh)))
        eye_region = eye_box_from_face(face)
        eye_region = crop(image, eye_region) if eye_region is not None else None
        if crop_kind == "face":
            verdict = classifier.classify(face_region)
        elif crop_kind == "eye":
            verdict = classifier.classify(eye_region)
        else:
            verdict = classifier.classify(face_region, eye_region)
        score = verdict.score if verdict.score is not None else 0.0
        scores.append(float(score))
        print(
            f"  {photo.name:40s} p={score:.3f} verdict={verdict.label:10s} "
            f"face={classifier.probability(face_region) if classifier.ready else 'n/a'}"
        )
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform", type=Path, default=PROJECT_ROOT / "src/assets/Reference Photos")
    parser.add_argument("--casual", type=Path, help="folder of non-uniform photos")
    parser.add_argument("--glasses-with", type=Path, help="folder of faces wearing glasses")
    parser.add_argument("--glasses-without", type=Path, help="folder of faces without glasses")
    parser.add_argument(
        "--glasses-crop",
        choices=("fused", "eye", "face"),
        default="fused",
        help="fused = face + eye band (what the app does), or a single view",
    )
    args = parser.parse_args()

    encoder = load_encoder()
    if encoder is None:
        return 1
    uniform_prompts = _prompts("uniform_prompts.npy")
    glasses_prompts = _prompts("glasses_prompts.npy")
    references = load_references(assets.assets_dir())
    detector = _face_detector()

    print(f"reference photos: {[r.name for r in references]}")
    print(f"vision providers: {encoder.session.get_providers()}")

    uniform_positive = uniform_scores(
        encoder, uniform_prompts, images_in(args.uniform), references
    )
    uniform_negative = (
        uniform_scores(encoder, uniform_prompts, images_in(args.casual), references)
        if args.casual
        else []
    )
    report(
        "uniform P(uniform)", uniform_positive, uniform_negative,
        C.UNIFORM_PROB_THRESHOLD,
    )

    if args.glasses_with or args.glasses_without:
        with_glasses = (
            glasses_scores(
                encoder, detector, glasses_prompts,
                images_in(args.glasses_with), args.glasses_crop,
            )
            if args.glasses_with
            else []
        )
        without_glasses = (
            glasses_scores(
                encoder, detector, glasses_prompts,
                images_in(args.glasses_without), args.glasses_crop,
            )
            if args.glasses_without
            else []
        )
        report(
            f"glasses P(glasses), {args.glasses_crop} crop",
            with_glasses, without_glasses, C.GLASSES_PROB_THRESHOLD,
        )
    return 0


def _prompts(filename: str) -> PromptBank | None:
    try:
        return PromptBank.load(assets.model_file(filename))
    except Exception:
        return None


def _face_detector() -> FaceDetector | None:
    try:
        return FaceDetector(assets.model_file(C.YUNET_MODEL_FILE))
    except Exception as exc:
        print(f"face detector unavailable ({exc})")
        return None


if __name__ == "__main__":
    sys.exit(main())