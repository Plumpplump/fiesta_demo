"""One-time model export: YOLO11n + MobileCLIP2-B (+ text prompt vectors).

Run from the project root:

    uv run --group export python scripts/export_models.py

or, on this machine, with the shared virtualenv:

    "C:\\Users\\david\\OneDrive\\Python Coding\\myvenv\\Scripts\\python.exe" scripts/export_models.py

torch/ultralytics/open_clip are only needed here; the app itself loads the
resulting .onnx files with onnxruntime.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
MODELS = SRC / "assets" / "models"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

CLIP_HUB_ID = "hf-hub:timm/MobileCLIP2-B-OpenCLIP"

UNIFORM_PROMPTS = {
    "in uniform": [
        "a photo of a student wearing a white school uniform shirt with red trim",
        "a person wearing a white polo shirt school uniform with a school crest",
        "a student wearing a school PE shirt",
    ],
    "not in uniform": [
        "a person wearing casual clothes, not a school uniform",
        "a photo of someone in a plain coloured t-shirt",
        "a person in home clothes",
    ],
}

GLASSES_PROMPTS = {
    "glasses": [
        "a close-up of a person wearing eyeglasses",
        "a face with glasses frames over the eyes",
    ],
    "no glasses": [
        "a close-up of a person without glasses",
        "a face with bare eyes and no frames",
    ],
}


def export_yolo(target: Path) -> bool:
    from ultralytics import YOLO

    print("exporting YOLO11n -> ONNX ...")
    model = YOLO("yolo11n.pt")
    exported = model.export(format="onnx", imgsz=640, opset=17)
    exported_path = Path(exported)
    shutil.copyfile(exported_path, target)
    print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def export_yunet(target: Path) -> bool:
    if target.is_file():
        print(f"YuNet already present: {target.name}")
        return True
    import urllib.request

    print("downloading YuNet face detector ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(YUNET_URL, target)
    print(f"  wrote {target} ({target.stat().st_size / 1e3:.0f} KB)")
    return True


def _save_prompts(encoder, tokenizer, prompts: dict[str, list[str]], stem: str) -> None:
    import torch

    labels: list[str] = []
    texts: list[str] = []
    for label, sentences in prompts.items():
        for sentence in sentences:
            labels.append(label)
            texts.append(sentence)

    tokens = tokenizer(texts)
    if not torch.is_tensor(tokens):
        tokens = torch.as_tensor(tokens)
    with torch.no_grad():
        vectors = encoder.encode_text(tokens.long())
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)

    npy_path = MODELS / f"{stem}_prompts.npy"
    meta_path = MODELS / f"{stem}_prompts.json"
    import numpy as np

    np.save(npy_path, vectors.float().cpu().numpy())
    meta_path.write_text(
        json.dumps({"labels": labels, "texts": texts}, indent=2), encoding="utf-8"
    )
    print(f"  wrote {npy_path.name} + {meta_path.name} ({len(texts)} prompts)")


def export_clip() -> bool:
    import torch
    import open_clip

    print(f"loading {CLIP_HUB_ID} (first run downloads a few hundred MB) ...")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_HUB_ID)
    tokenizer = open_clip.get_tokenizer(CLIP_HUB_ID)
    model.eval()

    image_size = getattr(model.visual, "image_size", 224)
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[-1]
    print(f"  image size: {image_size}")

    vision_path = MODELS / "mobileclip2_b_vision.onnx"
    text_path = MODELS / "mobileclip2_b_text.onnx"

    class VisionOnly(torch.nn.Module):
        def __init__(self, visual):
            super().__init__()
            self.visual = visual

        def forward(self, image):
            return self.visual(image)

    class TextOnly(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model

        def forward(self, tokens):
            return self.clip_model.encode_text(tokens)

    dummy_image = torch.randn(1, 3, int(image_size), int(image_size))
    try:
        torch.onnx.export(
            VisionOnly(model.visual).eval(),
            dummy_image,
            str(vision_path),
            input_names=["image"],
            output_names=["embedding"],
            opset_version=17,
            dynamic_axes={"image": {0: "batch"}},
            dynamo=False,
        )
        print(f"  wrote {vision_path.name} ({vision_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        print(f"  vision export FAILED: {exc}")
        return False

    try:
        dummy_tokens = torch.zeros((1, 77), dtype=torch.long)
        torch.onnx.export(
            TextOnly(model).eval(),
            dummy_tokens,
            str(text_path),
            input_names=["tokens"],
            output_names=["embedding"],
            opset_version=17,
            dynamic_axes={"tokens": {0: "batch"}},
            dynamo=False,
        )
        print(f"  wrote {text_path.name} ({text_path.stat().st_size / 1e6:.1f} MB)")
        _save_prompts(model, tokenizer, UNIFORM_PROMPTS, "uniform")
        _save_prompts(model, tokenizer, GLASSES_PROMPTS, "glasses")
    except Exception as exc:
        print(f"  text export FAILED (glasses stays 'unknown'): {exc}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    args = parser.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)

    export_yunet(MODELS / "face_detection_yunet_2023mar.onnx")

    if not args.skip_yolo:
        export_yolo(MODELS / "yolo11n.onnx")

    if not args.skip_clip:
        export_clip()

    print("\nDone. Files in", MODELS)
    for path in sorted(MODELS.iterdir()):
        print(f"  {path.name:45s} {path.stat().st_size / 1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())