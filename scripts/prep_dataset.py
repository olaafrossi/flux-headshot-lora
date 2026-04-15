"""Prepare a FLUX LoRA training dataset from raw headshot photos.

Reads images from dataset/raw/, detects the face with MediaPipe, crops to a
head+shoulders frame, resizes to 1024 on the longest side, and writes both
the image and a matching caption file to dataset/train/.

Run from the project root:

    python scripts/prep_dataset.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

try:
    import cv2
except ImportError:
    cv2 = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def load_face_detector():
    if cv2 is None:
        return None
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    return cascade


def detect_face_bbox(detector, image: Image.Image):
    """Return (x, y, w, h) of the largest detected face, or None."""
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def head_shoulders_crop(
    image: Image.Image,
    face_bbox: tuple[int, int, int, int],
    target_aspect: float = 3 / 4,
) -> Image.Image:
    """Expand a face bbox to a head+shoulders framing, clipped to image bounds."""
    iw, ih = image.size
    x, y, w, h = face_bbox

    cx = x + w / 2
    cy = y + h / 2 + h * 0.4  # shift down to keep hair + get shoulders

    crop_h = h * 3.2
    crop_w = crop_h * target_aspect

    left = max(0, int(round(cx - crop_w / 2)))
    top = max(0, int(round(cy - crop_h / 2)))
    right = min(iw, int(round(cx + crop_w / 2)))
    bottom = min(ih, int(round(cy + crop_h / 2)))

    return image.crop((left, top, right, bottom))


def center_crop(image: Image.Image, target_aspect: float = 3 / 4) -> Image.Image:
    iw, ih = image.size
    current = iw / ih
    if current > target_aspect:
        new_w = int(round(ih * target_aspect))
        offset = (iw - new_w) // 2
        return image.crop((offset, 0, offset + new_w, ih))
    new_h = int(round(iw / target_aspect))
    offset = (ih - new_h) // 2
    return image.crop((0, offset, iw, offset + new_h))


def resize_longest_side(image: Image.Image, longest: int) -> Image.Image:
    iw, ih = image.size
    if max(iw, ih) <= longest:
        return image
    if iw >= ih:
        new_w, new_h = longest, int(round(ih * longest / iw))
    else:
        new_h, new_w = longest, int(round(iw * longest / ih))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def prep_dataset(
    raw_dir: Path,
    train_dir: Path,
    trigger: str,
    size: int,
    detect_faces: bool,
) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)

    raw_images = sorted(
        p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not raw_images:
        print(f"No images found in {raw_dir}. Drop 15-25 photos and rerun.")
        return

    detector = load_face_detector() if detect_faces else None
    if detect_faces and detector is None:
        print(
            "WARNING: opencv-python-headless not installed — falling back to "
            "center crop. Run `pip install opencv-python-headless` for "
            "head+shoulders framing.",
            file=sys.stderr,
        )

    face_count = 0
    written = 0
    for idx, src in enumerate(tqdm(raw_images, desc="Prepping")):
        try:
            img = Image.open(src)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        except Exception as e:
            print(f"  skip {src.name}: {e}", file=sys.stderr)
            continue

        bbox = detect_face_bbox(detector, img) if detector is not None else None
        if bbox is not None:
            cropped = head_shoulders_crop(img, bbox)
            face_count += 1
        else:
            cropped = center_crop(img)

        resized = resize_longest_side(cropped, size)

        out_name = f"{idx:03d}_{src.stem}.jpg"
        out_path = train_dir / out_name
        resized.save(out_path, "JPEG", quality=95)
        out_path.with_suffix(".txt").write_text(
            f"a photo of {trigger}", encoding="utf-8"
        )
        written += 1

    print(
        f"\nPrepped {written} image(s) → {train_dir}  "
        f"(face detected on {face_count}/{written})"
    )
    print(f"Trigger word: {trigger}")
    if detect_faces and written and face_count < written * 0.7:
        print(
            "WARNING: face detection missed on many images. Check that photos "
            "are front-facing and not heavily filtered.",
            file=sys.stderr,
        )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Prep a FLUX LoRA dataset from raw headshot photos."
    )
    parser.add_argument("--raw-dir", type=Path, default=root / "dataset" / "raw")
    parser.add_argument("--train-dir", type=Path, default=root / "dataset" / "train")
    parser.add_argument("--trigger", default="ohwx_person")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--no-face-detect", action="store_true")
    args = parser.parse_args()

    prep_dataset(
        raw_dir=args.raw_dir,
        train_dir=args.train_dir,
        trigger=args.trigger,
        size=args.size,
        detect_faces=not args.no_face_detect,
    )


if __name__ == "__main__":
    main()
