"""OCR module for MediCheck Sverige.

Main entry points (these are what the pipeline will call):
    extract_text(source, variant="clahe")        -> list of {"text", "conf"}
    extract_text_auto_rotate(source, variant)    -> same, but tries 0/90/180/270 degrees
    full_text(results)                           -> one string with all detected text

`source` can be a file path (str/Path) or raw image bytes (what the UI will send).
"""
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch

_reader = None


def get_reader():
    """Load EasyOCR once and reuse it (loading takes several seconds)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["sv", "en"], gpu=torch.cuda.is_available())
    return _reader


# ---------------------------------------------------------------- image loading
def load_image(source):
    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(source))
    if img is None:
        raise ValueError("Could not read image (bad path or corrupt file).")
    return img


def _resize(img, min_side=800, max_side=1800):
    """Upscale tiny images (small print needs pixels), downscale huge ones (speed)."""
    h, w = img.shape[:2]
    short, long_ = min(h, w), max(h, w)
    scale = 1.0
    if short < min_side:
        scale = min_side / short
    elif long_ > max_side:
        scale = max_side / long_
    if scale != 1.0:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
    return img


# ---------------------------------------------------------------- preprocessing
VARIANTS = ["raw", "clahe", "clahe_denoise", "adaptive"]


def preprocess(img, variant="clahe"):
    """Return an image prepared according to `variant`.

    raw            : just resized, no filtering (baseline)
    clahe          : grayscale + local contrast boost (helps with glare/dim light)
    clahe_denoise  : clahe + noise removal (helps grainy phone photos)
    adaptive       : black/white thresholding (helps flat, high-contrast text, can hurt curved bottles)
    """
    img = _resize(img)
    if variant == "raw":
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if variant == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if variant == "clahe_denoise":
        den = cv2.fastNlMeansDenoising(gray, None, h=10)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(den)
    if variant == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
    raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------- OCR
def extract_text(source, variant="clahe", min_conf=0.0):
    """Run OCR on one image. Returns [{'text': str, 'conf': float}, ...]."""
    img = preprocess(load_image(source), variant)
    raw = get_reader().readtext(img)  # list of (bbox, text, confidence)
    return [
        {"text": t.strip(), "conf": float(c)}
        for _, t, c in raw
        if t.strip() and c >= min_conf
    ]


_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def extract_text_auto_rotate(source, variant="clahe", min_conf=0.0):
    """Try 4 orientations, keep the one with the most confident text.
    Useful for photos taken sideways or upside down. ~4x slower than extract_text."""
    base = load_image(source)
    best, best_score = [], -1.0
    for angle, code in _ROTATIONS.items():
        img = base if code is None else cv2.rotate(base, code)
        raw = get_reader().readtext(preprocess(img, variant))
        res = [
            {"text": t.strip(), "conf": float(c)}
            for _, t, c in raw
            if t.strip() and c >= min_conf
        ]
        score = sum(r["conf"] * len(r["text"]) for r in res)
        if score > best_score:
            best, best_score = res, score
    return best


def full_text(results):
    return " ".join(r["text"] for r in results)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr.py path/to/image.jpg")
        sys.exit(1)
    t0 = time.time()
    out = extract_text(sys.argv[1])
    print(f"({time.time() - t0:.1f}s)")
    for r in out:
        print(f"{r['conf']:.2f}  {r['text']}")