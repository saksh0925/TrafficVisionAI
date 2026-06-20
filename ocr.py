import easyocr
import cv2
import numpy as np
import re

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def extract_plate_text(img_bgr, bbox):
    """
    Crops license plate region and extracts text using EasyOCR.
    bbox = (x1, y1, x2, y2)
    Returns cleaned plate text or None.
    """
    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]

    padding = 8
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    plate_crop = img_bgr[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return None

    # Upscale generously — small plates lose character detail otherwise
    plate_crop = cv2.resize(plate_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # Mild denoise before threshold — reduces salt noise that breaks character edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    reader = get_reader()

    # detail=1 gives (bbox, text, confidence) per detected text region —
    # this lets us sort fragments left-to-right ourselves instead of relying
    # on EasyOCR's paragraph merging, which scrambles plate character order.
    # allowlist restricts to plate-valid characters, cutting misreads.
    results = reader.readtext(
        thresh,
        detail=1,
        paragraph=False,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    )

    if not results:
        return None

    # Sort fragments by horizontal position (left to right) and stack rows
    # top to bottom, in case the plate text spans two lines.
    results.sort(key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))

    # Keep only reasonably confident fragments
    fragments = [text for (_, text, conf) in results if conf > 0.25]

    if not fragments:
        return None

    raw = "".join(fragments).upper().strip()
    cleaned = re.sub(r'[^A-Z0-9]', '', raw)
    return cleaned if len(cleaned) >= 4 else None


def process_plates(img_bgr, detections):
    """
    For each detection that is a license_plate, extract text.
    Returns updated detections with plate_text field.
    """
    for det in detections:
        if det["class_name"] == "license_plate":
            text = extract_plate_text(img_bgr, det["bbox"])
            det["plate_text"] = text if text else "Unreadable"
        else:
            det["plate_text"] = None
    return detections