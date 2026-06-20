import cv2
import numpy as np
from ultralytics import YOLO
import os

VIOLATION_COLORS = {
    "helmet_violation":    (0, 0, 255),
    "no_seatbelt":         (0, 69, 255),
    "triple_riding":       (0, 140, 255),
    "wrong_side":          (0, 165, 255),
    "stopline_violation":  (0, 215, 255),
    "redlight_violation":  (0, 0, 200),
    "illegal_parking":     (128, 0, 128),
    "vehicle":              (0, 200, 0),
    "rider":                (0, 180, 0),
    "pedestrian":           (180, 180, 0),
    "license_plate":        (255, 255, 0),
    "traffic_light_red":     (0, 0, 255),
    "traffic_light_green":   (0, 200, 0),
    "traffic_light_yellow":  (0, 220, 255),
    "person":                (0, 180, 0),
    "motorcycle":            (0, 200, 0),
    "bicycle":               (0, 200, 0),
    "car":                   (0, 200, 0),
    "truck":                 (0, 200, 0),
    "bus":                   (0, 200, 0),
}

# Map raw class names from every trained model to a unified violation key
CLASS_NORMALIZE = {
    # helmet model (bahasa labels)
    "tanpa helm":        "helmet_violation",
    "without helmet":    "helmet_violation",
    "no helmet":         "helmet_violation",
    "no_helmet":         "helmet_violation",
    "pakai helm":        "helmet_ok",
    "with helmet":       "helmet_ok",
    "helmet":            "helmet_ok",

    # seatbelt model
    "noseatbelt":        "no_seatbelt",
    "no_seatbelt":       "no_seatbelt",
    "no seatbelt":       "no_seatbelt",
    "seatbelt":          "seatbelt_ok",

    # illegal parking model
    "illegal parking":   "illegal_parking",
    "illegal_parking":   "illegal_parking",

    # stop line model
    "stopline":          "stopline_violation",
    "stop line":         "stopline_violation",
    "stop_line":         "stopline_violation",

    # wrong side model
    "wrong-side":        "wrong_side",
    "wrong side":        "wrong_side",
    "wrong_side":        "wrong_side",
    "right-side":        "right_side_ok",
    "right side":        "right_side_ok",

    # license plate model
    "license_plate":     "license_plate",
    "license plate":     "license_plate",
    "number_plate":      "license_plate",
    "number plate":      "license_plate",
    "plate":             "license_plate",

    # red light violation model
    "red_light":         "traffic_light_red",
    "red light":         "traffic_light_red",
    "redlight":          "traffic_light_red",
    "green_light":       "traffic_light_green",
    "green light":       "traffic_light_green",
    "yellow_light":      "traffic_light_yellow",
    "yellow light":      "traffic_light_yellow",

    # triple riding model
    "motorcycle":        "motorcycle",
    "person":             "person",
}

VIOLATION_LABELS = {
    "helmet_violation":   "Helmet non-compliance",
    "no_seatbelt":        "Seatbelt non-compliance",
    "triple_riding":      "Triple riding",
    "wrong_side":         "Wrong-side driving",
    "stopline_violation": "Stop-line violation",
    "redlight_violation": "Red-light violation",
    "illegal_parking":    "Illegal parking",
}

NON_VIOLATION_LABELS = {
    "helmet_ok":            "Helmet compliant",
    "seatbelt_ok":          "Seatbelt compliant",
    "right_side_ok":        "Correct-side driving",
    "traffic_light_red":    "Red light",
    "traffic_light_green":  "Green light",
    "traffic_light_yellow": "Yellow light",
}


def _normalize(raw_name):
    key = raw_name.lower().strip()
    return CLASS_NORMALIZE.get(key, key)


def _iou_overlap_area(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _person_on_motorcycle(person_bbox, moto_bbox):
    px1, py1, px2, py2 = person_bbox
    mx1, my1, mx2, my2 = moto_bbox
    x_overlap = max(0, min(px2, mx2) - max(px1, mx1))
    person_w = px2 - px1
    if person_w == 0:
        return False
    horiz_ratio = x_overlap / person_w
    vertically_ok = py2 > my1 and py1 < my2
    return horiz_ratio > 0.3 and vertically_ok


class TrafficDetector:
    """
    Loads multiple specialized YOLOv8 models and runs them all on one image,
    merging results into a unified violation list.
    """

    MODEL_FILES = {
        "helmet":   "traffic_violations.pt",
        "seatbelt": "seatbelt.pt",
        "parking":  "illegal_parking.pt",
        "stopline": "stopline.pt",
        "wrongside":"wrong_side.pt",
        "plate":    "plate_detector.pt",
        "triple":   "triple_riding.pt",
        "redlight": "redlight_detector.pt",
    }

    def __init__(self, models_dir="models"):
        self.models = {}
        self.models_dir = models_dir

        for key, filename in self.MODEL_FILES.items():
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                try:
                    self.models[key] = YOLO(path)
                except Exception as e:
                    print(f"Could not load {filename}: {e}")

        # fallback general detector for vehicles/persons if no custom models found
        if not self.models:
            self.models["base"] = YOLO("yolov8n.pt")
            self.base_only = True
        else:
            self.base_only = False
            # always keep a general model too, for vehicle/person context
            self.models["base"] = YOLO("yolov8n.pt")

        self.custom = not self.base_only

        # Some models need stricter confidence because they're prone to false
        # positives on out-of-domain inputs (e.g. seatbelt model on a bike scene)
        self.MODEL_MIN_CONF = {
            "seatbelt": 0.55,
            "stopline": 0.5,
            "wrongside": 0.5,
            "parking": 0.5,
            "helmet": 0.4,
            "triple": 0.4,
            "base": 0.4,
        }

        # These models only make sense in certain scene contexts.
        # We gate them NEGATIVELY (block when clearly wrong context) rather than
        # positively (require exact class), because interior car shots often
        # don't show "car" as a detectable object at all.
        self.NEGATIVE_GATED = {
            "seatbelt": ["motorcycle", "bicycle"],  # skip seatbelt check on clear two-wheeler scenes
        }

    def detect(self, img_bgr, conf_threshold=0.4, debug=False):
        all_detections = []
        debug_log = []

        # First pass: run base model to know what vehicle types are present
        base_results = self.models["base"](img_bgr, conf=conf_threshold, verbose=False)
        present_classes = set()
        for result in base_results:
            for box in result.boxes:
                present_classes.add(result.names[int(box.cls[0])].lower())

        for model_key, model in self.models.items():
            model_conf = max(conf_threshold, self.MODEL_MIN_CONF.get(model_key, conf_threshold))

            # Skip negatively-gated models if a clearly conflicting vehicle type is present
            blocking_classes = self.NEGATIVE_GATED.get(model_key)
            if blocking_classes and (present_classes & set(blocking_classes)):
                debug_log.append(f"{model_key}: SKIPPED (blocked by {present_classes & set(blocking_classes)})")
                continue

            # Run once at a very low floor just to see what the model *would* say,
            # for debug visibility, regardless of the real threshold used below.
            if debug:
                probe = model(img_bgr, conf=0.1, verbose=False)
                probe_scores = []
                for result in probe:
                    for box in result.boxes:
                        probe_scores.append(f"{result.names[int(box.cls[0])]}={float(box.conf[0]):.2f}")
                debug_log.append(f"{model_key}: threshold={model_conf} | raw scores: {probe_scores if probe_scores else 'none'}")

            results = model(img_bgr, conf=model_conf, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    raw_name = result.names[cls_id]
                    norm_name = _normalize(raw_name)

                    is_violation = norm_name in VIOLATION_LABELS
                    label = VIOLATION_LABELS.get(
                        norm_name,
                        NON_VIOLATION_LABELS.get(norm_name, raw_name.replace("_", " ").title())
                    )

                    all_detections.append({
                        "class_name": norm_name,
                        "raw_name": raw_name,
                        "source_model": model_key,
                        "confidence": round(conf, 3),
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "is_violation": is_violation,
                        "plate_text": None,
                    })

        all_detections = self._dedupe(all_detections)
        all_detections = self._apply_triple_riding_rule(all_detections)
        all_detections = self._apply_redlight_rule(all_detections)

        if debug:
            return all_detections, debug_log
        return all_detections


    def _dedupe(self, detections, iou_thresh=0.6):
        """Remove near-duplicate boxes of the same violation type from overlapping models."""
        kept = []
        detections = sorted(detections, key=lambda d: -d["confidence"])

        for det in detections:
            duplicate = False
            for k in kept:
                if k["class_name"] == det["class_name"]:
                    area_overlap = _iou_overlap_area(k["bbox"], det["bbox"])
                    a1 = (k["bbox"][2]-k["bbox"][0]) * (k["bbox"][3]-k["bbox"][1])
                    a2 = (det["bbox"][2]-det["bbox"][0]) * (det["bbox"][3]-det["bbox"][1])
                    if a1 + a2 > 0 and area_overlap / min(a1, a2) > iou_thresh:
                        duplicate = True
                        break
            if not duplicate:
                kept.append(det)

        return kept

    def _apply_triple_riding_rule(self, detections):
        """If triple-riding model didn't fire but we see 3+ persons on 1 motorcycle, flag it."""
        already_flagged = any(d["class_name"] == "triple_riding" for d in detections)
        if already_flagged:
            return detections

        persons = [d for d in detections if d["class_name"] == "person"]
        motos = [d for d in detections if d["class_name"] == "motorcycle"]

        for moto in motos:
            riders = [p for p in persons if _person_on_motorcycle(p["bbox"], moto["bbox"])]
            if len(riders) >= 3:
                x1, y1, x2, y2 = moto["bbox"]
                detections.append({
                    "class_name": "triple_riding",
                    "raw_name": "triple_riding (rule-based)",
                    "source_model": "rule",
                    "confidence": round(min(r["confidence"] for r in riders), 3),
                    "bbox": (x1 - 4, y1 - 4, x2 + 4, y2 + 4),
                    "label": "Triple riding",
                    "is_violation": True,
                    "plate_text": None,
                })
        return detections

    def _apply_redlight_rule(self, detections):
        """
        The red-light model only detects traffic light colors and vehicles —
        it does not by itself know if a SPECIFIC vehicle is violating the light.
        Rule: if a red light is detected AND a vehicle (car/truck/bus/motorcycle)
        bounding box overlaps or sits past the light's vertical position in frame
        (i.e. vehicle is in the intersection while light is red), flag it.
        This is a simplified spatial heuristic suitable for single-frame photos.
        """
        red_lights = [d for d in detections if d["class_name"] == "traffic_light_red"]
        if not red_lights:
            return detections

        vehicles = [d for d in detections if d["class_name"] in
                    ("car", "truck", "bus", "motorcycle") and d["source_model"] != "rule"]

        violations = []
        for light in red_lights:
            lx1, ly1, lx2, ly2 = light["bbox"]
            light_cy = (ly1 + ly2) / 2

            for v in vehicles:
                vx1, vy1, vx2, vy2 = v["bbox"]
                vehicle_cy = (vy1 + vy2) / 2
                # Vehicle is considered "at/past" the light if its vertical
                # center is below the light's vertical center (closer to camera/
                # further into the intersection) by a meaningful margin.
                if vehicle_cy > light_cy + (vy2 - vy1) * 0.3:
                    violations.append({
                        "class_name": "redlight_violation",
                        "raw_name": "redlight_violation (rule-based)",
                        "source_model": "rule",
                        "confidence": round(light["confidence"], 3),
                        "bbox": (vx1, vy1, vx2, vy2),
                        "label": "Red-light violation",
                        "is_violation": True,
                        "plate_text": None,
                    })

        return detections + violations


def draw_detections(img_bgr, detections, show_compliant=False):
    annotated = img_bgr.copy()

    for det in detections:
        if not det["is_violation"] and not show_compliant:
            # Skip drawing "helmet_ok" / "seatbelt_ok" boxes to reduce clutter,
            # but still draw base vehicle/person boxes
            if det["class_name"] in NON_VIOLATION_LABELS:
                continue

        x1, y1, x2, y2 = det["bbox"]
        color = VIOLATION_COLORS.get(det["class_name"], (200, 200, 200))
        thickness = 3 if det["is_violation"] else 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label_text = f"{det['label']} {det['confidence']:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, 1)
        ty = max(y1 - 6, th + 6)

        cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 8, ty + 2), color, -1)
        cv2.putText(annotated, label_text, (x1 + 4, ty - 2),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated