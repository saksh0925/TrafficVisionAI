---
title: TrafficVision AI
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# TrafficVision AI
## Automated Traffic Violation Detection Using Computer Vision

---

## Setup (5 minutes)

### Step 1: Install Python 3.10+
Download from python.org

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the app
```bash
streamlit run app.py
```

Open browser at http://localhost:8501

---

## Project Structure

```
TrafficVisionAI/
├── app.py              ← Main Streamlit UI
├── detector.py         ← YOLOv8 multi-model detection
├── preprocessor.py     ← Image enhancement
├── ocr.py              ← License plate OCR
├── evidence.py         ← PDF report generation
├── requirements.txt    ← Dependencies
├── Dockerfile           ← Hugging Face Spaces deployment
├── packages.txt          ← System-level dependencies
├── TRAINING_COLAB.py   ← Google Colab training script
└── models/
    ├── traffic_violations.pt
    ├── seatbelt.pt
    ├── illegal_parking.pt
    ├── stopline.pt
    ├── wrong_side.pt
    ├── triple_riding.pt
    ├── plate_detector.pt
    └── redlight_detector.pt
```

---

## Datasets Used

All datasets sourced from Roboflow Universe, trained on Google Colab (free T4 GPU).

---

## Violations Detected

1. Helmet non-compliance
2. Seatbelt non-compliance
3. Triple riding
4. Wrong-side driving
5. Stop-line violation
6. Red-light violation
7. Illegal parking

Plus license plate detection and OCR.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object detection | YOLOv8 (Ultralytics) — 7 specialized models |
| Image processing | OpenCV |
| OCR | EasyOCR |
| Web UI | Streamlit |
| PDF reports | ReportLab |
| Language | Python 3.10+ |
| Training | Google Colab (free GPU) |
| Deployment | Hugging Face Spaces (Docker) |