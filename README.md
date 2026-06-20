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
├── detector.py         ← YOLOv8 detection
├── preprocessor.py     ← Image enhancement
├── ocr.py              ← License plate OCR
├── evidence.py         ← PDF report generation
├── requirements.txt    ← Dependencies
├── TRAINING_COLAB.py   ← Google Colab training script
└── models/
    └── traffic_violations.pt  ← Put trained model here
```

---

## Training Your Own Model (Google Colab)

1. Open TRAINING_COLAB.py
2. Copy cells into a new Google Colab notebook
3. Get free Roboflow API key at roboflow.com
4. Run all cells (~1-2 hours on free GPU)
5. Download best.pt and put in models/ folder

---

## Datasets Used

- Helmet detection: Roboflow Universe
- Seatbelt detection: Kaggle
- License plates: Indian License Plate dataset
- Vehicle detection: YOLOv8 pretrained (COCO)

---

## Violations Detected

1. Helmet non-compliance
2. Seatbelt non-compliance
3. Triple riding
4. Wrong-side driving
5. Stop-line violation
6. Red-light violation
7. Illegal parking

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object detection | YOLOv8 (Ultralytics) |
| Image processing | OpenCV |
| OCR | EasyOCR |
| Web UI | Streamlit |
| PDF reports | ReportLab |
| Language | Python 3.10+ |
| Training | Google Colab (free GPU) |
