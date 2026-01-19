# Face Recognition Based Attendance System

A robust face recognition–based attendance system designed for CCTV environments using **SCRFD (face detection)** and **ArcFace (face recognition)**.

This project supports:
- Multiple face detection
- Harsh lighting conditions
- Crowded classroom scenes
- CPU-only execution
- High accuracy identity recognition

---

##  Features

- SCRFD-based high-recall face detection
- ArcFace 512-D embedding recognition
- Multi-scale + gamma correction pipeline
- Temporal smoothing & hysteresis
- Soft-NMS for crowded scenes
- Secure attendance-ready architecture

---

## Technologies Used

| Component | Technology |
|---------|------------|
Face Detection | SCRFD (InsightFace) |
Face Recognition | ArcFace |
Inference | ONNX Runtime (CPU) |
Language | Python 3.10 |
Similarity Metric | Cosine Similarity |

---

##  Project Structure

Face-Recognition-Based-Attendance-System/
│
├── embeddings/
│ ├── embeddings.npy
│ └── labels.npy
│
├── data_all/
│ ├── reference_images/htno_pics/
│ └── test_inputs/
│
├── outputs_extreme/
│
├── scripts/
│ ├── recognize_extreme.py
│ ├── arcface_embed.py
│ └── retinaface_detect.py
│
├── requirements.txt
└── README.md

## ⚙️ Installation

pip install -r requirements.txt

To Generate Face Embeddings
  -> python scripts/arcface_embed.py

To Run Face Recognition
  -> python scripts/recognize_extreme.py

Results will be saved in:
  -> outputs_extreme/
## Content in requirements.txt
numpy==1.26.4
opencv-python-headless==4.12.0.88
insightface==0.7.3
onnxruntime==1.17.3
scikit-learn==1.7.2
tqdm==4.67.1
