# PlacementMentorAI

AI-powered placement readiness platform for students with two modules:

## Phase 1 — Placement Score Prediction
- **Model**: Random Forest Regressor trained on 15 skill features
- **Input**: Academic scores, technical skills, soft skills
- **Output**: Placement score (0–100), suggested role, weak areas, learning recommendations

## Phase 2 — Interview Behaviour Analysis
- **Model**: YOLO26-Pose (17 COCO keypoints) + Rule-Based Behaviour Classifier
- **Input**: Interview image (JPG/PNG) or video (MP4/AVI)
- **Output**: Behaviour classification (Confident / Nervous / Distracted), confidence score, feature breakdown, summary
- **Features Extracted**: Head Tilt Angle, Spine Alignment, Hand Movement, Eye Contact Proxy, Shoulder Tension

## Final Placement Readiness Score
```
final_score = (placement_score × 0.7) + (behaviour_score × 0.3)
```

---

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py

# Open browser at http://localhost:5000
```

### Docker
```bash
# Build and run
docker compose up -d

# Or manually
docker build -t placementmentorai .
docker run -p 5000:5000 placementmentorai
```

---

## Project Structure

```
PlacementMentorAI/
├── app.py                          # Flask application (all routes)
├── model.pkl                       # Phase 1 Random Forest model
├── train_model.py                  # Phase 1 training script
├── placementmentor_dataset.csv     # Phase 1 dataset
├── cv_module/                      # Phase 2 Computer Vision module
│   ├── __init__.py
│   ├── models/                     # Trained models (best.pt)
│   ├── training/
│   │   ├── train.py                # YOLO26-Pose + classifier training
│   │   └── dataset.yaml            # YOLO dataset configuration
│   ├── inference/
│   │   ├── image_predict.py        # Image behaviour analysis
│   │   ├── video_predict.py        # Video behaviour analysis
│   │   ├── pose_utils.py           # Keypoint extraction & features
│   │   └── behaviour_classifier.py # Rule-based classifier
│   └── outputs/                    # Processed images/videos
├── templates/
│   ├── index.html                  # Placement score form
│   ├── result.html                 # Prediction results
│   └── interview_analysis.html     # Interview analysis UI
├── static/
│   ├── style.css                   # Design system
│   └── script.js                   # Interview analysis JS
├── uploads/                        # User uploads
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── DEPLOYMENT.md                   # AWS EC2 deployment guide
└── DATASET_GUIDE.md                # Dataset recommendations
```

---

## API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Placement score prediction form |
| `/predict` | POST | Process placement prediction |
| `/interview` | GET | Interview behaviour analysis page |
| `/analyze-image` | POST | Analyze image for behaviour |
| `/analyze-video` | POST | Analyze video for behaviour |
| `/calculate-final-score` | POST | Compute final readiness score |
| `/outputs/<filename>` | GET | Serve processed files |

---

## Tech Stack
- **Backend**: Flask, Python 3.10+
- **ML/CV**: Ultralytics YOLO26-Pose, OpenCV, PyTorch, scikit-learn
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript, Bootstrap 5
- **Deployment**: Docker, Gunicorn, AWS EC2

---

## Training

```bash
# Generate synthetic training data
python -m cv_module.training.train --mode generate --samples 1000

# Train behaviour classifier
python -m cv_module.training.train --mode classifier --dataset cv_module/training/synthetic_features.csv

# Fine-tune YOLO26-Pose (requires labelled pose data)
python -m cv_module.training.train --mode pose --epochs 100
```

See [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed dataset recommendations.
