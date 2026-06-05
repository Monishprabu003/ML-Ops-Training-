# Dataset Guide — Interview Behaviour Analysis

## Overview

The behaviour analysis system uses a **two-stage pipeline**:
1. **YOLO26-Pose** extracts 17 COCO keypoints from images/video
2. **Behaviour Classifier** maps extracted features to 3 classes: Confident, Nervous, Distracted

This guide covers dataset options for both stages.

---

## Stage 1: Pose Estimation Dataset

### Option A: Use Pretrained Model (Recommended for Start)

The pretrained `yolo26l-pose.pt` model is trained on **COCO-Pose** (200K+ images, 250K+ person instances with 17 keypoints). For most interview scenarios, this works well out of the box.

**No additional data needed for pose estimation if using the pretrained model.**

### Option B: Fine-Tune on Interview-Specific Data

If pose detection is poor for your specific camera angle or environment:

#### Recommended Datasets
| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **COCO Keypoints** | 200K images | Gold standard for pose estimation | [cocodataset.org](https://cocodataset.org) |
| **MPII Human Pose** | 25K images | Multi-person pose, diverse activities | [human-pose.mpi-inf.mpg.de](http://human-pose.mpi-inf.mpg.de) |
| **CrowdPose** | 20K images | Challenging crowded scenes | [GitHub](https://github.com/Jeff-sjtu/CrowdPose) |

#### Custom Data Collection for Interviews
1. Record 30–60 second interview clips in your target environment
2. Vary: lighting, camera angle, distance, background
3. Annotate using [CVAT](https://cvat.ai) or [Label Studio](https://labelstud.io)
4. Minimum: 500 images recommended
5. Export in YOLO-Pose format

---

## Stage 2: Behaviour Classification Dataset

### Data Requirements
To train/improve the behaviour classifier, you need **labelled feature vectors** with these columns:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `head_tilt_angle` | float | -60° to 60° | Angle between eyes relative to horizontal |
| `spine_alignment_score` | float | 0–100 | Shoulder-hip vertical alignment |
| `hand_movement_delta` | float | 0–100 | Wrist position change between frames |
| `eye_contact_proxy` | float | 0–100 | Nose position relative to frame center |
| `shoulder_tension_ratio` | float | 0–100 | Shoulder height relative to hips |
| `behaviour_label` | int | 0, 1, 2 | 0=Confident, 1=Nervous, 2=Distracted |

### Option A: Collect Real Data

#### Recording Setup
- **Camera**: Webcam or phone at eye level
- **Resolution**: 720p or higher
- **Lighting**: Well-lit, front-facing
- **Background**: Clean, minimal distractions
- **Duration**: 30–60 seconds per sample

#### Collection Protocol
1. **Confident Samples**: Ask participants to:
   - Sit upright with good posture
   - Maintain eye contact with camera
   - Keep hands still on desk or lap
   - Speak confidently about a prepared topic

2. **Nervous Samples**: Ask participants to:
   - Fidget with hands, touch face frequently
   - Shift in seat, adjust position often
   - Break eye contact frequently
   - Speak about an uncomfortable topic

3. **Distracted Samples**: Ask participants to:
   - Look at phone or off-camera frequently
   - Turn head to look at different areas
   - Appear disengaged, lean back
   - Show minimal response to questions

#### Labelling Guidelines
- Each video → multiple frames → multiple feature vectors
- Label at video level, then propagate to all extracted features
- Minimum: **200 samples per class** (600 total)
- Ideal: **500+ samples per class** (1500+ total)

### Option B: Use Public Datasets

| Dataset | Classes | Size | Relevance |
|---------|---------|------|-----------|
| **DAiSEE** | Engagement/Boredom/Confusion/Frustration | 9K videos | High (student engagement) |
| **EmotiW** | 7 emotions | 1.8K videos | Medium (emotion → behaviour mapping) |
| **AffectNet** | 8 emotions | 450K images | Medium (facial features) |
| **FER2013** | 7 emotions | 35K images | Low (faces only, no body) |

> **Note**: Public datasets may need label remapping. E.g., DAiSEE engagement levels can be mapped to:
> - High Engagement → Confident (0)
> - Low Engagement → Distracted (2)
> - Confusion/Frustration → Nervous (1)

### Option C: Generate Synthetic Data (Quick Start)

Use the built-in synthetic data generator for pipeline testing:

```bash
python -m cv_module.training.train --mode generate --samples 1000
```

This creates realistic feature distributions but should **not** be used as a substitute for real data in production.

---

## Data Augmentation

### For Pose Model Fine-Tuning
- ✅ Mosaic augmentation
- ✅ Rotation ±10°
- ✅ Brightness ±0.3
- ❌ Horizontal flip (disabled — face orientation matters)
- ❌ Vertical flip

### For Behaviour Classifier
- Add Gaussian noise to feature values (σ=2-5% of range)
- Slight random scaling of features (0.95–1.05×)
- SMOTE for class imbalance if needed

---

## Dataset Split

| Split | Proportion | Purpose |
|-------|-----------|---------|
| **Train** | 70% | Model training |
| **Validation** | 20% | Hyperparameter tuning, early stopping |
| **Test** | 10% | Final evaluation (never used during training) |

Ensure stratified splitting to maintain class balance across splits.

---

## Training Commands

```bash
# 1. Generate synthetic data for testing
python -m cv_module.training.train --mode generate --samples 1000

# 2. Train behaviour classifier on features
python -m cv_module.training.train --mode classifier --dataset cv_module/training/synthetic_features.csv

# 3. Fine-tune YOLO26-Pose (requires labelled pose data)
python -m cv_module.training.train --mode pose --epochs 100 --batch-size 16
```
