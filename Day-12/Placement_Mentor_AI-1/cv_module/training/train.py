"""
YOLO26-Pose Training Pipeline for Interview Behaviour Analysis

This module provides two training paths:
1. Fine-tune YOLO26-Pose on custom interview/pose data
2. Train a behaviour classifier (RandomForest/SVM) on extracted feature vectors

Usage:
    # Fine-tune YOLO26-Pose model
    python -m cv_module.training.train --mode pose --epochs 100

    # Train behaviour classifier on labelled features
    python -m cv_module.training.train --mode classifier --data features.csv
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def train_pose_model(
    base_model: str = 'yolo26l-pose.pt',
    dataset_yaml: str = None,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 16,
    image_size: int = 640,
    output_dir: str = None,
):
    """
    Fine-tune YOLO26-Pose on a custom pose dataset.

    Args:
        base_model: Base YOLO pose model to fine-tune
        dataset_yaml: Path to dataset.yaml
        epochs: Number of training epochs
        patience: Early stopping patience
        batch_size: Training batch size
        image_size: Input image size
        output_dir: Directory to save trained model
    """
    from ultralytics import YOLO

    # Default paths
    if dataset_yaml is None:
        dataset_yaml = os.path.join(os.path.dirname(__file__), 'dataset.yaml')

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  YOLO26-Pose Fine-Tuning")
    print("=" * 60)
    print(f"  Base Model:   {base_model}")
    print(f"  Dataset:      {dataset_yaml}")
    print(f"  Epochs:       {epochs}")
    print(f"  Patience:     {patience}")
    print(f"  Batch Size:   {batch_size}")
    print(f"  Image Size:   {image_size}x{image_size}")
    print(f"  Output Dir:   {output_dir}")
    print("=" * 60)

    # Load base model
    model = YOLO(base_model)

    # Train with specified configuration
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        imgsz=image_size,
        # Augmentations
        mosaic=1.0,           # Mosaic ON
        degrees=10.0,         # Rotation ±10°
        hsv_v=0.3,            # Brightness ±0.3
        flipud=0.0,           # Vertical flip OFF
        fliplr=0.0,           # Horizontal flip OFF (interview face orientation matters)
        # Training parameters
        optimizer='auto',     # MuSGD or SGD based on YOLO26 defaults
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        weight_decay=0.0005,
        # Loss functions
        pose=12.0,            # OKS pose loss weight
        cls=0.5,              # Classification loss weight
        box=7.5,              # Box loss weight
        # Output
        project=output_dir,
        name='yolo26_pose_interview',
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    # Copy best model to models directory
    best_model_path = Path(output_dir) / 'yolo26_pose_interview' / 'weights' / 'best.pt'
    target_path = Path(output_dir) / 'best.pt'

    if best_model_path.exists():
        import shutil
        shutil.copy2(str(best_model_path), str(target_path))
        print(f"\n✅ Best model saved to: {target_path}")
    else:
        print(f"\n⚠️  Best model not found at: {best_model_path}")

    print("\n📊 Training Results:")
    print(f"  Metrics: {results.results_dict if hasattr(results, 'results_dict') else 'See training logs'}")

    return results


def train_behaviour_classifier(
    features_csv: str,
    output_dir: str = None,
    model_type: str = 'random_forest',
):
    """
    Train a behaviour classifier on labelled feature vectors.

    The CSV should have columns:
    - head_tilt_angle
    - spine_alignment_score
    - hand_movement_delta
    - eye_contact_proxy
    - shoulder_tension_ratio
    - behaviour_label (0=Confident, 1=Nervous, 2=Distracted)

    Args:
        features_csv: Path to CSV with labelled features
        output_dir: Directory to save classifier
        model_type: 'random_forest' or 'svm'
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import pickle

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Behaviour Classifier Training")
    print("=" * 60)

    # Load data
    df = pd.read_csv(features_csv)
    print(f"  Dataset Size: {len(df)} samples")
    print(f"  Columns:      {df.columns.tolist()}")

    feature_columns = [
        'head_tilt_angle',
        'spine_alignment_score',
        'hand_movement_delta',
        'eye_contact_proxy',
        'shoulder_tension_ratio',
    ]

    # Validate columns
    for col in feature_columns + ['behaviour_label']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    X = df[feature_columns].values
    y = df['behaviour_label'].values

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Class Distribution:")
    class_names = {0: 'Confident', 1: 'Nervous', 2: 'Distracted'}
    for cls, count in zip(unique, counts):
        print(f"    {class_names.get(cls, cls)}: {count}")

    # Split: 70% train, 20% val, 10% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.222, random_state=42, stratify=y_train_val
    )  # 0.222 * 0.9 ≈ 0.2

    print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    if model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
        )
        classifier.fit(X_train_scaled, y_train)
    elif model_type == 'svm':
        classifier = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42,
        )
        classifier.fit(X_train_scaled, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Evaluate
    print(f"\n📊 Validation Results ({model_type}):")
    val_pred = classifier.predict(X_val_scaled)
    print(classification_report(y_val, val_pred,
                                target_names=['Confident', 'Nervous', 'Distracted']))

    print(f"\n📊 Test Results ({model_type}):")
    test_pred = classifier.predict(X_test_scaled)
    print(classification_report(y_test, test_pred,
                                target_names=['Confident', 'Nervous', 'Distracted']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    # Save classifier and scaler
    classifier_path = os.path.join(output_dir, 'behaviour_classifier.pkl')
    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')

    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n✅ Classifier saved to: {classifier_path}")
    print(f"✅ Scaler saved to:     {scaler_path}")

    # Save training metadata
    metadata = {
        'model_type': model_type,
        'feature_columns': feature_columns,
        'class_names': class_names,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'timestamp': datetime.now().isoformat(),
    }

    metadata_path = os.path.join(output_dir, 'classifier_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return classifier, scaler


def generate_synthetic_features(n_samples: int = 1000, output_path: str = None):
    """
    Generate synthetic feature data for testing the classifier training pipeline.
    This creates realistic-looking feature distributions for each behaviour class.

    Args:
        n_samples: Total number of samples to generate
        output_path: Path to save the CSV
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'synthetic_features.csv')

    np.random.seed(42)
    samples_per_class = n_samples // 3

    data = []

    # Class 0: Confident
    for _ in range(samples_per_class):
        data.append({
            'head_tilt_angle': np.random.normal(0, 5),           # Near 0
            'spine_alignment_score': np.random.normal(85, 8),     # High
            'hand_movement_delta': np.random.normal(10, 5),       # Low
            'eye_contact_proxy': np.random.normal(80, 10),        # High
            'shoulder_tension_ratio': np.random.normal(25, 8),    # Low
            'behaviour_label': 0,
        })

    # Class 1: Nervous
    for _ in range(samples_per_class):
        data.append({
            'head_tilt_angle': np.random.normal(-10, 8),          # Some tilt
            'spine_alignment_score': np.random.normal(60, 12),    # Medium
            'hand_movement_delta': np.random.normal(55, 15),      # High
            'eye_contact_proxy': np.random.normal(40, 12),        # Medium-low
            'shoulder_tension_ratio': np.random.normal(65, 10),   # High
            'behaviour_label': 1,
        })

    # Class 2: Distracted
    for _ in range(samples_per_class):
        data.append({
            'head_tilt_angle': np.random.normal(-30, 12),         # Large tilt
            'spine_alignment_score': np.random.normal(40, 15),    # Low
            'hand_movement_delta': np.random.normal(30, 10),      # Medium
            'eye_contact_proxy': np.random.normal(20, 10),        # Very low
            'shoulder_tension_ratio': np.random.normal(45, 12),   # Medium
            'behaviour_label': 2,
        })

    df = pd.DataFrame(data)

    # Clip values to realistic ranges
    df['head_tilt_angle'] = df['head_tilt_angle'].clip(-60, 60)
    df['spine_alignment_score'] = df['spine_alignment_score'].clip(0, 100)
    df['hand_movement_delta'] = df['hand_movement_delta'].clip(0, 100)
    df['eye_contact_proxy'] = df['eye_contact_proxy'].clip(0, 100)
    df['shoulder_tension_ratio'] = df['shoulder_tension_ratio'].clip(0, 100)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} synthetic samples → {output_path}")

    return df


def main():
    """Main entry point for training pipeline."""
    parser = argparse.ArgumentParser(
        description='PlacementMentorAI - Training Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['pose', 'classifier', 'generate'],
        required=True,
        help='Training mode: pose (YOLO26-Pose), classifier (behaviour), generate (synthetic data)'
    )
    parser.add_argument('--base-model', default='yolo26l-pose.pt',
                        help='Base YOLO model for pose training')
    parser.add_argument('--dataset', default=None,
                        help='Path to dataset.yaml (pose) or features.csv (classifier)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--image-size', type=int, default=640,
                        help='Image size for pose training')
    parser.add_argument('--model-type', default='random_forest',
                        choices=['random_forest', 'svm'],
                        help='Classifier model type')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for trained model')

    args = parser.parse_args()

    if args.mode == 'pose':
        train_pose_model(
            base_model=args.base_model,
            dataset_yaml=args.dataset,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            image_size=args.image_size,
            output_dir=args.output_dir,
        )

    elif args.mode == 'classifier':
        if args.dataset is None:
            print("Error: --dataset (features CSV) is required for classifier training")
            print("Tip: Run with --mode generate first to create synthetic data")
            sys.exit(1)
        train_behaviour_classifier(
            features_csv=args.dataset,
            output_dir=args.output_dir,
            model_type=args.model_type,
        )

    elif args.mode == 'generate':
        generate_synthetic_features(
            n_samples=args.samples,
            output_path=args.dataset,
        )


if __name__ == '__main__':
    main()
