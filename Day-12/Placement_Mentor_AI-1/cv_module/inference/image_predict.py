"""
Image-based Behaviour Analysis
Analyzes static images from interviews
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

from ultralytics import YOLO

from .pose_utils import PoseProcessor
from .behaviour_classifier import BehaviourClassifier, get_behaviour_name


class ImageBehaviourAnalyzer:
    """Analyze interview behaviour from static images"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize with YOLO pose model
        model_path: path to YOLO pose model (best.pt)
        """
        if model_path is None:
            # Use default YOLOv8 pose model
            self.model = YOLO('yolo26l-pose.pt')
        else:
            self.model = YOLO(model_path)
        
        self.pose_processor = PoseProcessor(confidence_threshold=0.5)
        self.behaviour_classifier = BehaviourClassifier()
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze interview behaviour from image
        
        Returns: {
            'behaviour': str,  # 'Confident', 'Nervous', 'Distracted'
            'confidence_score': float,  # 0-1
            'behaviour_score': float,  # 0-100
            'dominant_signal': str,
            'summary': str,
            'features': dict,  # Individual feature scores
            'processed_image_path': str,  # Path to visualized image
        }
        """
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'error': f'Failed to read image: {image_path}',
                'behaviour': None,
                'confidence_score': 0,
                'behaviour_score': 0,
            }
        
        height, width = image.shape[:2]
        
        # Run YOLO pose detection
        results = self.model(image, verbose=False)
        
        if not results or len(results) == 0:
            return {
                'error': 'No person detected in image',
                'behaviour': None,
                'confidence_score': 0,
                'behaviour_score': 0,
            }
        
        result = results[0]
        
        # Extract keypoints
        keypoints = self.pose_processor.extract_keypoints(result)
        if keypoints is None or len(keypoints) == 0:
            return {
                'error': 'Failed to extract pose keypoints',
                'behaviour': None,
                'confidence_score': 0,
                'behaviour_score': 0,
            }
        
        # Filter low-confidence keypoints
        keypoints = self.pose_processor.filter_keypoints(keypoints)
        
        # Extract behaviour features
        features = self.pose_processor.extract_behaviour_features(
            keypoints, 
            prev_keypoints=None,
            frame_height=height,
            frame_width=width
        )
        
        # Classify behaviour
        behaviour_class, class_confidence = self.behaviour_classifier.classify_frame(features)
        behaviour_name = get_behaviour_name(behaviour_class)
        
        # Calculate behaviour score (0-100)
        behaviour_score = class_confidence * 100
        
        # Get dominant signal
        dominant_signal = self.behaviour_classifier.get_dominant_signal(features, behaviour_class)
        
        # Generate summary
        analysis_dict = {
            'avg_confidence': class_confidence,
            'confident_ratio': 1.0 if behaviour_class == 0 else (0.5 if behaviour_class == 1 else 0.0),
            'nervous_ratio': 1.0 if behaviour_class == 1 else (0.5 if behaviour_class == 0 else 0.0),
            'distracted_ratio': 1.0 if behaviour_class == 2 else (0.5 if behaviour_class == 0 else 0.0),
            'frame_count': 1,
        }
        summary = self.behaviour_classifier.generate_summary(
            behaviour_class, 
            class_confidence, 
            features, 
            analysis_dict
        )
        
        # Create visualization
        visualized_image = self._visualize_results(image.copy(), keypoints, features, behaviour_name, behaviour_score)
        
        # Save visualization
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'analyzed_image_{timestamp}.jpg'
        cv2.imwrite(str(output_path), visualized_image)
        
        return {
            'behaviour': behaviour_name,
            'confidence_score': float(class_confidence),
            'behaviour_score': float(behaviour_score),
            'dominant_signal': dominant_signal,
            'summary': summary,
            'features': {
                'head_tilt_angle': float(features['head_tilt_angle']),
                'spine_alignment_score': float(features['spine_alignment_score']),
                'hand_movement_delta': float(features['hand_movement_delta']),
                'eye_contact_proxy': float(features['eye_contact_proxy']),
                'shoulder_tension_ratio': float(features['shoulder_tension_ratio']),
            },
            'processed_image_path': str(output_path),
            'error': None,
        }
    
    def _visualize_results(self, image: np.ndarray, keypoints: np.ndarray, 
                          features: Dict, behaviour: str, score: float) -> np.ndarray:
        """Add keypoints, skeleton, and text to image"""
        
        from .pose_utils import visualize_keypoints
        
        # Draw keypoints and skeleton
        image = visualize_keypoints(image, keypoints)
        
        # Add behaviour label and score
        text = f'{behaviour}: {score:.1f}%'
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, (0, 255, 0) if behaviour == 'Confident' else (0, 165, 255) if behaviour == 'Nervous' else (0, 0, 255), 
                   2)
        
        # Add feature information
        y_offset = 70
        feature_text = [
            f"Head Tilt: {features['head_tilt_angle']:.1f}°",
            f"Spine: {features['spine_alignment_score']:.1f}%",
            f"Hand Movement: {features['hand_movement_delta']:.1f}%",
            f"Eye Contact: {features['eye_contact_proxy']:.1f}%",
            f"Shoulder Tension: {features['shoulder_tension_ratio']:.1f}%",
        ]
        
        for text_item in feature_text:
            cv2.putText(image, text_item, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return image
