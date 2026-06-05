"""
Video-based Behaviour Analysis
Analyzes video interviews with temporal processing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from collections import deque

from ultralytics import YOLO

from .pose_utils import PoseProcessor
from .behaviour_classifier import BehaviourClassifier, get_behaviour_name


class VideoBehaviourAnalyzer:
    """Analyze interview behaviour from video"""
    
    def __init__(self, model_path: str = None, fps_target: int = 5):
        """
        Initialize with YOLO pose model
        model_path: path to YOLO pose model (best.pt)
        fps_target: process video at this FPS (default 5 for efficiency)
        """
        if model_path is None:
            self.model = YOLO('yolo26l-pose.pt')
        else:
            self.model = YOLO(model_path)
        
        self.pose_processor = PoseProcessor(confidence_threshold=0.5)
        self.behaviour_classifier = BehaviourClassifier()
        self.fps_target = fps_target
        self.sliding_window_size = 10  # Frames for temporal smoothing
    
    def analyze_video(self, video_path: str, max_frames: Optional[int] = None) -> Dict:
        """
        Analyze interview behaviour from video
        
        Args:
            video_path: path to video file
            max_frames: limit analysis to N frames (useful for testing)
        
        Returns: {
            'behaviour': str,
            'confidence_score': float,
            'behaviour_score': float,
            'dominant_signal': str,
            'summary': str,
            'features_over_time': list,  # Feature evolution
            'behaviour_distribution': dict,
            'processed_video_path': str,
        }
        """
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'error': f'Failed to open video: {video_path}',
                'behaviour': None,
                'confidence_score': 0,
                'behaviour_score': 0,
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame sampling interval
        frame_interval = max(1, int(fps / self.fps_target))
        
        # Process frames
        frame_idx = 0
        sample_idx = 0
        frame_behaviours = []
        frame_features_list = []
        prev_keypoints = None
        sample_frames = []  # For video output
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            # Limit frames for testing
            if max_frames and sample_idx >= max_frames:
                break
            
            # Run YOLO pose detection
            results = self.model(frame, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                # Extract keypoints
                keypoints = self.pose_processor.extract_keypoints(result)
                if keypoints is not None and len(keypoints) > 0:
                    keypoints = self.pose_processor.filter_keypoints(keypoints)
                    
                    # Apply temporal smoothing
                    keypoints_smoothed = self.pose_processor.apply_temporal_smoothing(
                        keypoints, 
                        window_size=self.sliding_window_size
                    )
                    
                    # Extract features
                    features = self.pose_processor.extract_behaviour_features(
                        keypoints_smoothed,
                        prev_keypoints=prev_keypoints,
                        frame_height=height,
                        frame_width=width
                    )
                    
                    # Classify behaviour
                    behaviour_class, class_confidence = self.behaviour_classifier.classify_frame(features)
                    
                    frame_behaviours.append((behaviour_class, class_confidence))
                    frame_features_list.append(features)
                    prev_keypoints = keypoints_smoothed
                    
                    # Store frame for video output
                    sample_frames.append((frame, keypoints, features, behaviour_class, 
                                        class_confidence * 100))
            
            frame_idx += 1
            sample_idx += 1
            
            # Progress indicator
            if sample_idx % 30 == 0:
                print(f"Processed {sample_idx} frames...")
        
        cap.release()
        
        if not frame_behaviours:
            return {
                'error': 'No pose detected in video',
                'behaviour': None,
                'confidence_score': 0,
                'behaviour_score': 0,
            }
        
        # Classify sequence
        dominant_behaviour, avg_confidence, analysis = self.behaviour_classifier.classify_sequence(
            frame_behaviours,
            window_size=self.sliding_window_size
        )
        
        behaviour_name = get_behaviour_name(dominant_behaviour)
        behaviour_score = avg_confidence * 100
        
        # Get dominant signal (from most common behaviour)
        dominant_signal = self.behaviour_classifier.get_dominant_signal(
            frame_features_list[len(frame_features_list)//2],  # Use middle frame
            dominant_behaviour
        )
        
        # Generate summary
        summary = self.behaviour_classifier.generate_summary(
            dominant_behaviour,
            avg_confidence,
            frame_features_list[len(frame_features_list)//2],
            analysis
        )
        
        # Create output video with visualization
        output_video_path = self._create_output_video(
            video_path,
            sample_frames,
            fps,
            height,
            width,
            behaviour_name,
            behaviour_score
        )
        
        # Prepare features over time
        features_over_time = []
        for i, features in enumerate(frame_features_list):
            features_over_time.append({
                'frame': i,
                'head_tilt_angle': float(features['head_tilt_angle']),
                'spine_alignment_score': float(features['spine_alignment_score']),
                'hand_movement_delta': float(features['hand_movement_delta']),
                'eye_contact_proxy': float(features['eye_contact_proxy']),
                'shoulder_tension_ratio': float(features['shoulder_tension_ratio']),
            })
        
        return {
            'behaviour': behaviour_name,
            'confidence_score': float(avg_confidence),
            'behaviour_score': float(behaviour_score),
            'dominant_signal': dominant_signal,
            'summary': summary,
            'behaviour_distribution': {
                'confident': float(analysis['confident_ratio']),
                'nervous': float(analysis['nervous_ratio']),
                'distracted': float(analysis['distracted_ratio']),
            },
            'total_frames_analyzed': analysis['frame_count'],
            'features_over_time': features_over_time,
            'processed_video_path': output_video_path,
            'error': None,
        }
    
    def _create_output_video(self, input_video_path: str, sample_frames: List, 
                            fps: float, height: int, width: int,
                            behaviour: str, score: float) -> str:
        """Create output video with visualization"""
        
        from .pose_utils import visualize_keypoints
        
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'analyzed_video_{timestamp}.mp4'
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps / max(1, int(fps / self.fps_target)), 
                             (width, height))
        
        colour_map = {
            0: (0, 255, 0),      # Confident - Green
            1: (0, 165, 255),    # Nervous - Orange
            2: (0, 0, 255),      # Distracted - Red
        }
        
        behaviour_names_map = {0: 'Confident', 1: 'Nervous', 2: 'Distracted'}
        
        for frame, keypoints, features, behaviour_class, frame_score in sample_frames:
            # Draw keypoints and skeleton
            frame = visualize_keypoints(frame, keypoints)
            
            # Add behaviour label
            behaviour_text = f'{behaviour_names_map[behaviour_class]}: {frame_score:.1f}%'
            color = colour_map[behaviour_class]
            cv2.putText(frame, behaviour_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1.2, color, 2)
            
            # Add overall summary
            cv2.putText(frame, f'Overall: {behaviour} ({score:.1f}%)', (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"Output video saved to: {output_path}")
        
        return str(output_path)
