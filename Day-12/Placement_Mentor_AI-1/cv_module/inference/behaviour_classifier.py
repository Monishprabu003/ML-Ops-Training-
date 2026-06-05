"""
Behaviour Classifier
Classifies interview behaviour into 3 categories:
- 0: Confident
- 1: Nervous
- 2: Distracted
"""

import numpy as np
from typing import Dict, Tuple, List
import json


class BehaviourClassifier:
    """
    Classifies interview behaviour based on extracted features
    """
    
    def __init__(self):
        # Define behaviour thresholds
        self.confident_thresholds = {
            'head_tilt_angle': (-15, 15),  # Upright head position
            'spine_alignment_score': (70, 100),  # Good spine alignment
            'hand_movement_delta': (0, 20),  # Low hand movement
            'eye_contact_proxy': (60, 100),  # Good eye contact
            'shoulder_tension_ratio': (0, 40),  # Relaxed shoulders
        }
        
        self.nervous_indicators = {
            'hand_movement_delta': (30, 100),  # Frequent hand movement
            'shoulder_tension_ratio': (60, 100),  # Shoulder tension
            'eye_contact_proxy': (20, 50),  # Some eye contact issues
            'head_tilt_angle': (-30, -15),  # Some head tilt
        }
        
        self.distracted_indicators = {
            'eye_contact_proxy': (0, 30),  # Poor eye contact
            'head_tilt_angle': (-60, -20),  # Head turned away
            'spine_alignment_score': (0, 50),  # Poor posture
        }
    
    def classify_frame(self, features: Dict[str, float]) -> Tuple[int, float]:
        """
        Classify a single frame's behaviour
        Returns: (behaviour_class, confidence_score)
        - 0: Confident
        - 1: Nervous
        - 2: Distracted
        """
        
        confident_score = self._calculate_class_score(features, 'confident')
        nervous_score = self._calculate_class_score(features, 'nervous')
        distracted_score = self._calculate_class_score(features, 'distracted')
        
        scores = [confident_score, nervous_score, distracted_score]
        behaviour_class = np.argmax(scores)
        confidence = scores[behaviour_class] / (sum(scores) + 1e-6)
        
        return behaviour_class, confidence
    
    def _calculate_class_score(self, features: Dict[str, float], behaviour_type: str) -> float:
        """Calculate score for a specific behaviour type"""
        
        if behaviour_type == 'confident':
            thresholds = self.confident_thresholds
            score = 0
            count = 0
            
            for feature_name, (min_val, max_val) in thresholds.items():
                if feature_name in features:
                    feat_val = features[feature_name]
                    # Score based on proximity to threshold range
                    if min_val <= feat_val <= max_val:
                        distance = min(abs(feat_val - min_val), abs(feat_val - max_val))
                        max_distance = max(abs(max_val - min_val) / 2, 1)
                        score += max(0, 100 - (distance / max_distance * 100))
                    else:
                        distance = min(abs(feat_val - min_val), abs(feat_val - max_val))
                        score += max(0, 50 - (distance / 50 * 50))
                    count += 1
            
            return score / max(count, 1)
        
        elif behaviour_type == 'nervous':
            indicators = self.nervous_indicators
            score = 0
            count = 0
            
            for feature_name, (min_val, max_val) in indicators.items():
                if feature_name in features:
                    feat_val = features[feature_name]
                    # Score based on proximity to nervous indicators
                    if min_val <= feat_val <= max_val:
                        distance = min(abs(feat_val - min_val), abs(feat_val - max_val))
                        max_distance = max(abs(max_val - min_val) / 2, 1)
                        score += max(0, 100 - (distance / max_distance * 100))
                    count += 1
            
            return score / max(count, 1)
        
        elif behaviour_type == 'distracted':
            indicators = self.distracted_indicators
            score = 0
            count = 0
            
            for feature_name, (min_val, max_val) in indicators.items():
                if feature_name in features:
                    feat_val = features[feature_name]
                    # Score based on proximity to distracted indicators
                    if min_val <= feat_val <= max_val:
                        distance = min(abs(feat_val - min_val), abs(feat_val - max_val))
                        max_distance = max(abs(max_val - min_val) / 2, 1)
                        score += max(0, 100 - (distance / max_distance * 100))
                    count += 1
            
            return score / max(count, 1)
        
        return 0
    
    def classify_sequence(self, frame_behaviours: List[Tuple[int, float]], 
                         window_size: int = 10) -> Tuple[int, float, Dict]:
        """
        Classify a sequence of frames using temporal smoothing
        Returns: (dominant_behaviour, confidence, analysis_dict)
        """
        if not frame_behaviours:
            return 0, 0.5, {}
        
        # Apply temporal smoothing using sliding window
        smoothed_behaviours = []
        for i in range(len(frame_behaviours)):
            start = max(0, i - window_size // 2)
            end = min(len(frame_behaviours), i + window_size // 2)
            window = frame_behaviours[start:end]
            
            avg_confidence = np.mean([conf for _, conf in window])
            behaviour_votes = np.bincount([b for b, _ in window])
            dominant = np.argmax(behaviour_votes)
            
            smoothed_behaviours.append((dominant, avg_confidence))
        
        # Get overall dominant behaviour
        final_behaviours = [b for b, _ in smoothed_behaviours]
        final_confidences = [c for _, c in smoothed_behaviours]
        
        behaviour_counts = np.bincount(final_behaviours, minlength=3)
        dominant_behaviour = np.argmax(behaviour_counts)
        
        # Calculate statistics
        confident_ratio = behaviour_counts[0] / len(final_behaviours)
        nervous_ratio = behaviour_counts[1] / len(final_behaviours)
        distracted_ratio = behaviour_counts[2] / len(final_behaviours)
        avg_confidence = np.mean(final_confidences)
        
        analysis = {
            'confident_ratio': confident_ratio,
            'nervous_ratio': nervous_ratio,
            'distracted_ratio': distracted_ratio,
            'avg_confidence': avg_confidence,
            'frame_count': len(final_behaviours),
        }
        
        return dominant_behaviour, avg_confidence, analysis
    
    def get_dominant_signal(self, features: Dict[str, float], behaviour_class: int) -> str:
        """Identify the strongest signal for a behaviour"""
        
        if behaviour_class == 0:  # Confident
            signals = {
                'eye_contact': features.get('eye_contact_proxy', 0),
                'spine_alignment': features.get('spine_alignment_score', 0),
                'low_tension': 100 - features.get('shoulder_tension_ratio', 0),
                'low_hand_movement': 100 - features.get('hand_movement_delta', 0),
            }
        elif behaviour_class == 1:  # Nervous
            signals = {
                'hand_movement': features.get('hand_movement_delta', 0),
                'shoulder_tension': features.get('shoulder_tension_ratio', 0),
                'head_instability': abs(features.get('head_tilt_angle', 0)),
            }
        else:  # Distracted
            signals = {
                'poor_eye_contact': 100 - features.get('eye_contact_proxy', 0),
                'head_turned_away': abs(features.get('head_tilt_angle', 0)),
                'poor_posture': 100 - features.get('spine_alignment_score', 0),
            }
        
        dominant_signal = max(signals, key=signals.get)
        return dominant_signal
    
    def generate_summary(self, behaviour_class: int, confidence: float, 
                        features: Dict[str, float], analysis: Dict) -> str:
        """Generate human-readable summary"""
        
        behaviour_names = {
            0: 'Confident',
            1: 'Nervous',
            2: 'Distracted'
        }
        
        behaviour_name = behaviour_names[behaviour_class]
        
        if behaviour_class == 0:
            if analysis.get('avg_confidence', 0) > 0.8:
                summary = f"Candidate displayed excellent confidence throughout the interview with strong eye contact and upright posture. " \
                         f"Minimal hand movement and relaxed shoulders indicate composure and control."
            else:
                summary = f"Candidate showed overall confidence with good eye contact and posture, though with some minor fluctuations in composure."
        
        elif behaviour_class == 1:
            if analysis.get('nervous_ratio', 0) > 0.7:
                summary = f"Candidate exhibited signs of nervousness including frequent hand movements and shoulder tension. " \
                         f"Recommend stress management techniques and more mock interviews for practice."
            else:
                summary = f"Candidate showed occasional signs of nervousness but generally maintained composure. " \
                         f"Some coaching on handling interview anxiety could be beneficial."
        
        else:  # Distracted
            if analysis.get('distracted_ratio', 0) > 0.7:
                summary = f"Candidate appeared distracted, frequently looking away and showing poor eye contact. " \
                         f"Focus on active listening and maintaining engagement during interviews is needed."
            else:
                summary = f"Candidate showed some moments of distraction but generally maintained focus. " \
                         f"Improving concentration during high-pressure situations would be helpful."
        
        return summary


def get_behaviour_name(behaviour_class: int) -> str:
    """Get behaviour name from class"""
    names = {0: 'Confident', 1: 'Nervous', 2: 'Distracted'}
    return names.get(behaviour_class, 'Unknown')
