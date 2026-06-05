"""
Pose Estimation Utilities
Extracts and processes COCO keypoints from YOLO poses
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional

# COCO Keypoints (17 points)
COCO_KEYPOINTS = {
    0: 'Nose',
    1: 'Left Eye',
    2: 'Right Eye',
    3: 'Left Ear',
    4: 'Right Ear',
    5: 'Left Shoulder',
    6: 'Right Shoulder',
    7: 'Left Elbow',
    8: 'Right Elbow',
    9: 'Left Wrist',
    10: 'Right Wrist',
    11: 'Left Hip',
    12: 'Right Hip',
    13: 'Left Knee',
    14: 'Right Knee',
    15: 'Left Ankle',
    16: 'Right Ankle'
}


class PoseProcessor:
    """Process YOLO pose keypoints for behaviour analysis"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.keypoint_history = []  # For temporal smoothing
        
    def extract_keypoints(self, pose_results) -> Optional[np.ndarray]:
        """
        Extract 17 COCO keypoints from YOLO pose results
        Returns: (17, 3) array with [x, y, confidence] for each keypoint
        """
        try:
            if hasattr(pose_results, 'keypoints'):
                keypoints = pose_results.keypoints
                if keypoints is not None and len(keypoints) > 0:
                    # Get first person's keypoints (17, 3)
                    # Convert from PyTorch tensors to numpy for YOLO26 compatibility
                    raw_kpts = keypoints[0].xy[0] if len(keypoints[0].xy) > 0 else None
                    raw_conf = keypoints[0].conf[0] if hasattr(keypoints[0], 'conf') and len(keypoints[0].conf) > 0 else None
                    
                    # Ensure numpy arrays (handles both tensor and array inputs)
                    kpts = raw_kpts.cpu().numpy() if hasattr(raw_kpts, 'cpu') else np.array(raw_kpts) if raw_kpts is not None else None
                    conf = raw_conf.cpu().numpy() if hasattr(raw_conf, 'cpu') else np.array(raw_conf) if raw_conf is not None else None
                    
                    if kpts is not None:
                        if conf is not None:
                            # Combine coordinates and confidence
                            keypoints_with_conf = np.hstack([kpts, conf.reshape(-1, 1)])
                        else:
                            # If no confidence scores, assume all are confident
                            keypoints_with_conf = np.hstack([kpts, np.ones((len(kpts), 1))])
                        
                        return keypoints_with_conf
        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            return None
        
        return None
    
    def filter_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Filter low-confidence keypoints"""
        filtered = keypoints.copy()
        filtered[filtered[:, 2] < self.confidence_threshold] = 0
        return filtered
    
    def apply_temporal_smoothing(self, keypoints: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply moving average smoothing across frames"""
        self.keypoint_history.append(keypoints)
        
        # Keep only last window_size frames
        if len(self.keypoint_history) > window_size:
            self.keypoint_history.pop(0)
        
        # Average across frames
        smoothed = np.mean(self.keypoint_history, axis=0)
        return smoothed
    
    def calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3
        Returns angle in degrees
        """
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def calculate_head_tilt_angle(self, keypoints: np.ndarray) -> float:
        """
        Calculate head tilt angle using eyes and ears
        Positive = right tilt, Negative = left tilt
        """
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        
        if left_eye[2] < self.confidence_threshold or right_eye[2] < self.confidence_threshold:
            return 0.0
        
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    
    def calculate_spine_alignment_score(self, keypoints: np.ndarray) -> float:
        """
        Calculate spine alignment (0-100 score)
        Based on shoulder-hip alignment
        """
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Check confidence
        if any(kpt[2] < self.confidence_threshold for kpt in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return 50.0
        
        # Calculate vertical alignment
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        hip_mid_x = (left_hip[0] + right_hip[0]) / 2
        
        # Calculate deviation from perfect alignment (lower is better)
        deviation = abs(shoulder_mid_x - hip_mid_x)
        
        # Normalize to 0-100 scale (assuming max acceptable deviation is 50 pixels)
        alignment_score = max(0, 100 - (deviation / 50 * 100))
        return alignment_score
    
    def calculate_hand_movement_delta(self, keypoints: np.ndarray, prev_keypoints: Optional[np.ndarray] = None) -> float:
        """
        Calculate hand movement magnitude
        Returns movement score (0-100)
        """
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        
        if prev_keypoints is None:
            return 0.0
        
        prev_left_wrist = prev_keypoints[9]
        prev_right_wrist = prev_keypoints[10]
        
        if any(kpt[2] < self.confidence_threshold for kpt in [left_wrist, right_wrist]):
            return 0.0
        
        # Calculate movement for each hand
        left_movement = self.calculate_distance(left_wrist[:2], prev_left_wrist[:2])
        right_movement = self.calculate_distance(right_wrist[:2], prev_right_wrist[:2])
        
        total_movement = left_movement + right_movement
        
        # Normalize to 0-100 scale
        movement_score = min(100, total_movement / 2)
        return movement_score
    
    def calculate_eye_contact_proxy(self, keypoints: np.ndarray, frame_height: int, frame_width: int) -> float:
        """
        Estimate eye contact proxy (0-100)
        Based on nose position relative to center (camera is typically at center)
        """
        nose = keypoints[0]
        
        if nose[2] < self.confidence_threshold:
            return 50.0
        
        # Check if looking towards camera (nose near center)
        center_x = frame_width / 2
        center_y = frame_height / 2
        
        # Calculate deviation from center
        deviation_x = abs(nose[0] - center_x) / (frame_width / 2)
        deviation_y = abs(nose[1] - center_y) / (frame_height / 2)
        
        total_deviation = (deviation_x + deviation_y) / 2
        
        # Normalize to 0-100 (lower deviation = higher score)
        eye_contact = max(0, 100 - (total_deviation * 100))
        return eye_contact
    
    def calculate_shoulder_tension_ratio(self, keypoints: np.ndarray) -> float:
        """
        Calculate shoulder tension (0-100)
        Based on shoulder height and position
        Raised shoulders = higher tension
        """
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        if any(kpt[2] < self.confidence_threshold for kpt in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return 50.0
        
        # Calculate shoulder-hip vertical distance
        left_distance = left_hip[1] - left_shoulder[1]
        right_distance = right_hip[1] - right_shoulder[1]
        
        avg_distance = (left_distance + right_distance) / 2
        
        # Normalize: closer together (smaller distance) = higher tension
        # Assuming normal distance is around 200 pixels
        tension = max(0, 100 - (avg_distance / 200 * 100))
        return min(100, tension)
    
    def extract_behaviour_features(self, keypoints: np.ndarray, 
                                   prev_keypoints: Optional[np.ndarray] = None,
                                   frame_height: int = 480, 
                                   frame_width: int = 640) -> Dict[str, float]:
        """
        Extract all behaviour features from keypoints
        """
        features = {
            'head_tilt_angle': self.calculate_head_tilt_angle(keypoints),
            'spine_alignment_score': self.calculate_spine_alignment_score(keypoints),
            'hand_movement_delta': self.calculate_hand_movement_delta(keypoints, prev_keypoints),
            'eye_contact_proxy': self.calculate_eye_contact_proxy(keypoints, frame_height, frame_width),
            'shoulder_tension_ratio': self.calculate_shoulder_tension_ratio(keypoints),
        }
        
        return features


def visualize_keypoints(frame: np.ndarray, keypoints: np.ndarray, skeleton_connections: List[Tuple[int, int]] = None) -> np.ndarray:
    """
    Draw keypoints and skeleton on frame
    """
    if skeleton_connections is None:
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Body
        ]
    
    output_frame = frame.copy()
    
    # Draw skeleton
    for connection in skeleton_connections:
        p1_idx, p2_idx = connection
        if keypoints[p1_idx][2] > 0.5 and keypoints[p2_idx][2] > 0.5:
            p1 = tuple(map(int, keypoints[p1_idx][:2]))
            p2 = tuple(map(int, keypoints[p2_idx][:2]))
            cv2.line(output_frame, p1, p2, (0, 255, 0), 2)
    
    # Draw keypoints
    for idx, kpt in enumerate(keypoints):
        if kpt[2] > 0.5:
            pos = tuple(map(int, kpt[:2]))
            cv2.circle(output_frame, pos, 5, (0, 0, 255), -1)
            cv2.putText(output_frame, str(idx), (pos[0] + 5, pos[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return output_frame
