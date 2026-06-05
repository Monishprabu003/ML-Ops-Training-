"""
Computer Vision Module for Interview Behaviour Analysis
"""

from .inference.image_predict import ImageBehaviourAnalyzer
from .inference.video_predict import VideoBehaviourAnalyzer

__all__ = ['ImageBehaviourAnalyzer', 'VideoBehaviourAnalyzer']
