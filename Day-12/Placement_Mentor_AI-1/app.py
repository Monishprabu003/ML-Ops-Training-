"""
PlacementMentorAI - Flask Application
Phase 1: Placement Score Prediction (Random Forest)
Phase 2: Interview Behaviour Analysis (YOLO26-Pose + Behaviour Classifier)
"""

import os
import pickle
import traceback
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'cv_module', 'outputs')

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ─────────────────────────────────────────────
# Phase 1: Load Placement Prediction Model
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

placement_model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        placement_model = pickle.load(f)
    print(f"✅ Placement model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Could not load placement model: {e}")

FEATURES = [
    'CGPA', 'Python_Skill', 'Java_Skill', 'SQL_Skill', 'DSA_Skill',
    'Web_Development_Skill', 'Cloud_Skill', 'ML_Skill', 'Cybersecurity_Skill',
    'Aptitude_Score', 'Communication_Skill', 'Problem_Solving_Skill',
    'Confidence_Level', 'Projects_Count', 'Certifications_Count'
]

# ─────────────────────────────────────────────
# Phase 2: Lazy-load CV Analyzers (heavy models)
# ─────────────────────────────────────────────
_image_analyzer = None
_video_analyzer = None


def get_image_analyzer():
    """Lazy-load the image behaviour analyzer."""
    global _image_analyzer
    if _image_analyzer is None:
        from cv_module import ImageBehaviourAnalyzer
        custom_model = os.path.join(os.path.dirname(__file__), 'cv_module', 'models', 'best.pt')
        model_path = custom_model if os.path.exists(custom_model) else None
        _image_analyzer = ImageBehaviourAnalyzer(model_path=model_path)
        print("✅ Image Behaviour Analyzer loaded")
    return _image_analyzer


def get_video_analyzer():
    """Lazy-load the video behaviour analyzer."""
    global _video_analyzer
    if _video_analyzer is None:
        from cv_module import VideoBehaviourAnalyzer
        custom_model = os.path.join(os.path.dirname(__file__), 'cv_module', 'models', 'best.pt')
        model_path = custom_model if os.path.exists(custom_model) else None
        _video_analyzer = VideoBehaviourAnalyzer(model_path=model_path, fps_target=5)
        print("✅ Video Behaviour Analyzer loaded")
    return _video_analyzer


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def get_status(score):
    """Get placement status from score."""
    if score >= 90:
        return "Excellent – Highly Placed"
    elif score >= 80:
        return "Very Good – Strong Candidate"
    elif score >= 70:
        return "Good – Competitive"
    elif score >= 60:
        return "Average – Needs Improvement"
    else:
        return "Below Average – Focus on Skill Building"


def get_suggested_role(features_dict):
    """Suggest a role based on strongest skills."""
    skill_roles = {
        'Python_Skill': 'Python Developer / Data Scientist',
        'Java_Skill': 'Java Backend Developer',
        'SQL_Skill': 'Database Engineer / Data Analyst',
        'DSA_Skill': 'Software Engineer (SDE)',
        'Web_Development_Skill': 'Full Stack Web Developer',
        'Cloud_Skill': 'Cloud Engineer / DevOps',
        'ML_Skill': 'Machine Learning Engineer',
        'Cybersecurity_Skill': 'Cybersecurity Analyst',
    }
    tech_skills = {k: features_dict.get(k, 0) for k in skill_roles}
    best_skill = max(tech_skills, key=tech_skills.get)
    return skill_roles[best_skill]


def get_weak_areas(features_dict):
    """Identify weak skill areas (< 6 out of 10 or < 60 for aptitude)."""
    weak = []
    for key, val in features_dict.items():
        if key == 'CGPA':
            if val < 7.0:
                weak.append(f"CGPA ({val}) – Aim for 7.0+")
        elif key == 'Aptitude_Score':
            if val < 60:
                weak.append(f"Aptitude Score ({val}) – Practice aptitude tests")
        elif key in ('Projects_Count', 'Certifications_Count'):
            if val < 3:
                weak.append(f"{key.replace('_', ' ')} ({val}) – Build more projects/get certifications")
        else:
            if val < 6:
                weak.append(f"{key.replace('_', ' ')} ({val}/10) – Needs improvement")
    return weak if weak else ["No major weaknesses detected. Keep improving!"]


def get_recommendations(features_dict):
    """Generate learning recommendations based on weak areas."""
    recs = []
    if features_dict.get('Python_Skill', 10) < 6:
        recs.append("Complete a Python bootcamp on Coursera or freeCodeCamp")
    if features_dict.get('Java_Skill', 10) < 6:
        recs.append("Practice Java OOP through HackerRank challenges")
    if features_dict.get('SQL_Skill', 10) < 6:
        recs.append("Learn SQL on SQLZoo or LeetCode Database problems")
    if features_dict.get('DSA_Skill', 10) < 6:
        recs.append("Practice DSA daily on LeetCode — aim for 200+ problems")
    if features_dict.get('Web_Development_Skill', 10) < 6:
        recs.append("Build 2–3 full-stack projects with React/Flask")
    if features_dict.get('Cloud_Skill', 10) < 6:
        recs.append("Get AWS Cloud Practitioner or Azure Fundamentals certification")
    if features_dict.get('ML_Skill', 10) < 6:
        recs.append("Take Andrew Ng's ML course on Coursera")
    if features_dict.get('Cybersecurity_Skill', 10) < 6:
        recs.append("Start with TryHackMe or Hack The Box for cybersecurity")
    if features_dict.get('Communication_Skill', 10) < 6:
        recs.append("Join a public speaking club or practice mock interviews")
    if features_dict.get('Problem_Solving_Skill', 10) < 6:
        recs.append("Solve logic puzzles and participate in coding contests")
    if features_dict.get('Confidence_Level', 10) < 6:
        recs.append("Practice mock interviews and presentations to build confidence")
    if not recs:
        recs.append("Continue your excellent work! Focus on mock interviews and networking.")
    return recs


# ═══════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════

# ─────────────────────────────────────────────
# Phase 1 Routes
# ─────────────────────────────────────────────

@app.route('/')
def home():
    """Serve the placement score prediction form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Process placement score prediction."""
    if placement_model is None:
        return render_template('result.html',
                               prediction=0,
                               status="Model not loaded",
                               role="N/A",
                               weak_areas=["Placement model could not be loaded."],
                               recommendations=["Check that model.pkl exists."])

    try:
        # Collect form data
        features_dict = {}
        for feature in FEATURES:
            val = request.form.get(feature, 0)
            features_dict[feature] = float(val)

        # Create input array
        input_data = pd.DataFrame([features_dict])
        input_array = input_data[FEATURES]

        # Predict
        prediction = placement_model.predict(input_array)[0]
        prediction = round(float(prediction), 1)

        # Clamp to 0–100
        prediction = max(0, min(100, prediction))

        status = get_status(prediction)
        role = get_suggested_role(features_dict)
        weak_areas = get_weak_areas(features_dict)
        recommendations = get_recommendations(features_dict)

        return render_template('result.html',
                               prediction=prediction,
                               status=status,
                               role=role,
                               weak_areas=weak_areas,
                               recommendations=recommendations)
    except Exception as e:
        traceback.print_exc()
        return render_template('result.html',
                               prediction=0,
                               status="Error",
                               role="N/A",
                               weak_areas=[f"Error: {str(e)}"],
                               recommendations=["Please try again."])


# ─────────────────────────────────────────────
# Phase 2 Routes
# ─────────────────────────────────────────────

@app.route('/interview')
def interview():
    """Serve the interview behaviour analysis page."""
    placement_score = request.args.get('placement_score', None)
    return render_template('interview_analysis.html',
                           placement_score=placement_score)


@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze interview behaviour from an uploaded image."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid file type. Allowed: JPG, PNG'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Analyze
        analyzer = get_image_analyzer()
        result = analyzer.analyze_image(filepath)

        if result.get('error'):
            return jsonify(result), 400

        # Convert processed image path to serve via Flask
        if result.get('processed_image_path'):
            processed_name = os.path.basename(result['processed_image_path'])
            result['processed_image_path'] = url_for('serve_output',
                                                      filename=processed_name)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    """Analyze interview behaviour from an uploaded video."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({'error': 'Invalid file type. Allowed: MP4, AVI, MOV, MKV'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Analyze
        analyzer = get_video_analyzer()
        result = analyzer.analyze_video(filepath)

        if result.get('error'):
            return jsonify(result), 400

        # Convert processed video path to serve via Flask
        if result.get('processed_video_path'):
            processed_name = os.path.basename(result['processed_video_path'])
            result['processed_video_path'] = url_for('serve_output',
                                                      filename=processed_name)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/calculate-final-score', methods=['POST'])
def calculate_final_score():
    """
    Calculate the final placement readiness score.
    Formula: final_score = (placement_score * 0.7) + (behaviour_score * 0.3)
    """
    try:
        data = request.get_json()
        placement_score = float(data.get('placement_score', 0))
        behaviour_score = float(data.get('behaviour_score', 0))

        # Clamp values
        placement_score = max(0, min(100, placement_score))
        behaviour_score = max(0, min(100, behaviour_score))

        final_score = round((placement_score * 0.7) + (behaviour_score * 0.3), 1)

        return jsonify({
            'placement_score': placement_score,
            'behaviour_score': behaviour_score,
            'final_score': final_score,
            'formula': '(placement_score × 0.7) + (behaviour_score × 0.3)',
        })

    except Exception as e:
        return jsonify({'error': f'Calculation failed: {str(e)}'}), 400


# ─────────────────────────────────────────────
# File Serving Routes
# ─────────────────────────────────────────────

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve processed output files (images/videos)."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  PlacementMentorAI")
    print("  Phase 1: Placement Score Prediction")
    print("  Phase 2: Interview Behaviour Analysis")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
