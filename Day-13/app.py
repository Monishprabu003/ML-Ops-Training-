from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
from werkzeug.utils import secure_filename

# Import CV modules for behaviour analysis
from cv_module.inference.image_predict import ImageBehaviourAnalyzer
from cv_module.inference.video_predict import VideoBehaviourAnalyzer

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('cv_module/outputs', exist_ok=True)

# Load the placement prediction model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize CV analysers (lazy load)
image_analyzer = None
video_analyzer = None

def get_image_analyzer():
    global image_analyzer
    if image_analyzer is None:
        image_analyzer = ImageBehaviourAnalyzer()
    return image_analyzer

def get_video_analyzer():
    global video_analyzer
    if video_analyzer is None:
        video_analyzer = VideoBehaviourAnalyzer()
    return video_analyzer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if model is None:
            return "Model not loaded. Please ensure model.pkl is in the root directory.", 500
            
        # Retrieve form data
        features = [
            'CGPA', 'Python_Skill', 'Java_Skill', 'SQL_Skill', 'DSA_Skill',
            'Web_Development_Skill', 'Cloud_Skill', 'ML_Skill', 'Cybersecurity_Skill',
            'Aptitude_Score', 'Communication_Skill', 'Problem_Solving_Skill',
            'Confidence_Level', 'Projects_Count', 'Certifications_Count'
        ]
        
        input_data = []
        for feature in features:
            try:
                # Convert input to float, default to 0 if not provided or invalid
                val = float(request.form[feature])
            except ValueError:
                val = 0.0
            input_data.append(val)
        
        # Create a DataFrame to pass to the model (in case model expects feature names)
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Make prediction
        prediction_score = model.predict(input_df)[0]
        prediction_score = max(0, min(100, prediction_score)) # Keep within 0-100 range
        
        # Define logic for additional outputs based on score and inputs
        
        # Placement Status
        if prediction_score >= 80:
            placement_status = "Highly Likely to get Placed"
        elif prediction_score >= 60:
            placement_status = "Likely to get Placed (Needs some improvement)"
        else:
            placement_status = "Needs Significant Improvement"
            
        # Suggested Role (basic logic based on highest skills)
        skill_scores = {
            'Software Developer': (input_data[1] + input_data[2] + input_data[4]) / 3.0, # Python + Java + DSA
            'Data Scientist / ML Engineer': (input_data[1] + input_data[3] + input_data[7]) / 3.0, # Python + SQL + ML
            'Web Developer': (input_data[5] + input_data[3]) / 2.0, # Web Dev + SQL
            'Cloud Engineer': (input_data[6] + input_data[3]) / 2.0, # Cloud + SQL
            'Security Analyst': (input_data[8] + input_data[1]) / 2.0, # Cybersecurity + Python
        }
        suggested_role = max(skill_scores, key=skill_scores.get)
        
        # Weak Areas (skills < threshold)
        skill_names = [
            'Python', 'Java', 'SQL', 'DSA', 'Web Development', 
            'Cloud', 'Machine Learning', 'Cybersecurity', 
            'Aptitude', 'Communication', 'Problem Solving', 'Confidence'
        ]
        skill_values = input_data[1:13] # Skills 1 through 12
        weak_areas = []
        
        for name, val in zip(skill_names, skill_values):
            if name == 'Aptitude':
                if val < 50:  # Assuming Aptitude is 1-100 scale
                    weak_areas.append(name)
            else:
                if val < 6:  # Assuming other skills are 1-10 scale
                    weak_areas.append(name)
                
        if not weak_areas:
            weak_areas = ["None (All skills are at a good level!)"]
            
        # Recommended Learning
        recommendations = {
            'Python': 'Practice coding on LeetCode or HackerRank. Take an advanced Python concepts course.',
            'Java': 'Build Object-Oriented projects. Understand JVM internals and Spring Boot.',
            'SQL': 'Practice writing complex queries, joins, and database design. Try HackerRank SQL.',
            'DSA': 'Focus on Data Structures & Algorithms. Practice 1-2 problems daily on LeetCode.',
            'Web Development': 'Build a full-stack project using React or Angular and Node.js/Django.',
            'Cloud': 'Learn AWS/Azure fundamentals. Try deploying your projects on the cloud.',
            'Machine Learning': 'Work on end-to-end ML projects (Kaggle). Understand the math behind algorithms.',
            'Cybersecurity': 'Learn about network security, cryptography, and ethical hacking basics.',
            'Aptitude': 'Practice quantitative aptitude and logical reasoning questions daily.',
            'Communication': 'Participate in mock interviews, group discussions, and present your projects.',
            'Problem Solving': 'Solve puzzles and participate in coding contests to improve algorithmic thinking.',
            'Confidence': 'Do mock interviews with peers. Build more projects to gain practical confidence.'
        }
        
        learning_recs = []
        for area in weak_areas:
            if area in recommendations:
                learning_recs.append(f"{area}: {recommendations[area]}")
                
        if len(weak_areas) == 1 and weak_areas[0] == "None (All skills are at a good level!)":
            learning_recs = ["Keep building projects, participating in hackathons, and practicing interviews!"]
        elif not learning_recs:
            learning_recs = ["Focus on building projects and practical skills."]

        return render_template('result.html', 
                               prediction=round(prediction_score, 2),
                               status=placement_status,
                               role=suggested_role,
                               weak_areas=weak_areas,
                               recommendations=learning_recs)

@app.route('/interview-analysis')
def interview_analysis():
    """Page for interview behaviour analysis"""
    return render_template('interview_analysis.html')


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze interview behaviour from image"""
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid file type. Allowed: JPG, PNG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_')
        filepath = os.path.join(UPLOAD_FOLDER, timestamp + filename)
        file.save(filepath)
        
        # Analyze image
        analyzer = get_image_analyzer()
        result = analyzer.analyze_image(filepath)
        
        # Clean up upload
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    """Analyze interview behaviour from video"""
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({'error': 'Invalid file type. Allowed: MP4, AVI, MOV, MKV'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_')
        filepath = os.path.join(UPLOAD_FOLDER, timestamp + filename)
        file.save(filepath)
        
        # Analyze video
        analyzer = get_video_analyzer()
        result = analyzer.analyze_video(filepath, max_frames=300)  # Limit to 300 frames for performance
        
        # Clean up upload
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/combined-analysis', methods=['POST'])
def combined_analysis():
    """Combine placement score and behaviour score"""
    
    try:
        data = request.get_json()
        
        placement_score = float(data.get('placement_score', 0))
        behaviour_score = float(data.get('behaviour_score', 0))
        
        # Calculate final score: 70% placement + 30% behaviour
        final_score = (placement_score * 0.7) + (behaviour_score * 0.3)
        
        return jsonify({
            'placement_score': placement_score,
            'behaviour_score': behaviour_score,
            'final_score': round(final_score, 2),
            'placement_weight': 70,
            'behaviour_weight': 30,
        })
    
    except Exception as e:
        print(f"Error in combined analysis: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
