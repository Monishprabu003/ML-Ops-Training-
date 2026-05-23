from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

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

if __name__ == '__main__':
    app.run(debug=True, port=5001)
