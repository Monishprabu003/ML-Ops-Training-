import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv('placementmentor_dataset.csv')

# Assuming the dataset has columns matching the features list and a target column
features = [
    'CGPA', 'Python_Skill', 'Java_Skill', 'SQL_Skill', 'DSA_Skill',
    'Web_Development_Skill', 'Cloud_Skill', 'ML_Skill', 'Cybersecurity_Skill',
    'Aptitude_Score', 'Communication_Skill', 'Problem_Solving_Skill',
    'Confidence_Level', 'Projects_Count', 'Certifications_Count'
]

# We need to find the target column. Let's assume it's 'Placement_Score' or similar.
# Let's check the columns
print("Columns in dataset:", df.columns.tolist())

target_col = 'Placement_Score'
if target_col not in df.columns:
    # If not exact match, try to find it
    for col in df.columns:
        if 'score' in col.lower() and 'placement' in col.lower():
            target_col = col
            break

print(f"Using target column: {target_col}")

X = df[features]
y = df[target_col]

# Train a RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model successfully trained and saved as model.pkl")
