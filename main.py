import pandas as pd
from joblib import load

# ==============================
# Load trained model
# ==============================
model = load("model.joblib")   # pastikan nama file sesuai

# ==============================
# Column order (MUST match training data)
# ==============================
columns = [
    'Unnamed: 0',
    'gender',
    'age',
    'hypertension',
    'heart_disease',
    'smoking_history',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level'
]

# ==============================
# Sample input (EDIT to test)
# ==============================
input_data = [
    1,              # Unnamed: 0
    'Male',         # gender
    35,             # age
    1,              # hypertension
    0,              # heart_disease
    'never',        # smoking_history
    28.5,           # bmi
    5.7,            # HbA1c_level
    140             # blood_glucose_level
]

# ==============================
# Convert to DataFrame
# ==============================
df = pd.DataFrame([input_data], columns=columns)

# ==============================
# Predict
# ==============================
prediction = model.predict(df)
probability = model.predict_proba(df) if hasattr(model, "predict_proba") else None

# ==============================
# Print result
# ==============================
print("\n=== Prediction Result ===")
print("Prediction:", int(prediction[0]))

if probability is not None:
    print("Probability:", probability[0])
