import gradio as gr
import joblib
import numpy as np

# Load the saved Random Forest model and the StandardScaler
model = joblib.load('/content/heart_disease_model.joblib')
scaler = joblib.load('/content/scaler.joblib')  # Load the saved StandardScaler

# Function to make predictions with preprocessing
def predict_heart_disease(age, sex, chest_pain_type, resting_bp, cholesterol,
                          fasting_blood_sugar, resting_ecg, max_heart_rate,
                          exercise_angina, oldpeak, st_slope):
    # Collecting the input features into a numpy array
    features = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol,
                          fasting_blood_sugar, resting_ecg, max_heart_rate,
                          exercise_angina, oldpeak, st_slope]])
    
    # Apply the same scaling that was used during training
    scaled_features = scaler.transform(features)

    # Making the prediction using the scaled input
    prediction = model.predict(scaled_features)[0]

    # Mapping the prediction to the output label
    return "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

# Building the Gradio interface
interface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Slider(0, 120, step=1, label="Age"),
        gr.Radio(choices=[0, 1], label="Sex (0 = Female, 1 = Male)"),
        gr.Dropdown(choices=[1, 2, 3, 4], label="Chest Pain Type"),
        gr.Slider(70, 200, step=1, label="Resting Blood Pressure (mm Hg)"),
        gr.Slider(0, 600, step=1, label="Serum Cholesterol (mg/dl)"),
        gr.Radio(choices=[0, 1], label="Fasting Blood Sugar (>120 mg/dl)"),
        gr.Dropdown(choices=[0, 1, 2], label="Resting ECG Results"),
        gr.Slider(60, 220, step=1, label="Maximum Heart Rate Achieved"),
        gr.Radio(choices=[0, 1], label="Exercise Induced Angina (0 = No, 1 = Yes)"),
        gr.Slider(0.0, 6.0, step=0.1, label="Oldpeak = ST Depression"),
        gr.Dropdown(choices=[1, 2, 3], label="Slope of Peak Exercise ST Segment")
    ],
    outputs="text",
    title="Heart Disease Prediction",
    description="Enter patient data to predict the presence of heart disease."
)

# Launching the Gradio interface
interface.launch(debug=True)
