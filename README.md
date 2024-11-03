# Heart Disease Detection

## Objective
The objective of this project is to build a system that can predict if a patient has heart disease based on their health vitals.

## Dataset
The dataset used in this project contains various health vitals of patients that are crucial for predicting the presence of heart disease. Key columns include patient identifiers, health measurements, and indicators of heart disease.

### Features:
- **age**: Age of the patient
- **sex**: Gender of the patient (male/female)
- **chest pain type**: Type of chest pain experienced
- **resting blood pressure**: Resting blood pressure in mm Hg
- **serum cholesterol**: Serum cholesterol in mg/dl
- **fasting blood sugar**: Fasting blood sugar (greater than 120 mg/dl)
- **resting electrocardiogram results**: Results of resting electrocardiogram
- **maximum heart rate achieved**: Maximum heart rate achieved by the patient
- **oldpeak = ST depression**: ST depression induced by exercise relative to rest
- **slope of the peak exercise ST segment**: Slope of the peak exercise ST segment
- **exercise induced angina**: Angina induced by exercise (yes/no)
- **class target**: Target variable indicating presence (1) or absence (0) of heart disease

## Approach

### Data Preprocessing:
- **Loading the Dataset**: The dataset is loaded using Pandas.
- **Handling Missing Values**: The dataset is checked for missing values and appropriate actions are taken.
- **Feature Engineering**: Categorical columns are converted to appropriate types for analysis, and numeric features are standardized using `StandardScaler`.

### Exploratory Data Analysis (EDA):
- Summary statistics are generated for numeric columns to understand the data distribution.
- A count plot of the target variable is created to visualize the distribution of heart disease cases.
- A correlation heatmap is generated to identify relationships between features.

### Modeling
- The dataset is split into training and testing sets using an 80-20 split.
- A Random Forest Classifier is instantiated and trained on the training set.
- The model is evaluated using accuracy, confusion matrix, and classification report metrics.

## Model Evaluation:
- **Accuracy**: The model achieves an accuracy of **94.96%**.
- **Confusion Matrix**:
              precision    recall  f1-score   support

          0       0.95      0.93      0.94       107
          1       0.95      0.96      0.95       131

  accuracy                           0.95       238
 macro avg       0.95      0.95      0.95       238


## Model Saving
The trained Random Forest model is saved as `heart_disease_model.joblib` using joblib for easy deployment and future use.

## Running the Heart Disease Detection System
1. Clone the repository and ensure all dependencies are installed.
2. Load the model file `heart_disease_model.joblib`.
3. Use the model to predict heart disease on new patient data by preprocessing the input features as done during training.

### Deployment
A Gradio app is included in the project, with the main application script named `app.py`. This app loads the trained model and runs inference, allowing users to input health vitals and receive predictions on the likelihood of heart disease.

## Conclusion
This project provides a comprehensive framework for detecting heart disease using machine learning, showcasing techniques from data preprocessing and exploratory analysis to model evaluation and deployment.
