# ❤️ Heart Disease Prediction

A Machine Learning based web application that predicts the risk of heart disease using patient health-related features. The application is built with Python, Scikit-learn, XGBoost and Streamlit.

## 🚀 Project Overview

Heart disease is one of the major health concerns worldwide. This project uses Machine Learning classification algorithms to analyze patient health data and predict the possibility of heart disease.

The trained model is integrated into an interactive Streamlit dashboard where users can enter patient information and receive a prediction along with the estimated risk probability.

## 🧠 Machine Learning Models

The following classification algorithms were implemented and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Neural Network
- Decision Tree
- Random Forest
- XGBoost
- Naive Bayes

After comparing the models, the selected model was saved as `best_model.pkl` and used for the Streamlit application.

## 📊 Features

The application uses the following patient-related features:

- Gender
- Age
- Education Level
- Current Smoker
- Cigarettes Per Day
- Blood Pressure Medication
- Previous Stroke
- Hypertension
- Diabetes
- Total Cholesterol
- Systolic Blood Pressure
- Diastolic Blood Pressure
- BMI
- Heart Rate
- Glucose

## 💻 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib
- Seaborn
- Streamlit

