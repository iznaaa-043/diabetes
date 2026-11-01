import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Diabetes Progression Prediction", page_icon="🩺", layout="centered")

model_path = "best_diabetes_model.joblib"

if not os.path.exists(model_path):
    diabetes = load_diabetes(as_frame=True)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))])
    pipe.fit(diabetes.data, diabetes.target)
    joblib.dump(pipe, model_path)

model = joblib.load(model_path)

st.title("Diabetes Progression Predictor")
st.write("Enter the 10 standardized physiological features to predict diabetes disease progression.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", value=0.0, step=0.01, format="%.4f")
    sex = st.number_input("Sex", value=0.0, step=0.01, format="%.4f")
    bmi = st.number_input("BMI (Body Mass Index)", value=0.0, step=0.01, format="%.4f")
    bp = st.number_input("BP (Average Blood Pressure)", value=0.0, step=0.01, format="%.4f")
    s1 = st.number_input("S1 (Total Serum Cholesterol)", value=0.0, step=0.01, format="%.4f")

with col2:
    s2 = st.number_input("S2 (Low-Density Lipoproteins)", value=0.0, step=0.01, format="%.4f")
    s3 = st.number_input("S3 (High-Density Lipoproteins)", value=0.0, step=0.01, format="%.4f")
    s4 = st.number_input("S4 (Total Cholesterol / HDL)", value=0.0, step=0.01, format="%.4f")
    s5 = st.number_input("S5 (Log of Serum Triglycerides)", value=0.0, step=0.01, format="%.4f")
    s6 = st.number_input("S6 (Blood Sugar Level)", value=0.0, step=0.01, format="%.4f")

if st.button("Predict"):
    features = pd.DataFrame(
        [[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]],
        columns=["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    )
    prediction = model.predict(features)[0]
    st.success(f"Predicted Disease Progression Score: {prediction:.2f}")
