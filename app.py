import streamlit as st
import pandas as pd
import numpy as np
import joblib

# IMPORTANT: These imports are REQUIRED for loading pickle files
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Student Math Score Predictor", layout="centered")

st.title("📊 Student Math Score Predictor")

# Load saved artifacts
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("best_model.pkl")

st.write("Enter student details to predict math score")

gender = st.selectbox("Gender", ["male", "female"])
race_ethnicity = st.selectbox(
    "Race/Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)
parental_education = st.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

if st.button("Predict Math Score"):
    input_df = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race_ethnicity],
        "parental level of education": [parental_education],
        "lunch": [lunch],
        "test preparation course": [test_prep],
        "reading score": [reading_score],
        "writing score": [writing_score]
    })

    transformed_data = preprocessor.transform(input_df)
    prediction = model.predict(transformed_data)

    st.success(f"✅ Predicted Math Score: {int(prediction[0])}")