# Student-Math-Score-Predictor
A web application that predicts students’ math scores based on demographic and academic features using machine learning models. Built with Python, Scikit-Learn, XGBoost, CatBoost, and deployed via Streamlit.


This project predicts the **Math Score** of students based on demographic and academic features using machine learning models. The app is built with **Python**, **Scikit-Learn**, **XGBoost**, **CatBoost**, and **Streamlit** for interactive web deployment.

---

## Features

- Predict math scores from features such as:
  - Gender
  - Race/Ethnicity
  - Parental level of education
  - Lunch type
  - Test preparation course
  - Reading and Writing scores
- Trained multiple ML models and selected the best-performing model automatically
- Preprocessing handled with a **ColumnTransformer** for scaling and encoding
- Deployable as an interactive web app using Streamlit

---

## How it Works

1. **Data Preprocessing**:  
   Numeric features are scaled with `StandardScaler`.  
   Categorical features are one-hot encoded with `OneHotEncoder`.  

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_features = ['reading score', 'writing score']
cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features)
])

