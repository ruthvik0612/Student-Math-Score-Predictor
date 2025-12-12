import streamlit as st
import pandas as pd
import joblib

st.title("Student Math Score Predictor")

preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('best_model.pkl')

gender = st.selectbox('Gender', ['female', 'male'])
race = st.selectbox('Race/Ethnicity', ['group A','group B','group C','group D','group E'])
parent_edu = st.selectbox('Parental Level of Education', ["some high school", "high school", "some college",
                                                        "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.selectbox('Lunch', ['standard', 'free/reduced'])
test_prep = st.selectbox('Test Preparation Course', ['none', 'completed'])
reading_score = st.number_input('Reading Score', min_value=0, max_value=100, value=70)
writing_score = st.number_input('Writing Score', min_value=0, max_value=100, value=70)

if st.button('Predict Math Score'):
    input_df = pd.DataFrame([{
        'gender': gender,
        'race/ethnicity': race,
        'parental level of education': parent_edu,
        'lunch': lunch,
        'test preparation course': test_prep,
        'reading score': reading_score,
        'writing score': writing_score
    }])
    
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)
    
    st.success(f'Predicted Math Score: {prediction[0]:.2f}')
