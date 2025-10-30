# --------------------------------------------------
# 🚢 Titanic Survival Prediction App using Streamlit
# --------------------------------------------------

import streamlit as st
import pandas as pd
import pickle

# ------------------------------
# Load Trained Model and Scaler
# ------------------------------
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------------------
# Streamlit App Setup
# ------------------------------
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict the survival probability using the Logistic Regression model.")

# ------------------------------
# Input Fields
# ------------------------------
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 32.0)
hascabin = st.selectbox("Has Cabin Info (1 = Yes, 0 = No)", [0, 1])
embarked_q = st.selectbox("Embarked_Q (1 = Yes, 0 = No)", [0, 1])
embarked_s = st.selectbox("Embarked_S (1 = Yes, 0 = No)", [0, 1])
title_miss = st.selectbox("Title_Miss (1 = Yes, 0 = No)", [0, 1])
title_mr = st.selectbox("Title_Mr (1 = Yes, 0 = No)", [0, 1])
title_mrs = st.selectbox("Title_Mrs (1 = Yes, 0 = No)", [0, 1])
title_other = st.selectbox("Title_Other (1 = Yes, 0 = No)", [0, 1])

# ------------------------------
# Prepare Input Data
# ------------------------------
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [1 if sex == 'female' else 0],  # same mapping as training
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'HasCabin': [hascabin],
    'Embarked_Q': [embarked_q],
    'Embarked_S': [embarked_s],
    'Title_Miss': [title_miss],
    'Title_Mr': [title_mr],
    'Title_Mrs': [title_mrs],
    'Title_Other': [title_other]
})

# Ensure correct feature order
expected_features = ['Pclass','Sex','Age','SibSp','Parch','Fare',
                     'HasCabin','Embarked_Q','Embarked_S',
                     'Title_Miss','Title_Mr','Title_Mrs','Title_Other']
input_data = input_data[expected_features]

# ------------------------------
# Scale Input Data (Very Important!)
# ------------------------------
input_scaled = scaler.transform(input_data)

# ------------------------------
# Make Prediction
# ------------------------------
if st.button(" Predict Survival"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"Passenger likely **SURVIVED** with probability {probability:.2f}")
    else:
        st.error(f" Passenger likely **DID NOT SURVIVE** with probability {(1 - probability):.2f}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Developed as part of the Titanic ML Assignment using Logistic Regression and Streamlit.")
