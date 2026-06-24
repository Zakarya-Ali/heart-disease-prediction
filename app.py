# ============================================================
# HEART DISEASE PREDICTION STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import pickle


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="centered"
)


# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    """
    This function loads the trained .pkl model.
    Make sure the .pkl file is in the same GitHub folder as app.py.
    """
    with open("heart_disease_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()


# ------------------------------------------------------------
# App title and description
# ------------------------------------------------------------

st.title("❤️ Heart Disease Prediction App")

st.write(
    "This app predicts whether a person is likely to have heart disease "
    "based on medical input values."
)

st.warning(
    "This app is for machine learning practice only. "
    "It should not be used as a real medical diagnosis tool."
)


# ------------------------------------------------------------
# Sidebar information
# ------------------------------------------------------------

st.sidebar.title("About This App")

st.sidebar.write(
    """
    This app uses a trained machine learning model saved as a `.pkl` file.

    Target meaning:

    - 0 = No Heart Disease
    - 1 = Heart Disease
    """
)


# ------------------------------------------------------------
# User input section
# ------------------------------------------------------------

st.header("Enter Patient Details")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=52
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Type 0",
        1: "Type 1",
        2: "Type 2",
        3: "Type 3"
    }[x]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=125
)

chol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=700,
    value=212
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

restecg = st.selectbox(
    "Resting ECG Result",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Type 0",
        1: "Type 1",
        2: "Type 2"
    }[x]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=168
)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Type 0",
        1: "Type 1",
        2: "Type 2"
    }[x]
)

ca = st.selectbox(
    "Number of Major Vessels",
    options=[0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Type 0",
        1: "Type 1",
        2: "Type 2",
        3: "Type 3"
    }[x]
)


# ------------------------------------------------------------
# Convert input into DataFrame
# ------------------------------------------------------------

# IMPORTANT:
# The column order must be exactly the same as the order used during training.

input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})


# ------------------------------------------------------------
# Display input data
# ------------------------------------------------------------

st.subheader("Patient Input Data")
st.dataframe(input_data)


# ------------------------------------------------------------
# Make prediction
# ------------------------------------------------------------

if st.button("Predict"):

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Prediction: Heart Disease")
    else:
        st.success("Prediction: No Heart Disease")

    # Show probability only if the model supports predict_proba
    if hasattr(model, "predict_proba"):
        prediction_probability = model.predict_proba(input_data)

        probability_no_disease = prediction_probability[0][0]
        probability_disease = prediction_probability[0][1]

        st.write("### Prediction Probability")

        st.write(f"Probability of No Heart Disease: **{probability_no_disease:.2%}**")
        st.write(f"Probability of Heart Disease: **{probability_disease:.2%}**")

        st.progress(float(probability_disease))


# ------------------------------------------------------------
# Column explanation
# ------------------------------------------------------------

st.markdown("---")

st.subheader("Column Explanation")

st.write(
    """
    - **age**: Age of the person  
    - **sex**: Sex of the person  
    - **cp**: Chest pain type  
    - **trestbps**: Resting blood pressure  
    - **chol**: Cholesterol level  
    - **fbs**: Fasting blood sugar  
    - **restecg**: Resting ECG result  
    - **thalach**: Maximum heart rate achieved  
    - **exang**: Exercise induced angina  
    - **oldpeak**: ST depression  
    - **slope**: Slope of ST segment  
    - **ca**: Number of major vessels  
    - **thal**: Thalassemia value  
    """
)

st.caption("Created using Python, Scikit-learn, and Streamlit.")
