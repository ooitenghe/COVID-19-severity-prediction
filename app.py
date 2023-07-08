import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained Random Forest model
model_path = "random_forest.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Function to preprocess user input
def preprocess_input(age, gender, sore_throat, fever, flu, loss_of_taste, loss_of_smell, cough, breathing_difficulties, diarrhea, other_symptoms):
    data = {
        "Age": [age],
        "Gender": [gender],
        "SoreThroat": [sore_throat],
        "Fever": [fever],
        "Flu": [flu],
        "LossOfTaste": [loss_of_taste],
        "LossOfSmell": [loss_of_smell],
        "Cough": [cough],
        "BreathingDifficulties": [breathing_difficulties],
        "Diarrhea": [diarrhea],
        "OtherSymptoms": [other_symptoms]
    }
    input_df = pd.DataFrame(data)
    return input_df

# Function to predict severity
def predict_severity(input_df):
    severity_mapping = {0: "Mild", 1: "Moderate", 2: "Severe"}
    prediction = model.predict(input_df)[0]
    severity = severity_mapping[prediction]
    return severity

# Streamlit web app
def main():
    # App title
    st.title("COVID-19 Severity Prediction")

    # User input section
    st.sidebar.title("User Input")
    age = st.sidebar.number_input("Age", min_value=0, max_value=150, value=30)
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    sore_throat = st.sidebar.selectbox("Presence of Sore Throat", ["No", "Yes"])
    fever = st.sidebar.selectbox("Presence of Fever", ["No", "Yes"])
    flu = st.sidebar.selectbox("Presence of Flu", ["No", "Yes"])
    loss_of_taste = st.sidebar.selectbox("Presence of Loss of Taste", ["No", "Yes"])
    loss_of_smell = st.sidebar.selectbox("Presence of Loss of Smell", ["No", "Yes"])
    cough = st.sidebar.selectbox("Presence of Cough", ["No", "Yes"])
    breathing_difficulties = st.sidebar.selectbox("Presence of Breathing Difficulties", ["No", "Yes"])
    diarrhea = st.sidebar.selectbox("Presence of Diarrhea", ["No", "Yes"])
    other_symptoms = st.sidebar.selectbox("Presence of Other Symptoms", ["No", "Yes"])

    input_df = preprocess_input(age, gender, sore_throat, fever, flu, loss_of_taste, loss_of_smell, cough, breathing_difficulties, diarrhea, other_symptoms)

    # Predict severity and display result
    if st.button("Predict"):
        severity = predict_severity(input_df)
        st.success(f"The predicted severity of the patient's condition is: {severity}.")

# Run the app
if __name__ == "__main__":
    main()
