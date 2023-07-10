import streamlit as st
import pandas as pd
import numpy as np
import pickle

DATASET_PATH = ""
MODEL_PATH = "./model/random_forest.pkl"

def main():
    @st.cache(persist=True)
    def load_dataset() -> pd.DataFrame:
        covid19_df = pd.read_csv(DATASET_PATH, encoding="UTF-8")
        covid19_df = pd.DataFrame(np.sort(covid19_df.values, axis=0),
                                index=covid19_df.index,
                                columns=covid19_df.columns)
        return covid19_df

    def user_input_features() -> pd.DataFrame:
        age_cat = st.sidebar.selectbox("Age category", options=["1","2","3","4","5"])
        gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
        sore_throat = st.sidebar.selectbox("Sore Throat", options=["No", "Yes"])
        fever = st.sidebar.selectbox("Fever", options=["No", "Yes"])
        flu = st.sidebar.selectbox("Flu", options=["No", "Yes"])
        loss_of_taste = st.sidebar.selectbox("Loss of Taste", options=["No", "Yes"])
        loss_of_smell = st.sidebar.selectbox("Loss of Smell", options=["No", "Yes"])
        cough = st.sidebar.selectbox("Cough", options=["No", "Yes"])
        breathing_difficulties = st.sidebar.selectbox("Breathing Difficulties", options=["No", "Yes"])
        diarrhea = st.sidebar.selectbox("Diarrhea", options=["No", "Yes"])
        other_symptoms = st.sidebar.selectbox("Other Symptoms", options=["No", "Yes"])
        if gender == 'Male':
            gender = 1
        features = pd.DataFrame({
            "AgeCategory": [age_cat],
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
        })

        return features

    st.set_page_config(
        page_title="COVID-19 Severity Prediction App",
        page_icon="images/heart-fav.png"
    )

    st.title("COVID-19 Severity Prediction")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor.png",
                 caption="I'll help you diagnose your COVID-19 condition!",
                 width=150)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!

        Here, a logistic regression model using an undersampling technique
        was constructed using survey data of over 300k US residents from the year 2020.
        This application is based on it because it has proven to be better than the random forest
        (it achieves an accuracy of about 80%, which is quite good).

        To predict your heart disease status, simply follow the steps bellow:
        1. Enter the parameters that best descibe you;
        2. Press the "Predict" button and wait for the result.

        **Keep in mind that this results is not equivalent to a medical diagnosis!
        This model would never be adopted by health care facilities because of its less
        than perfect accuracy, so if you have any problems, consult a human doctor.**

        **Author: Ooi Teng He**

        """)

    covid19 = load_dataset()

    st.sidebar.title("Feature Selection")

    input_df = user_input_features()
    df = pd.concat([input_df, covid19], axis=0)
    df = df.drop(columns=["hasil dignosis"])

    cat_cols = ["umur", "jantina", "sakit tekak", "demam", "hilang deria rasa",
                "hilang deria bau", "batuk", "sesak nafas", "cirit birit", "lain-lain"]
    for cat_col in cat_cols:
        dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
        df = pd.concat([df, dummy_col], axis=1)
        del df[cat_col]

    df = df[:1]
    df.fillna(0, inplace=True)

    rf_model = pickle.load(open(MODEL_PATH, "rb"))

    if submit:
        prediction = rf_model.predict(df)
        prediction_proba = rf_model.predict_proba(df)

        if prediction == 1:
            st.markdown(f"Based on the provided information, the prediction is that you are in the **Mild Condition** "
                    f"with a probability of {round(prediction_proba[0][0] * 100, 2)}%.")
            st.image("images/heart-okay.jpg",
                     caption="Your heart seems to be okay! - Dr. Logistic Regression")
        elif prediction == 2:
            st.markdown(f"Based on the provided information, the prediction is that you are in the **Moderate Condition** "
                        f"with a probability of {round(prediction_proba[0][1] * 100, 2)}%.")
            st.image("images/heart-bad.jpg",
                     caption="I'm not satisfied with the condition of your heart! - Dr. Logistic Regression")
        else:
            st.markdown(f"Based on the provided information, the prediction is that you are in the **Severe Condition** "
                        f"with a probability of {round(prediction_proba[0][2] * 100, 2)}%.")
            st.image("images/heart-bad.jpg",
                     caption="I'm not satisfied with the condition of your heart! - Dr. Logistic Regression")

if __name__ == "__main__":
    main()
