import streamlit as st
import pandas as pd
import numpy as np
import pickle

DATASET_PATH = "./data/Covid-19 Cleaned Data.csv"
MODEL_PATH = "./model/random_forest_model.pkl"

def main():
    @st.cache_data(persist=True)
    def load_dataset() -> pd.DataFrame:
        covid19_df = pd.read_csv(DATASET_PATH, encoding="UTF-8")
        covid19_df = pd.DataFrame(np.sort(covid19_df.values, axis=0),
                                index=covid19_df.index,
                                columns=covid19_df.columns)
        return covid19_df

    def user_input_features() -> pd.DataFrame:
        age_cat = st.sidebar.selectbox("Age category", options=["<9","10-19","20-24","25-59",">60"])
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
        comorbid = st.sidebar.selectbox("Comorbid", options=["No", "Yes"])

        if age_cat == '<9':
            age_cat = 1
        elif age_cat == '10-19':
            age_cat = 2
        elif age_cat == '20-24':
            age_cat = 3
        elif age_cat == '25-59':
            age_cat = 4
        else:
            age_cat = 5

        if gender == 'Male':
            gender = 1
        else:
            gender = 0
        
        if sore_throat == 'Yes':
            sore_throat = 1
        else:
            sore_throat = 0
        
        if fever == 'Yes':
            fever = 1
        else:
            fever = 0

        if flu == 'Yes':
            flu = 1
        else:
            flu = 0

        if loss_of_taste == 'Yes':
            loss_of_taste = 1
        else:
            loss_of_taste = 0

        if loss_of_smell == 'Yes':
            loss_of_smell = 1
        else:
            loss_of_smell = 0

        if cough == 'Yes':
            cough = 1
        else:
            cough = 0

        if breathing_difficulties == 'Yes':
            breathing_difficulties = 1
        else:
            breathing_difficulties = 0

        if diarrhea == 'Yes':
            diarrhea = 1
        else:
            diarrhea = 0

        if other_symptoms == 'Yes':
            other_symptoms = 1
        else:
            other_symptoms = 0

        if comorbid == 'Yes':
            comorbid = 1
        else:
            comorbid = 0

        features = pd.DataFrame({
            "umur": [age_cat],
            "jantina": [gender],
            "sakit tekak": [sore_throat],
            "demam": [fever],
            "selesema": [flu],
            "hilang deria rasa": [loss_of_taste],
            "hilang deria bau": [loss_of_smell],
            "batuk": [cough],
            "sesak nafas": [breathing_difficulties],
            "cirit birit": [diarrhea],
            "lain-lain": [other_symptoms],
            "komorbid": [comorbid]
        })

        return features

    st.set_page_config(
        page_title="COVID-19 Severity Prediction App",
        page_icon="images/covid.jpg"
    )

    st.title("COVID-19 Severity Prediction")
    st.subheader("Are you wondering about the severity of your COVID-19 condition? "
                 "This app will help you to diagnose it!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor.jpeg")
    with col2:
        st.markdown("""
        Did you know that machine learning models can help predict the severity of COVID-19?
        In this app, you can estimate your COVID-19 severity (mild, moderate, or severe) based on your symptoms.

        Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.
        Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment.
        However, some will become seriously ill and require medical attention. 
        Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. 
        Anyone can get sick with COVID-19 and become seriously ill or die at any age. 

        To predict your COVID-19 severity, follow these steps:
        1. To start the prediction, first you need to select your age category and gender.
        2. Select the parameters that best describe your symptoms like fever, sore throat, flu, loss of smell, loss of taste, etc.
        3. Press the "Predict" button and wait for the result.

        **Keep in mind that this result is not equivalent to a medical diagnosis!
        Consult a healthcare professional for accurate diagnosis and advice.**



        """)
        submit = st.button("Predict")

    covid19 = load_dataset()

    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=100)

    input_df = user_input_features()
    df = pd.concat([input_df, covid19], axis=0)
    df = df.drop(columns=["hasil dignosis"])

    cat_cols = ["umur", "jantina", "sakit tekak", "demam", "selesema", "hilang deria rasa","hilang deria bau", "batuk", "sesak nafas", "cirit birit", "lain-lain", "komorbid"]
    for cat_col in cat_cols:
        dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
        df = pd.concat([df, dummy_col], axis=1)
        df = df[input_df.columns]  # Keep only the original input columns
    
    df = df.iloc[:1, :]  # Keep only the first row
    df.fillna(0, inplace=True)

    rf_model = pickle.load(open(MODEL_PATH, "rb"))

    if submit:
        prediction = rf_model.predict(df)
        prediction_proba = rf_model.predict_proba(df)

        if prediction == 1:
            st.markdown(f"Based on the provided information, the prediction is that you are in the **Mild Condition** "
                    f"with a probability of {round(prediction_proba[0][0] * 100, 2)}%.")
            st.image("images/mild.jpg",
                     use_column_width=True,  # Adjusts the image width to the column width
                     output_format="JPEG",  # Output format of the image
                     width=300,  # Adjust the width of the image as desired
            )
            st.markdown('<p style="text-align: center;">Your COVID-19 condition seems to be mild!</p>', unsafe_allow_html=True)
        elif prediction == 2:
            st.markdown(f"Based on the provided information, the prediction is that you are in the **Moderate Condition** "
                        f"with a probability of {round(prediction_proba[0][1] * 100, 2)}%.")
            st.image("images/moderate.jpg",
                     use_column_width=True,  # Adjusts the image width to the column width
                     output_format="JPEG",  # Output format of the image
                     width=300,  # Adjust the width of the image as desired
            )
            st.markdown('<p style="text-align: center;">Your COVID-19 condition seems to be moderate!</p>', unsafe_allow_html=True)
        else:
            st.markdown(f"Based on the provided information, the prediction is that you are in the **Severe Condition** "
                        f"with a probability of {round(prediction_proba[0][2] * 100, 2)}%.")
            st.image("images/severe.jpeg",
                    use_column_width=True,  # Adjusts the image width to the column width
                    output_format="JPEG",  # Output format of the image
                    width=300,  # Adjust the width of the image as desired
            )
            st.markdown('<p style="text-align: center;">Your COVID-19 condition seems to be severe!</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
