import numpy as np
import pandas as pd
import streamlit as st
from Prediction import predict
from PMI_Diabetic_Classifier import outlier_treatment

st.title("PIMA Female Diabetic Prediction")
st.markdown(
    '''
    This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
    The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, 
    based on certain diagnostic measurements included in the dataset. Several constraints were placed on the
    selection of these instances from a larger database.
    In particular, all patients here are females at least 21 years old of Pima Indian heritage.
    '''
)

col1, col2 = st.columns(2)

with col1:
    input_num1 = st.number_input("No. of Pregnancies: ", value=2)
    input_num2 = st.number_input("Glucose Reading: ", value=200)
    input_num3 = st.number_input("Cummulative Blood Pressure Reading(mm Hg): ", value=250)
    input_num4 = st.number_input("Skin Thickness(mm): ", value=20)
with col2:
    input_num5 = st.number_input("Insulin(mu/ML): ", value=100)
    input_num6 = st.number_input("BMI: ", value=30)
    input_num7 = st.number_input("Diabetic Pedigree Function: ", value=0.5)
    input_num8 = st.number_input("Age (>19): ", value=30, min_value=20)

if st.button("Predict Diabetic"):
    df = test = pd.DataFrame(data=[[input_num1, input_num2, input_num3, input_num4, input_num5, input_num6, input_num7, input_num8]],
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
                                    'Age'])
    result = predict(df)

    if result[0] == 1:
        st.text("Subject is Diabetic!!!")
    else:
        st.text("Subject is Non-Diabetic!!!")