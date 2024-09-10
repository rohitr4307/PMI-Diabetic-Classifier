import streamlit as st

st.header("PIMA Diabetic Classification")

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

st.button("Predict Diabetic")