import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💊",
    layout="centered"
)

MODEL_PATH = "best_insurance_model.joblib"

st.title("💊 Medical Insurance Cost Prediction App")
st.write("Enter the details below to predict insurance charges.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please make sure best_insurance_model.joblib is in the project folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

age = st.slider("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

st.subheader("Input Data")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Insurance Charges"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")
