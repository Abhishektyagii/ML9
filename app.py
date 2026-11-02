
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to predict loan approval.")

if not os.path.exists("models/loan_model.joblib"):
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

model = joblib.load("models/loan_model.joblib")

st.subheader("Applicant Information")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Yes", "No"])

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=2000
)

loan_amount = st.number_input(
    "Loan Amount (in thousands)",
    min_value=1,
    value=150
)

loan_term = st.selectbox(
    "Loan Amount Term (months)",
    [120, 180, 240, 300, 360]
)

credit_history = st.selectbox(
    "Credit History",
    [1, 0],
    format_func=lambda x: "Good" if x == 1 else "Poor"
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "Approved":
        st.success("Congratulations! Loan is predicted as APPROVED.")
    else:
        st.error("Loan is predicted as REJECTED.")

    st.write("Predicted Status:", prediction)

    st.info(
        "This is an educational ML prediction based on a sample dataset, "
        "not an actual bank lending decision."
    )
