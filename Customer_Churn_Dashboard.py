# Gender -> 1 Female, 0 Male
# Churn -> 1 yes, 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the x --> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st 
import joblib
import numpy as np 
import pandas as pd

# Load the scaler, model, and customer data
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
df = pd.read_csv("customer_churn_data.csv")  # Must include CustomerID and all required columns

st.title("Customer Churn Prediction Dashboard")
st.divider()

# ---- Section 1: View Churn Risk by Customer ID ----
st.header("🔍 View Churn Risk by Customer ID")

customer_ids = df["CustomerID"].unique()
selected_customer_id = st.selectbox("Select a Customer ID", customer_ids)

customer_data = df[df["CustomerID"] == selected_customer_id].iloc[0]

st.write("### Customer Information")
st.write(f"**Age**: {customer_data['Age']}")
st.write(f"**Gender**: {customer_data['Gender']}")
st.write(f"**Tenure**: {customer_data['Tenure']}")
st.write(f"**Monthly Charges**: € {customer_data['MonthlyCharges']:.2f}")
st.write(f"**Contract Type**: {customer_data['ContractType']}")
st.write(f"**Internet Service**: {customer_data['InternetService']}")
st.write(f"**Total Charges**: € {customer_data['TotalCharges']:.2f}")
st.write(f"**Technical Support**: {customer_data['TechSupport']}")
st.write(f"**Churn**: {customer_data["Churn"]}")

# Prepare input for model
gender_val = 1 if customer_data["Gender"] == "Female" else 0 
x = [customer_data["Age"], gender_val, customer_data["Tenure"], customer_data["MonthlyCharges"]]
x_df = pd.DataFrame([x], columns=["Age", "Gender", "Tenure", "MonthlyCharges"])
x_scaled = scaler.transform(x_df)

# Predict probability only if model supports it
if hasattr(model, "predict_proba"):
    probability = model.predict_proba(x_scaled)[0][1]

    if probability >= 0.7:
        category = "High"
    elif probability >= 0.4:
        category = "Medium"
    else:
        category = "Low"

    st.write(f"### Churn Probability: {probability * 100:.1f}%")
    st.write(f"### Risk Level: **{category}**")

else:
    st.warning("This model does not support probability predictions.")

st.divider()

