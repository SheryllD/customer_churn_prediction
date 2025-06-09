# Customer Churn Dashboard -- Streamlit 

# Gender -> 1 Female, 0 Male
# Churn -> 1 yes, 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the x --> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st 
import joblib
import numpy as np 
import pandas as pd
from fpdf import FPDF
import tempfile
import plotly.graph_objects as go

pdf = FPDF()

# Loading the scaler, model, and customer data
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
df = pd.read_csv("customer_churn_data.csv")  # Must include CustomerID and all required columns

st.title("Customer Churn Prediction")
st.write(
    "This Streamlit app demonstrates how machine learning can help predict customer churn in the telecom industry.\n\n"
    "Using customer demographics and service usage data, the model identifies which users are most likely to cancel their subscription.\n\n"
    "The project covers the full data science workflow, from data cleaning and feature engineering to model training and deployment.\n\n"
    "It’s a practical example of how predictive analytics can be used to tackle real business problems in subscription-based services."
)

st.divider()

# ---- Section 1: View Churn Risk by Customer ID ----
st.header("View Churn Risk by Customer ID")

customer_ids = df["CustomerID"].unique()
selected_customer_id = st.selectbox("Select a Customer ID", customer_ids)

customer_data = df[df["CustomerID"] == selected_customer_id].iloc[0]

st.write("#### Customer Information:")
st.write(f"**Age**: {customer_data['Age']}")
st.write(f"**Gender**: {customer_data['Gender']}")
st.write(f"**Contract Type**: {customer_data['ContractType']}")
st.write(f"**Monthly Charges**: € {customer_data['MonthlyCharges']:.2f}")
st.write(f"**Tenure**: {customer_data['Tenure']}")
st.write(f"**Total Charges**: € {customer_data['TotalCharges']:.2f}")
st.write(f"**Internet Service**: {customer_data['InternetService']}")
st.write(f"**Technical Support**: {customer_data['TechSupport']}")

# Prepare input for model
gender_val = 1 if customer_data["Gender"] == "Female" else 0 
x = [customer_data["Age"], gender_val, customer_data["Tenure"], customer_data["MonthlyCharges"]]
x_df = pd.DataFrame([x], columns=["Age", "Gender", "Tenure", "MonthlyCharges"])
x_scaled = scaler.transform(x_df)

# Predict probability 
if hasattr(model, "predict_proba"):
    probability = model.predict_proba(x_scaled)[0][1]

    if probability >= 0.7:
        category = "High"
    elif probability >= 0.4:
        category = "Medium"
    else:
        category = "Low"

    st.divider()
    st.write(f"#### Churn Probability: {probability * 100:.1f}%")
    st.write(f"#### Risk Level: **{category}**")
    st.write(f"#### Churned: {customer_data["Churn"]}")

    fig = go.Figure(go.Indicator(
    mode="gauge+number", 
    value=probability *100,
    number={'suffix': "%"}, 
    gauge={
        'axis':{'range': [0,100]}, 
        'bar': {'color': "black"}, 
        'steps':[
            {'range': [0, 20], 'color': "#999999"},
            {'range': [20, 40], 'color': "#d65f5f"},
            {'range': [40, 60], 'color': "#f4e04d"},
            {'range': [60, 80], 'color': "#b7d26b"},
            {'range': [80, 100], 'color': "#60c271"}
            ],
            'threshold': {
                'line': {'color': "black", "width":4}, 
                'thickness': 0.75,
                'value': probability * 100 
                }
            }
        ))
    st.plotly_chart(fig)

else:
    st.warning("This model does not support probability predictions.")

st.divider()

# --- PDF Report Generation ---
st.write("Download the Report 📄 ")

if st.button("Generate PDF Report"):
    predicted = "Churn" if model.predict(x_scaled)[0] == 1 else "Not Churn"

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Customer Churn Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Customer ID: {customer_data['CustomerID']}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {customer_data['Age']}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {customer_data['Gender']}", ln=True)
    pdf.cell(200, 10, txt=f"Tenure: {customer_data['Tenure']}", ln=True)
    pdf.cell(200, 10, txt=f"Monthly Charges: € {customer_data['MonthlyCharges']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Contract Type: {customer_data['ContractType']}", ln=True)
    pdf.cell(200, 10, txt=f"Internet Service: {customer_data['InternetService']}", ln=True)
    pdf.cell(200, 10, txt=f"Total Charges: € {customer_data['TotalCharges']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Technical Support: {customer_data['TechSupport']}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Prediction: {predicted}", ln=True)

    if hasattr(model, "predict_proba"):
        pdf.cell(200, 10, txt=f"Churn Probability: {probability * 100:.1f}%", ln=True)
        pdf.cell(200, 10, txt=f"Risk Level: {category}", ln=True)

    # Save PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        tmp_path = tmp_file.name

    # Display download button
    with open(tmp_path, "rb") as file:
        st.download_button(
            label="Download the Report 📄 ",
            data=file,
            file_name=f"customer_{customer_data['CustomerID']}_churn_report.pdf",
            mime="application/pdf"
        )
