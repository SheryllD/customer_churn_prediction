# ---- Section 2: Manual Prediction Tool ----
import streamlit as st
import joblib
import pandas as pd
from fpdf import FPDF
import tempfile
import plotly.graph_objects as go

# Load models and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Manual Prediction Tool")
st.write(
    "Use this tool to predict whether a customer is likely to churn based on just a few key details.\n\n"
    "**Steps:**\n"
    "1. Enter the customer’s age.\n"
    "2. Select their gender.\n"
    "3. Specify how long they’ve been a customer (tenure, in months).\n"
    "4. Enter their current monthly charge.\n\n"
    "Once all fields are filled in, click **'Predict'** to see the churn classification.\n\n"
    "This simple interface allows you to explore how basic customer characteristics can influence churn predictions using machine learning."
)


age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Enter Gender", ["Male", "Female"])
tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)
monthlyCharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)

predictbutton = st.button("Predict")


if predictbutton:
    gender_selected = 1 if gender == "Female" else 0 
    x = [age, gender_selected, tenure, monthlyCharge]
    x_df = pd.DataFrame([x], columns=["Age", "Gender", "Tenure", "MonthlyCharges"])
    x_scaled = scaler.transform(x_df)
    prediction = model.predict(x_scaled)[0]
    predicted = "Churn" if prediction == 1 else "Not Churn"

    st.write(f"#### Prediction: **{predicted}**")

    # Display churn probability 
    if hasattr(model, "predict_proba"): 
        probability = model.predict_proba(x_scaled)[0][1]
        st.write(f"#### Probability Score: **{probability * 100:.1f}%**")

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
    st.info("Fill in the values and click 'Predict'.")

# ------- Function to download it to PDF File 


download_button = st.button("Generate PDF")

if download_button:
    gender_selected = 1 if gender == "Female" else 0
    x = [age, gender_selected, tenure, monthlyCharge]
    x_df = pd.DataFrame([x], columns=["Age", "Gender", "Tenure", "MonthlyCharges"])
    x_scaled = scaler.transform(x_df)
    prediction = model.predict(x_scaled)[0]
    predicted = "Churn" if prediction == 1 else "Not Churn"

    st.write(f"### Prediction: **{predicted}**")

    # Generate PDF content
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Customer Churn Prediction", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Tenure: {tenure}", ln=True)
    pdf.cell(200, 10, txt=f"Monthly Charge: {monthlyCharge}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {predicted}", ln=True)

    # Save PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        tmp_file_path = tmp_file.name

    # Display download button
    with open(tmp_file_path, "rb") as file:
        st.download_button(
            label="Download PDF 📄 ",
            data=file,
            file_name="churn_prediction.pdf",
            mime="application/pdf"
        )
