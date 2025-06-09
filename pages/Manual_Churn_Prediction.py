# ---- Section 2: Manual Prediction Tool ----
import streamlit 
import joblib
import pandas as pd

# Load models and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.titel("🧮 Manual Prediction Tool")
st.write("Enter values below and click 'Predict' to see the churn classification.")

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
    st.write(f"### Prediction: **{predicted}**")
else:
    st.info("Fill in the values and click 'Predict'.")
