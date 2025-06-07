# Gender -> 1 Female, 0 Male
# Churn -> 1 yes, 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the x --> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st 
import joblib
import numpy as np 

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl") 

st.title("Customer Churn Prediction App")
st.divider()
st.write("Please entere the values and hit the predict button for getting a prediction. ")

st.divider()

age = st.number_input("Enter age", min_value = 10, max_value = 100, value= 30)

gender = st.selectbox("Enter the Gender", ["Male", "Female"])
                      
tenure = st.number_input("Enter Tenure", min_value = 0, max_value=130, value=10)

monthlyCharge = st.number_input("Enter Monthly Charge", min_value=30, max_value = 150)

st.divider()

predictbutton = st.button("Predict")

if predictbutton: 

    gender_selected = 1 if gender == "Female" else 0 
    x = [age, gender_selected, tenure, monthlyCharge]
    x1 = np.array(x)
    x_array = scaler.transform([x1])
    prediction = model.predict(x_array[0])
    predicted = "Churn" if prediction == 1 else "Not Churn"
    st.write(f"Predicted: {predicted}")

else:
    st.write("Please enter the values and use predict button")