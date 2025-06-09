# Customer Churn Prediction with Machine Learning

This project is a predictive machine learning application that identifies the likelihood of customer churn based on user data such as age, gender, tenure, and monthly charges. Built with Python, Streamlit, and scikit-learn, the tool helps businesses understand and retain at-risk customers by surfacing churn probability and key customer insights.

Try the live app here: [https://predict-customer-churn-with-ml.streamlit.app/](https://predict-customer-churn-with-ml.streamlit.app/)

## Features

• Predict customer churn using a trained machine learning model  
• Input customer data manually or select existing customer IDs  
• View churn probability and risk level (Low, Medium, High)  
• Download individual churn predictions as a PDF report  
• Clean Streamlit interface with data explanations and model feedback  

## Dataset

Source: Telecom Customer Churn Dataset from Kaggle  
Link: https://www.kaggle.com/datasets/abdullah0a/telecom-customer-churn-insights-for-analysis/data

The dataset includes customer demographics, service usage patterns, and a churn label. Features include age, gender, tenure, monthly charges, contract type, internet service, and technical support.

## Technology Stack

• Python (Pandas, NumPy, scikit-learn)  
• Streamlit for interactive user interface  
• FPDF for exporting prediction results to PDF  
• StandardScaler for feature scaling  
• SVC model trained with probability prediction enabled  

## Project Structure

```

customer_churn_prediction  
│  
├── Customer_Churn_Prediction.py          Main Streamlit dashboard  
├── pages/                                Sub-pages for modular views  
│   └── Manual_Churn_Prediction.py        Manual input prediction tool  
├── model.pkl                             Trained machine learning model  
├── scaler.pkl                            Scaler used during training  
├── customer_churn_data.csv               Dataset with customer information  
├── LICENSE                               Open-source license (MIT)  
└── README.md                             Project documentation  

```

## How to Run Locally

1. Clone the repository  
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Install the required packages  
```bash
pip install -r requirements.txt
```

3. Launch the app  
```bash
streamlit run app.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Created by Sheryll Dumapal  
Connect on LinkedIn: https://www.linkedin.com/in/sheryll-dumapal
Read more on Medium: https://medium.com/@sherylldumapal
