# Customer Churn Prediction with Machine Learning

This project is a predictive machine learning application that identifies the likelihood of customer churn based on user data such as age, gender, tenure, and monthly charges. Built with Python, Streamlit, and scikit-learn, the tool helps businesses understand and retain at-risk customers by surfacing churn probability and key customer insights.

Try the live app here: [https://predict-customer-churn-with-ml.streamlit.app/](https://predict-customer-churn-with-ml.streamlit.app/)

---

## Features

• Predict customer churn using a trained machine learning model  
• Input customer data manually or select existing customer IDs  
• View churn probability and risk level (Low, Medium, High)  
• Download individual churn predictions as a PDF report  
• Clean Streamlit interface with data explanations and model feedback  

---

## Dataset

Source: Telecom Customer Churn Dataset from Kaggle  
Link: https://www.kaggle.com/datasets/abdullah0a/telecom-customer-churn-insights-for-analysis/data

The dataset includes customer demographics, service usage patterns, and a churn label. Features include age, gender, tenure, monthly charges, contract type, internet service, and technical support.

---

## Technology Stack

• Python (Pandas, NumPy, scikit-learn)  
• Streamlit for interactive user interface  
• FPDF for exporting prediction results to PDF  
• StandardScaler for feature scaling  
• Multiple ML models: SVC, Logistic Regression, Random Forest, KNN, Decision Tree, XGBoost  
• SMOTE for handling class imbalance

---

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

---

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
streamlit run Customer_Churn_Prediction.py
```

---

## Methodology Summary

### 1. Data Review and Preprocessing
- Handled missing values (`InternetService`) and checked for duplicates
- Encoded categorical columns (`Gender`, `Churn`)
- Selected numerical features for modeling

### 2. Train-Test Split and Scaling
- Performed 80-20 split using `train_test_split`
- Applied `StandardScaler` to numerical features

### 3. Class Imbalance Handling
- Applied **SMOTE** to training data to balance churn vs no-churn classes

### 4. Model Training and Tuning
- Trained and tuned:
  - Logistic Regression ()
  - K-Nearest Neighbours (via GridSearchCV)
  - Support Vector Machine ()
  - Decision Tree
  - Random Forest (with hyperparameter tuning)
  - XGBoost (`scale_pos_weight=1`)

### 5. Evaluation & Metrics
- Used custom evaluation function to show:
  - Accuracy
  - Classification report
  - Confusion matrix
  - ROC curve and AUC (where applicable)
- Compared models using visual bar plots

### 6. Visual Insights
- Plotted actual vs predicted churn counts to assess model bias
- Found prediction skew toward class 1; addressed partially with SMOTE and weighted models

### 7. Deployment via Streamlit
- Interactive dashboard for churn prediction
- Manual and ID-based input
- Risk categorisation with PDF export functionality

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Created by Sheryll Dumapal  
Connect on LinkedIn: https://www.linkedin.com/in/sheryll-dumapal  
Read more on Medium: https://medium.com/@sherylldumapal
