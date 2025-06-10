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
- Loaded dataset and reviewed schema and null values
- Found `InternetService` had 297 missing entries; filled them with an empty string
- Checked for and confirmed absence of duplicates
- Reviewed churn distribution (highly imbalanced)
- Plotted histograms and bar plots for key features
- Encoded categorical columns:
  - `Gender`: 1 for Female, 0 for Male  
  - `Churn`: 1 for Yes, 0 for No
- Selected only numerical features: `Age`, `Gender`, `Tenure`, `MonthlyCharges`

### 2. Train-Test Split and Scaling
- Used `train_test_split(test_size=0.2)`
- Applied `StandardScaler` for feature scaling
- Saved the scaler to disk with `joblib`

### 3. Class Imbalance Handling
- Found `Churn=Yes` represented 88% of the dataset
- Applied **SMOTE** on training data to synthetically oversample minority class (`Churn=0`)

### 4. Model Training and Tuning
Trained the following models:
- **Logistic Regression**
- **K-Nearest Neighbours** (GridSearchCV for `n_neighbors` and `weights`)
- **Support Vector Machine** (GridSearchCV for `C` and `kernel`)
- **Decision Tree** (GridSearchCV for multiple parameters)
- **Random Forest** (GridSearchCV with tuning for `n_estimators`, `max_features`, `bootstrap`)
- **XGBoost** (using `scale_pos_weight=1`, `eval_metric='logloss'`)

### 5. Evaluation & Metrics
- Used a custom evaluation function for each model:
  - Accuracy
  - Confusion Matrix
  - Classification Report (precision, recall, F1)
  - ROC Curve and AUC (when applicable)
- Stored and compared results across models visually
- Observed that despite high overall accuracy, most models struggled with correctly identifying `Churn=0` cases

### 6. Visual Insights
- Created bar plots for churn distribution and tenure comparisons
- Compared `MonthlyCharges` and `Tenure` by churn class and gender
- Plotted `Actual vs Predicted` churn counts

### 7. Final Model
- Selected the model with the best accuracy as final model (`Logistic Regression`, 86%)
- Saved it using `joblib` for Streamlit deployment

### 8. Deployment via Streamlit
- Developed interactive UI with Streamlit
- Allows manual input or customer ID-based prediction
- Renders churn risk level and offers downloadable prediction reports in PDF

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Created by Sheryll Dumapal  
Connect on LinkedIn: https://www.linkedin.com/in/sheryll-dumapal  
Read more on Medium: https://medium.com/@sherylldumapal
