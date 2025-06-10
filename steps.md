# Customer Churn Prediction Pipeline - Summary & Steps

This markdown file summarises the key steps taken in the churn prediction project, including data exploration, preprocessing, resampling, model training, tuning, and evaluation.

---

## 1. **Dataset Loading & Initial Review**
- Loaded `customer_churn_data.csv` using pandas:
```python
import pandas as pd
df = pd.read_csv("customer_churn_data.csv")
```
- Inspected the first few rows using `df.head()`
- Checked column datatypes with `df.info()`
- Identified missing values:

```python
df.isna().sum()
```
- Found 297 missing values in the `InternetService` column
- Filled missing values with empty string:
```python
df["InternetService"] = df["InternetService"].fillna("")
```
- Checked for duplicated rows using `df.duplicated().sum()` (result: 0 duplicates)

---

## 2. **Exploratory Data Analysis (EDA)**
- Descriptive statistics using `df.describe()`
- Plotted histograms of:
  - `MonthlyCharges`
  - `Tenure`
- Analysed churn distribution:
```python
df["Churn"].value_counts().plot(kind="bar")
```
- Grouped by churn and calculated means:
  - Average `MonthlyCharges`, `Tenure`, `Age` by `Churn` and `Gender`

---

## 3. **Feature Engineering**
- Selected features: `Age`, `Gender`, `Tenure`, `MonthlyCharges`
- Encoded `Gender`: Female = 1, Male = 0
```python
x = df[["Age", "Gender", "Tenure", "MonthlyCharges"]].copy()
x["Gender"] = x["Gender"].apply(lambda x: 1 if x == "Female" else 0)
```
- Encoded `Churn`: Yes = 1, No = 0
```python
y = df[["Churn"]].copy()
y["Churn"] = y["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
```

---

## 4. **Train-Test Split**
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
- Used a fixed `random_state` for reproducibility
- Note: Did **not** use stratified splitting in this project

---

## 5. **Feature Scaling**
- Applied `StandardScaler` to training and test sets:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```
- Saved the scaler object using `joblib`

---

## 6. **Handling Class Imbalance with SMOTE**
- Applied SMOTE only on training data to generate synthetic examples for the minority class:
```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)
```

---

## 7. **Model Training & Hyperparameter Tuning**
Trained and tuned the following models:

- **Logistic Regression**
  - With `class_weight='balanced'`
- **K-Nearest Neighbours (KNN)**
  - Tuned `n_neighbors`, `weights`
- **Support Vector Machine (SVM)**
  - Tuned `C`, `kernel`
  - Used `class_weight='balanced'`
- **Decision Tree**
  - Tuned `criterion`, `splitter`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Random Forest**
  - Tuned `n_estimators`, `max_features`, `bootstrap`
- **XGBoost**
  - With `scale_pos_weight=1`
  - Added directly to the model evaluation pipeline

---

## 8. **Model Evaluation**
- Used a reusable function `model_performance()` that:
  - Prints `accuracy`, `classification_report`
  - Displays confusion matrix
  - Plots ROC curve (if applicable)
- Stored accuracy results in a dictionary for model comparison
- Plotted comparison bar chart
- Saved the best model to disk using `joblib`

---

## 9. **Issue: Biased Predictions Toward Class 1**
- Several models predicted only class 1 (churn)
- Cause: class imbalance in test set (e.g., 172 churn vs 28 no-churn)
- Partial resolution steps:
  - Applied SMOTE to training data
  - Evaluated with `macro F1` and `balanced accuracy` instead of plain accuracy

---

## 10. **Next Suggestions**
- Explore new feature combinations or external data
- Apply SHAP or LIME for explainable ML
- Use model confidence thresholds for prediction cut-offs
- Deploy model via Flask/FastAPI or Streamlit app
