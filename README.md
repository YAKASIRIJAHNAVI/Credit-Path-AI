# 💰 Credit Path AI - Loan Default Prediction System

An AI-powered loan default prediction system developed using **Machine Learning (XGBoost)** and **Streamlit**. The application helps financial institutions evaluate the risk of loan applicants by predicting the probability of loan repayment or default based on key financial and credit-related features.

---

## 📌 Project Overview

Credit Path AI predicts whether a loan applicant is likely to **repay** or **default** on a loan. The system combines machine learning with an interactive Streamlit web application to provide real-time predictions, probability analysis, and risk assessment.

---

## 🎯 Objectives

- Predict loan default risk using machine learning.
- Assist financial institutions in making informed loan approval decisions.
- Provide an easy-to-use web interface for users.
- Visualize prediction probabilities and applicant risk factors.

---

## 🛠 Technologies Used

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- NumPy
  

---

## 📂 Dataset

- **Dataset:** German Credit Dataset
- **Samples:** 1000
- **Target Variable:** Loan Repayment / Loan Default

---

## 📊 Features Used

The machine learning model uses the following seven core features:

- Age
- Annual Income
- Loan Amount
- Credit Score
- Employment Years
- Number of Past Delinquencies
- Debt-to-Income Ratio (DTI)

The Streamlit application also collects additional information such as:

- Annual Savings
- Monthly Expenses
- Number of Bank Accounts
- Credit Cards
- Occupation
- Education Level
- Dependents
- Bankruptcy History
- Previous Loan Default
- Late Payments
- Marital Status
- Home Ownership
- Loan Purpose
- Years at Current Residence

These additional features are displayed in the risk analysis section to provide a more comprehensive assessment.

---

## ⚙️ Data Preprocessing

The following preprocessing techniques are applied:

- Data Cleaning
- Feature Selection
- Feature Engineering
- Debt-to-Income Ratio (DTI) Calculation
- Feature Scaling using StandardScaler
- Input Validation

---

## 🤖 Machine Learning Model

- **Algorithm:** XGBoost Classifier
- **Accuracy:** 87.5%

### Performance

| Metric | Score |
|---------|-------|
| Accuracy | 87.5% |
| Precision | 86% |
| Recall | 82% |
| F1 Score | 0.84 |
| ROC-AUC | 0.92 |

---

## 🔄 Prediction Workflow

1. User enters loan applicant details.
2. Input data is validated.
3. Debt-to-Income Ratio is calculated.
4. Features are scaled using the saved StandardScaler.
5. The XGBoost model predicts:
   - Repayment Probability
   - Default Probability
6. The application compares the default probability with a threshold of **0.45**.
7. The applicant is classified as:
   - ✅ Low Risk
   - ⚠️ High Risk
8. Results are displayed along with:
   - Probability Chart
   - Risk Score
   - Risk Factors
   - Recommendation

---

## 📈 Probability Analysis

The system uses the `predict_proba()` function of the trained XGBoost model.

Example:

Repayment Probability = 82%

Default Probability = 18%

If the Default Probability is greater than or equal to **45%**, the applicant is classified as **High Risk**. Otherwise, the applicant is classified as **Low Risk**.

---

## 💻 Streamlit Application

The application consists of three main sections:

### 📋 Quick Input
Allows users to enter the seven core features for quick prediction.

### 📝 Detailed Input
Collects additional financial, employment, and personal information for comprehensive analysis.

### 📊 Analysis
Displays:
- Loan Risk Prediction
- Probability Visualization
- Risk Factors
- Overall Risk Score
- Loan Recommendation

---

## 📁 Project Structure

```
Credit-Path-AI/
│
├── app.py
├── requirements.txt
├── dataset/
│   └── german_credit.csv
├── notebooks/
│   └── machine_learning_models.ipynb
├── assets/
│   └── screenshots/
├── README.md
└── .gitignore
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/YAKASIRIJAHNAVI/Credit-Path-AI.git
```

Navigate to the project folder

```bash
cd Credit-Path-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📷 Application Features

- Real-Time Loan Prediction
- Interactive Dashboard
- Probability Chart
- Risk Score Calculation
- Dynamic Recommendations
- Responsive User Interface

---

## 🚀 Future Enhancements

- Explainable AI using SHAP values
- Loan Approval Report Generation (PDF)
- Database Integration
- User Authentication
- Cloud Deployment
- Multi-language Support

---

## 👩‍💻 Author

**Yakasiri Jahnavi**

- GitHub: https://github.com/YAKASIRIJAHNAVI
- LinkedIn: https://linkedin.com/in/jahnavi4608

---

## 📜 License

This project is developed for educational and research purposes.
