# 📡 Customer Churn Prediction

> Predicting which telecom customers are likely to cancel their service — enabling proactive retention strategies before revenue is lost.

---

## 📌 Problem Statement

Customer churn is one of the most critical challenges in the telecom industry. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a machine learning pipeline to identify at-risk customers **before** they churn, giving the business time to intervene.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 customers |
| Features | 21 (demographics, services, billing) |
| Target | `Churn` (Yes / No) |
| Class balance | 73.5% No Churn / 26.5% Churn |

---

## 🔍 Workflow

```
Raw Data
   │
   ├── Exploratory Data Analysis (EDA)
   │      └── Distributions, correlations, churn patterns
   │
   ├── Preprocessing
   │      ├── Fix TotalCharges dtype (hidden string values)
   │      ├── Binary encoding (Yes/No → 0/1)
   │      ├── Map "No service" values → 0
   │      ├── One-Hot Encoding (Contract, InternetService, PaymentMethod)
   │      └── Feature selection via correlation analysis
   │
   ├── Modeling
   │      ├── Logistic Regression (baseline)
   │      ├── Random Forest
   │      └── XGBoost with scale_pos_weight (handles class imbalance)
   │
   └── Explainability
          └── SHAP feature importance
```

---

## 📈 Results

| Model | ROC-AUC | Churn Recall | Churn F1 |
|---|---|---|---|
| Logistic Regression | 0.842 | 0.56 | 0.60 |
| Random Forest | 0.822 | 0.50 | 0.55 |
| **XGBoost (Balanced)** | **0.841** | **0.79** | **0.63** |

**Selected model: XGBoost (Balanced)**

XGBoost with `scale_pos_weight=2.77` was chosen as the final model. Despite similar ROC-AUC scores, it identifies **79% of churned customers** — catching 94 more at-risk customers per 1,409 test samples compared to the baseline. In a business context, a missed churn is far more costly than a false alarm.

---

## 🔎 Key Findings (SHAP Analysis)

SHAP (SHapley Additive exPlanations) reveals which features drive the model's predictions:

| Rank | Feature | Business Insight |
|---|---|---|
| 1 | `Contract_Month-to-month` | Strongest churn driver — flexible contracts = easy to leave |
| 2 | `tenure` | New customers (< 10 months) are the highest churn risk |
| 3 | `InternetService_Fiber optic` | Fiber customers churn more than DSL — possible quality/price issue |
| 4 | `MonthlyCharges` | Higher bills correlate with higher churn |
| 5 | `PaymentMethod_Electronic check` | Manual payment users churn more than auto-pay users |

---

## 💡 Business Recommendations

1. **Promote long-term contracts** — offer discounts for annual/two-year plans to month-to-month customers
2. **Onboarding loyalty program** — focus retention efforts on customers in their first 10 months
3. **Fiber optic investigation** — review pricing and service quality for fiber customers
4. **Auto-payment incentives** — encourage electronic check users to switch to automatic billing
5. **Proactive outreach** — use model scores to flag high-risk customers and trigger retention campaigns

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![SHAP](https://img.shields.io/badge/SHAP-0.44-brightgreen)

```
pandas          — data manipulation
numpy           — numerical operations
matplotlib      — visualizations
seaborn         — statistical plots
scikit-learn    — ML models and evaluation
xgboost         — gradient boosting
shap            — model explainability
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook customer_churn_prediction.ipynb
```

Or run directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── customer_churn_prediction.ipynb   # Main notebook (EDA → Modeling → SHAP)
├── README.md                         # Project overview
└── requirements.txt                  # Python dependencies
```

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
xgboost>=2.0
shap>=0.44
kagglehub
```
