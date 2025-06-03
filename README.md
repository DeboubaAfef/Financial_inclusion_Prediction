# 💰 Financial Inclusion Prediction App

This project uses **XGBoost** and **Streamlit** to predict whether an individual has access to a **bank account** based on demographic and socio-economic features. The dataset comes from the **Zindi Financial Inclusion Challenge** focused on Africa.

---

## 📊 Project Overview

Financial inclusion is a major global development goal. This app helps predict if a person is likely to have a bank account using features like:
- Age
- Gender
- Country
- Household size
- Education level
- Job type
- Cellphone access
- Marital status
- Relationship with head of household
- Urban or rural location

---

## 🧱 Project Structure

```bash
.
├── streamlit_model.py          # Streamlit app
├── xgboost_model.pkl           # Trained XGBoost model
├── Financial_inclusion_dataset.csv  # Original dataset
├── feature_importance.png     # Visualization of top features
├── confusion_matrix.png       # Model evaluation visualization
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
