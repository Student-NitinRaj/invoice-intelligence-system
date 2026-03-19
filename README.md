# 💼 Vendor Invoice Intelligence System  
### 🚀 Freight Cost Prediction & Invoice Risk Flagging  

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Business Objectives](#business-objectives)
- [Data Sources](#data-sources)
- [Exploratory Data Analysis](#exploratory-data-analysis-eda)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Application](#end-to-end-application)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run-this-project)
- [Author & Contact](#author--contact)

---

## 📊 Project Overview

This project implements an **end-to-end machine learning system** designed to support finance teams by:

1. **Predicting expected freight cost** for vendor invoices  
2. **Flagging high-risk invoices** that require manual approval  

---

## 🎯 Business Objectives

### 1. Freight Cost Prediction (Regression)

**Objective:**  
Predict expected freight cost using invoice quantity and value.

**Why it matters:**
- Freight is a major cost component  
- Helps in budgeting & forecasting  
- Improves vendor negotiation  

---

### 2. Invoice Risk Flagging (Classification)

**Objective:**  
Predict whether an invoice should be **flagged for manual approval**

**Why it matters:**
- Prevents financial leakage  
- Detects anomalies  
- Improves audit efficiency  

---

## 📂 Data Sources

Data is stored in a **SQLite database (`inventory.db`)** with:

- `vendor_invoice` → invoice data  
- `purchases` → item-level data  
- `purchase_prices` → reference prices  
- `inventory` → stock snapshots  

---

## 📈 Exploratory Data Analysis (EDA)

Key questions explored:

- Do flagged invoices have higher risk?  
- Does freight depend on quantity?  
- Are abnormal invoices statistically different?  

---

## 🤖 Models Used

### Regression (Freight Prediction)
- Linear Regression  
- Decision Tree Regressor  
- ✅ Random Forest Regressor (Final Model)

### Classification (Invoice Flagging)
- Logistic Regression  
- Decision Tree  
- ✅ Random Forest Classifier (Final Model)

---

## 📊 Evaluation Metrics

### Freight Prediction
- MAE  
- RMSE  
- R² Score  

### Invoice Flagging
- Accuracy  
- Precision / Recall / F1-score  
- Classification Report  

---

## 🖥 End-to-End Application

A **Streamlit dashboard** is built to:

- Input invoice data  
- Predict freight cost  
- Flag risky invoices  
- Display results in real-time  

---

## 📁 Project Structure
invoice-intelligence-system/
│
├── data/
│ └── inventory.db
│
├── freight_cost_prediction/
│ ├── train.py
│ └── models/
│ └── predict_freight_model.pkl
│
├── Invoice_flagging/
│ ├── train.py
│ └── models/
│ ├── predict_flag_invoice.pkl
│ └── scaler.pkl
│
├── inference/
│ ├── predict_freight.py
│ └── predict_invoice_flag.py
│
├── app.py
├── requirements.txt
└── README.md

## Auther
Name:- NITIN RAJ
MAIL:- nitinraj3152005@gmail.com

Thank You!!