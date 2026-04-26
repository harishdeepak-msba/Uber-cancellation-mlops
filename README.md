# 🚖 Uber Ride Cancellation Prediction
### End-to-End MLOps Pipeline · From Raw Data to Deployed API

<p align="center">
  <a href="https://mybinder.org/v2/gh/harishdeepak-msba/uber-cancellation-mlops/HEAD?filepath=notebooks/Uber_Cancellation_MLOps_Full.ipynb">
    <img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder" height="28"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-3.2-FF6600?style=flat-square&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/MLflow-Tracked-0194E2?style=flat-square&logo=mlflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

<p align="center">
  <i>Predicting ride cancellations before they happen — a full production-grade ML lifecycle<br/>
  covering feature engineering, experiment tracking, API deployment, and drift monitoring.</i>
</p>

---

## 📌 Table of Contents
- [Problem Statement](#-problem-statement)
- [Results](#-results)
- [Leakage Fix](#-leakage-fix--why-auc-was-100)
- [Project Dashboard](#️-project-dashboard)
- [MLOps Lifecycle](#-mlops-lifecycle-covered)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [API Usage](#-api-usage)
- [Tech Stack](#️-tech-stack)
- [Author](#-author)

---

## 🎯 Problem Statement

Every cancelled Uber ride costs the platform money through **wasted driver dispatch**, **degraded driver satisfaction**, and **customer churn risk**.

This project builds a machine learning model to flag high-risk bookings **before the ride begins** — using only pre-ride metadata with strict no-leakage enforcement.

| | |
|---|---|
| **Task** | Binary Classification |
| **Target** | Did the customer cancel? (`1` = cancelled · `0` = completed) |
| **Dataset** | 150,000 Uber bookings (2024) |
| **Challenge** | Severe class imbalance — ~7% cancellations (13.3 : 1 ratio) |
| **Constraint** | Only features available *before* the ride starts |

---

## 📊 Results

### Model Performance — Corrected (No Leakage)

> Threshold selected via **Youden's J statistic** on the validation set — maximises sensitivity + specificity, avoiding threshold overfitting to the test set.

| Model | ROC AUC | F1 Score | Precision | Recall | Threshold |
|:---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.934 | 0.357 | 0.219 | 97.1% | 0.011 |
| Random Forest | 0.950 | 0.396 | 0.247 | 99.8% | 0.142 |
| ⭐ **XGBoost** | **0.964** | **0.496** | **0.332** | **98.7%** | 0.0001 |

> **Why these F1 scores?** Class imbalance (7% positives) makes F1 harder to maximise. The models achieve strong AUC (discrimination ability) and very high recall — catching nearly all cancellations — at the cost of some false alarms. This is the correct trade-off: missing a cancellation costs $15; a false alarm costs only $2.

---

### 💰 Business Impact (XGBoost · Test Set — 15,000 rides)

| Metric | Value |
|:---|---:|
| Cancellations caught (True Positives) | 1,036 |
| Missed cancellations (False Negatives) | 14 |
| False alarms (False Positives) | 2,089 |
| 🔴 Baseline cost — no model | $15,750 |
| 🟢 Model cost (FN + FP combined) | $4,388 |
| 💰 **Estimated Savings** | **$11,362** |
| 📉 **Cost Reduction** | **72.1%** |

---

## 🔍 Leakage Fix — Why AUC Was 1.00

> **This section demonstrates a real-world debugging skill — catching and correcting data leakage.**

### The Problem
The original pipeline computed customer history features **on the full dataset before the train/test split**:

```python
# ❌ WRONG — leakage! Test rows contaminate their own features
df['customer_cancel_history'] = df['Customer ID'].map(
    df[df['target_customer_cancelled'] == 1]['Customer ID'].value_counts()
)
```

This means the model saw how many times a customer cancelled — **including the very cancellation it was trying to predict**. Result: artificially perfect AUC = 1.00 across all models.

### The Fix — Split First, Aggregate Second

```python
# ✅ CORRECT — split the data first
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Compute history ONLY from training rows
cust_cancel_history = c_train[y_train == 1].value_counts().to_dict()

# Apply learned mapping to all splits (unseen test customers → 0)
X_train['customer_cancel_history'] = c_train.map(cust_cancel_history).fillna(0)
X_test['customer_cancel_history']  = c_test.map(cust_cancel_history).fillna(0)
```

After fixing: **AUC dropped from 1.00 → 0.964** — realistic and deployable.

> 💡 * When traced to customer-level aggregation features computed on the full dataset before splitting, shows a classic form of target leakage. After enforcing split-first ordering, AUC settled at a realistic 0.964 for XGBoost."*

---

## 🖼️ Project Dashboard

![Dashboard](plots/corrected_results_dashboard.png)

---

## ✅ MLOps Lifecycle Covered

```
Raw Data ──► Feature Engineering ──► MLflow Tracking ──► Model Training
    └──► Business Metrics ──► FastAPI Deployment ──► Drift Monitoring
```

- [x] **Problem framing** & exploratory data analysis (EDA)
- [x] **Feature engineering** — 49 features, strict no-leakage policy
- [x] **Leakage detection & correction** — identified and fixed customer history leakage
- [x] **Experiment tracking** with MLflow — 3 models with full reproducibility
- [x] **Business cost analysis** — dollar value of false negatives vs false positives
- [x] **FastAPI REST deployment** — single + batch endpoints with Swagger UI
- [x] **Drift detection** — PSI + KS test with automated retraining triggers
- [x] **Formal monitoring plan** — daily, weekly, monthly cadence

---

## 📂 Repository Structure

```
uber-cancellation-mlops/
│
├── 📓 notebooks/
│   └── Uber_Cancellation_MLOps_Full.ipynb   ← Full end-to-end notebook
│
├── 🌐 api/
│   └── fastapi_app.py                        ← REST API (/predict, /predict/batch)
│
├── 📡 monitoring/
│   └── monitoring.py                         ← PSI + KS drift detection framework
│
├── 🖼️ plots/
│   └── corrected_results_dashboard.png       ← All visualisations in one view
│
├── ⚙️ binder/
│   └── environment.yml                       ← Binder environment config
│
├── model_metrics.json                        ← Corrected evaluation metrics
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/harishdeepak-msba/uber-cancellation-mlops.git
cd uber-cancellation-mlops
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook notebooks/Uber_Cancellation_MLOps_Full.ipynb
```
> Or click the **Launch Binder** badge at the top — runs in your browser, zero setup.

### 3. Start the API
```bash
uvicorn api.fastapi_app:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 📡 API Usage

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "booking_time": "2024-06-15 08:30:00",
    "vehicle_type": "Go Mini",
    "pickup_location": "Saket",
    "drop_location": "Barakhamba Road",
    "avg_vtat": 7.5,
    "avg_ctat": 4.2,
    "payment_method": "UPI",
    "customer_total_bookings": 8,
    "customer_cancel_history": 2
  }'
```

**Response:**
```json
{
  "cancellation_probability": 0.062,
  "predicted_cancellation": false,
  "risk_level": "LOW",
  "recommendation": "No action needed. Standard dispatch.",
  "model_version": "XGBoost_v1.0"
}
```

### Available Endpoints

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/health` | Liveness check |
| `GET` | `/model-info` | Model version + performance metrics |
| `POST` | `/predict` | Single ride cancellation prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 100 rides) |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 🛠️ Tech Stack

| Category | Tools |
|:---|:---|
| **Modelling** | XGBoost · scikit-learn · SHAP |
| **Experiment Tracking** | MLflow |
| **API Serving** | FastAPI · Uvicorn · Pydantic |
| **Monitoring** | Custom PSI + KS drift detection |
| **Visualisation** | Matplotlib · Seaborn |
| **Reproducibility** | Binder · requirements.txt |

---

## 🔍 Key Technical Decisions

| Decision | Rationale |
|:---|:---|
| **Split before aggregation** | Prevents customer history leakage — root cause of AUC = 1.00 |
| **Youden's J threshold** | Maximises TPR − FPR on val set; more robust than F1 grid search |
| `scale_pos_weight = 13.29` | Handles 13 : 1 class imbalance in XGBoost natively |
| **Median imputation** | Fit on train only; applied to val/test separately |
| **PSI + KS monitoring** | PSI catches gradual drift; KS catches sudden distribution shifts |
| **Post-ride features excluded** | Booking Value, Ride Distance, Ratings only available after the ride |

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <b>Harish Deepak</b><br/>
      MSBA · University of Arizona<br/>
      <a href="https://github.com/harishdeepak-msba">🐙 GitHub</a>
    </td>
  </tr>
</table>

---

<p align="center">
  <i>If you found this project useful, please consider giving it a ⭐</i>
</p>
