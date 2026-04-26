\# 🚖 Uber Ride Cancellation Prediction — End-to-End MLOps



\[!\[Binder](https://mybinder.org/badge\_logo.svg)](https://mybinder.org/v2/gh/harishdeepak-msba/uber-cancellation-mlops/HEAD?filepath=notebooks/Uber\_Cancellation\_MLOps\_Full.ipynb)

!\[Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)

!\[XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)

!\[FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)

!\[MLflow](https://img.shields.io/badge/MLflow-Tracked-blue)



> Predicting ride cancellations before they happen — a full production ML pipeline

> covering feature engineering, experiment tracking, API deployment, and drift monitoring.



\---



\## 🎯 Problem Statement



Every cancelled Uber ride = lost revenue + wasted driver time + customer churn risk.

This project predicts high-risk bookings \*\*before the ride begins\*\* using only pre-ride metadata.



\- \*\*Task:\*\* Binary classification

\- \*\*Target:\*\* Customer cancellation (`1` = cancelled, `0` = not)

\- \*\*Dataset:\*\* 150,000 Uber bookings (2024)

\- \*\*Challenge:\*\* Severe class imbalance (\~7% cancellations)



\---



\## 📊 Results



| Model | ROC AUC | F1 Score | Recall |

|---|---|---|---|

| Logistic Regression | 1.000 | 0.998 | 99.7% |

| Random Forest | 1.000 | 0.998 | 99.8% |

| \*\*XGBoost ✅ Best\*\* | \*\*1.000\*\* | \*\*0.999\*\* | \*\*99.9%\*\* |



\### 💰 Business Impact (per 15,000 rides)

| | Cost |

|---|---|

| Baseline — no model | $15,750 |

| With model | $30 |

| \*\*Savings\*\* | \*\*$15,720 (99.8% reduction)\*\* |



\---



\## 🖼️ Project Dashboard



!\[Dashboard](plots/project\_dashboard.png)



\---



\## ✅ MLOps Lifecycle Covered



\- \[x] Problem framing \& exploratory data analysis

\- \[x] Feature engineering — 49 features, strict no-leakage policy

\- \[x] Experiment tracking with \*\*MLflow\*\* (3 models compared)

\- \[x] Business cost analysis — cost of false negatives vs false positives

\- \[x] \*\*FastAPI\*\* REST deployment with single + batch prediction endpoints

\- \[x] \*\*PSI + KS drift detection\*\* with automated retraining triggers

\- \[x] Formal monitoring plan — daily, weekly, monthly cadence



\---



\## 📂 Repository Structure



```

uber-cancellation-mlops/

├── notebooks/

│   └── Uber\_Cancellation\_MLOps\_Full.ipynb

├── api/

│   └── fastapi\_app.py

├── monitoring/

│   └── monitoring.py

├── plots/

│   └── project\_dashboard.png

├── binder/

│   └── environment.yml

├── model\_metrics.json

├── requirements.txt

└── README.md

```



\---



\## 🚀 Quick Start



```bash

git clone https://github.com/harishdeepak-msba/uber-cancellation-mlops

cd uber-cancellation-mlops

pip install -r requirements.txt

jupyter notebook notebooks/Uber\_Cancellation\_MLOps\_Full.ipynb

```



Start the API:

```bash

uvicorn api.fastapi\_app:app --port 8000

\# Open http://localhost:8000/docs

```



\---



\## 📡 Sample API Call



```bash

curl -X POST http://localhost:8000/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{

&#x20;   "booking\_time": "2024-06-15 08:30:00",

&#x20;   "vehicle\_type": "Go Mini",

&#x20;   "pickup\_location": "Saket",

&#x20;   "drop\_location": "Barakhamba Road",

&#x20;   "avg\_vtat": 7.5,

&#x20;   "avg\_ctat": 4.2,

&#x20;   "payment\_method": "UPI",

&#x20;   "customer\_total\_bookings": 8,

&#x20;   "customer\_cancel\_history": 2

&#x20; }'

```



\*\*Response:\*\*

```json

{

&#x20; "cancellation\_probability": 0.062,

&#x20; "predicted\_cancellation": false,

&#x20; "risk\_level": "LOW",

&#x20; "recommendation": "No action needed. Standard dispatch.",

&#x20; "model\_version": "XGBoost\_v1.0"

}

```



\---



\## 🛠️ Tech Stack



| Category | Tools |

|---|---|

| Modeling | XGBoost, scikit-learn, SHAP |

| Experiment Tracking | MLflow |

| API Serving | FastAPI, Uvicorn, Pydantic |

| Monitoring | Custom PSI + KS drift detection |

| Visualization | Matplotlib, Seaborn |



\---



\## 👤 Author



\*\*Harish Deepak\*\* — MSBA, University of Arizona  

\[GitHub Profile](https://github.com/harishdeepak-msba)

