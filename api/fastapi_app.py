"""
Uber Ride Cancellation Prediction - FastAPI Deployment
Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib, json, numpy as np, pandas as pd
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ── Load Model Artifacts ─────────────────────────────────────────────────────
MODEL_PATH   = "../models/xgb_model.pkl"
IMPUTER_PATH = "../models/imputer.pkl"
FEATURES_PATH= "../models/feature_names.pkl"
METRICS_PATH = "../models/metrics.json"

model         = joblib.load(MODEL_PATH)
imputer       = joblib.load(IMPUTER_PATH)
feature_names = joblib.load(FEATURES_PATH)

with open(METRICS_PATH) as f:
    metrics = json.load(f)

THRESHOLD = metrics.get("threshold", 0.30)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Uber Ride Cancellation Predictor",
    description="Predicts the probability of a customer cancelling their ride before it begins.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ─────────────────────────────────────────────────
class RideRequest(BaseModel):
    booking_time: str = Field(..., example="2024-06-15 08:30:00",
                              description="ISO datetime of booking: YYYY-MM-DD HH:MM:SS")
    vehicle_type: str = Field(..., example="Go Mini",
                              description="One of: Auto, Go Mini, Go Sedan, Bike, Premier Sedan, eBike, Uber XL")
    pickup_location: str   = Field(..., example="Saket")
    drop_location: str     = Field(..., example="Barakhamba Road")
    avg_vtat: float        = Field(..., example=5.2, description="Avg Vehicle Travel Arrival Time (mins)")
    avg_ctat: float        = Field(..., example=3.1, description="Avg Customer Travel Arrival Time (mins)")
    payment_method: Optional[str] = Field(None, example="UPI",
                                          description="UPI, Cash, Uber Wallet, Credit Card, Debit Card")
    customer_total_bookings: int  = Field(1, example=12)
    customer_cancel_history: int  = Field(0, example=1)

class PredictionResponse(BaseModel):
    booking_id: str
    cancellation_probability: float
    predicted_cancellation: bool
    risk_level: str
    recommendation: str
    model_version: str = "XGBoost_v1.0"
    threshold_used: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features_count: int
    threshold: float

# ── Feature Builder ───────────────────────────────────────────────────────────
TOP_PICKUPS = ['Khandsa','Barakhamba Road','Saket','Badarpur','Pragati Maidan',
               'Sector 18 Noida','Connaught Place','Dwarka','Lajpat Nagar','Gurgaon']
TOP_DROPS   = TOP_PICKUPS  # same set

VEHICLE_TYPES  = ['Auto','Go Mini','Go Sedan','Bike','Premier Sedan','eBike','Uber XL']
PAYMENT_METHODS= ['UPI','Cash','Uber Wallet','Credit Card','Debit Card','Unknown']

def build_features(req: RideRequest) -> np.ndarray:
    dt = pd.to_datetime(req.booking_time)
    hour    = dt.hour
    weekday = dt.dayofweek
    day     = dt.day
    month   = dt.month
    is_weekend = int(weekday in [5, 6])
    is_peak    = int(hour in [7,8,9,17,18,19,20])
    # time_period: Night(0-6), Morning(6-12), Afternoon(12-17), Evening(17-21), Late(21-24)
    if   hour <= 6:  tp = 'Night'
    elif hour <= 12: tp = 'Morning'
    elif hour <= 17: tp = 'Afternoon'
    elif hour <= 21: tp = 'Evening'
    else:            tp = 'Late'

    feat = {
        "Avg VTAT": req.avg_vtat,
        "Avg CTAT": req.avg_ctat,
        "hour": hour, "day": day, "month": month, "weekday": weekday,
        "is_weekend": is_weekend, "is_peak": is_peak,
        "missing_driver_rating": 1, "missing_customer_rating": 1,
        "missing_booking_value": 1, "missing_payment": int(req.payment_method is None),
        "customer_total_bookings": req.customer_total_bookings,
        "customer_cancel_history": req.customer_cancel_history,
    }

    # Vehicle dummies (reference: Auto)
    for vt in VEHICLE_TYPES[1:]:  # drop_first removes 'Auto'
        feat[f"vehicle_{vt}"] = int(req.vehicle_type == vt)

    # Payment dummies (reference: Cash)
    pm = req.payment_method or 'Unknown'
    for p in ['Credit Card','Debit Card','UPI','Uber Wallet','Unknown']:
        feat[f"pay_{p}"] = int(pm == p)

    # Pickup dummies
    pl = req.pickup_location if req.pickup_location in TOP_PICKUPS else 'Other'
    for loc in TOP_PICKUPS[1:]:  # drop_first removes first
        feat[f"pickup_{loc}"] = int(pl == loc)

    # Drop dummies
    dl = req.drop_location if req.drop_location in TOP_DROPS else 'Other'
    for loc in TOP_DROPS[1:]:
        feat[f"drop_{loc}"] = int(dl == loc)

    # Time period dummies (reference: Afternoon)
    for tper in ['Evening','Late','Morning','Night']:
        feat[f"tp_{tper}"] = int(tp == tper)

    # Build array aligned to feature_names
    row = [feat.get(f, 0.0) for f in feature_names]
    return np.array(row).reshape(1, -1)


def risk_label(prob: float) -> tuple[str, str]:
    if prob < 0.15:
        return "LOW", "No action needed. Standard dispatch."
    elif prob < 0.30:
        return "MEDIUM", "Monitor closely. Consider sending driver ETA updates."
    elif prob < 0.60:
        return "HIGH", "Send proactive message to customer. Consider incentive to retain booking."
    else:
        return "CRITICAL", "Very high cancellation risk. Apply aggressive retention strategy or pre-assign premium driver."

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {"message": "Uber Cancellation Predictor API", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        features_count=len(feature_names),
        threshold=THRESHOLD
    )

@app.get("/model-info", tags=["Info"])
def model_info():
    return {
        "model_type": "XGBoost Classifier",
        "version": "1.0.0",
        "features": len(feature_names),
        "threshold": THRESHOLD,
        "performance": metrics.get("models", {}).get("XGBoost", {}),
        "business_impact": metrics.get("business", {})
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: RideRequest):
    try:
        X = build_features(req)
        X_imp = imputer.transform(X)
        prob  = float(model.predict_proba(X_imp)[0, 1])
        cancelled = prob >= THRESHOLD
        risk, recommendation = risk_label(prob)

        import uuid
        return PredictionResponse(
            booking_id=str(uuid.uuid4())[:8].upper(),
            cancellation_probability=round(prob, 4),
            predicted_cancellation=cancelled,
            risk_level=risk,
            recommendation=recommendation,
            threshold_used=THRESHOLD
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(requests: list[RideRequest]):
    """Batch prediction endpoint (up to 100 rides at once)."""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Max 100 rides per batch request.")
    predictions = []
    for i, req in enumerate(requests):
        X = build_features(req)
        X_imp = imputer.transform(X)
        prob = float(model.predict_proba(X_imp)[0, 1])
        risk, rec = risk_label(prob)
        predictions.append({
            "index": i,
            "cancellation_probability": round(prob, 4),
            "predicted_cancellation": prob >= THRESHOLD,
            "risk_level": risk,
            "recommendation": rec
        })
    return {"predictions": predictions, "count": len(predictions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
