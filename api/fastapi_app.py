"""
Uber Ride Cancellation Prediction - FastAPI Deployment
Render.com compatible
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="🚖 Uber Ride Cancellation Predictor",
    description="""
Predicts the probability of a customer cancelling their Uber ride before it begins.

**Model:** XGBoost — AUC = 0.964, trained on 150,000 bookings

**Key facts:**
- Uses only features available before the ride starts (no leakage)
- Customer history computed from training set only
- Threshold via Youden's J statistic on validation set
- Business impact: $11,362 savings per 15K rides (72.1% cost reduction)
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model state ───────────────────────────────────────────────────────────────
model         = None
imputer       = None
feature_names = None
MODEL_LOADED  = False

@app.on_event("startup")
async def load_model():
    global model, imputer, feature_names, MODEL_LOADED
    try:
        import joblib, os
        base          = os.path.dirname(os.path.abspath(__file__))
        model         = joblib.load(os.path.join(base, "../models/xgb_model.pkl"))
        imputer       = joblib.load(os.path.join(base, "../models/imputer.pkl"))
        feature_names = joblib.load(os.path.join(base, "../models/feature_names.pkl"))
        MODEL_LOADED  = True
        print("✅ Model loaded")
    except Exception as e:
        print(f"⚠️  Demo mode — model files not found: {e}")
        MODEL_LOADED  = False

# ── Schemas ───────────────────────────────────────────────────────────────────
class RideRequest(BaseModel):
    booking_time:             str   = Field(..., example="2024-06-15 08:30:00")
    vehicle_type:             str   = Field(..., example="Go Mini",
                                            description="Auto, Go Mini, Go Sedan, Bike, Premier Sedan, eBike, Uber XL")
    pickup_location:          str   = Field(..., example="Saket")
    drop_location:            str   = Field(..., example="Barakhamba Road")
    avg_vtat:                float  = Field(..., example=5.2)
    avg_ctat:                float  = Field(..., example=3.1)
    payment_method:  Optional[str]  = Field(None, example="UPI",
                                            description="UPI, Cash, Uber Wallet, Credit Card, Debit Card")
    customer_total_bookings:  int   = Field(1,    example=12)
    customer_cancel_history:  int   = Field(0,    example=1)

class PredictionResponse(BaseModel):
    cancellation_probability: float
    predicted_cancellation:   bool
    risk_level:               str
    recommendation:           str
    model_version:            str
    threshold_used:           float
    mode:                     str

# ── Helpers ───────────────────────────────────────────────────────────────────
THRESHOLD   = 0.0001
TOP_PICKUPS = ['Khandsa','Barakhamba Road','Saket','Badarpur','Pragati Maidan',
               'Sector 18 Noida','Connaught Place','Dwarka','Lajpat Nagar','Gurgaon']
TOP_DROPS   = TOP_PICKUPS
VEHICLE_TYPES = ['Auto','Go Mini','Go Sedan','Bike','Premier Sedan','eBike','Uber XL']

def risk_label(prob: float):
    if prob < 0.15:   return "LOW",      "No action needed. Standard dispatch."
    elif prob < 0.35: return "MEDIUM",   "Monitor closely. Send driver ETA updates."
    elif prob < 0.65: return "HIGH",     "Send proactive message. Consider retention incentive."
    else:             return "CRITICAL", "Very high cancellation risk. Apply aggressive retention strategy."

def demo_predict(req: RideRequest) -> float:
    """Rule-based fallback when model files are unavailable."""
    score = 0.07
    try:
        hour = datetime.strptime(req.booking_time, "%Y-%m-%d %H:%M:%S").hour
        if hour in [7,8,9,17,18,19,20]:     score += 0.04
    except: pass
    if req.avg_vtat > 8:                     score += 0.05
    if req.customer_cancel_history > 2:      score += 0.08
    if req.payment_method is None:           score += 0.03
    return min(score, 0.95)

def build_features(req: RideRequest) -> np.ndarray:
    import pandas as pd
    dt      = pd.to_datetime(req.booking_time)
    hour    = dt.hour
    weekday = dt.dayofweek
    feat    = {f: 0.0 for f in feature_names}
    feat.update({
        "Avg VTAT": req.avg_vtat, "Avg CTAT": req.avg_ctat,
        "hour": float(hour), "day": float(dt.day),
        "month": float(dt.month), "weekday": float(weekday),
        "is_weekend": float(weekday in [5,6]),
        "is_peak": float(hour in [7,8,9,17,18,19,20]),
        "missing_driver_rating": 1.0, "missing_customer_rating": 1.0,
        "missing_booking_value": 1.0,
        "missing_payment": float(req.payment_method is None),
        "customer_total_bookings": float(req.customer_total_bookings),
        "customer_cancel_history": float(req.customer_cancel_history),
    })
    for vt in VEHICLE_TYPES[1:]:
        if f"vehicle_{vt}" in feat:
            feat[f"vehicle_{vt}"] = float(req.vehicle_type == vt)
    pm = req.payment_method or "Unknown"
    for p in ['Credit Card','Debit Card','UPI','Uber Wallet','Unknown']:
        if f"pay_{p}" in feat:
            feat[f"pay_{p}"] = float(pm == p)
    pl = req.pickup_location if req.pickup_location in TOP_PICKUPS else "Other"
    for loc in TOP_PICKUPS[1:]:
        if f"pickup_{loc}" in feat:
            feat[f"pickup_{loc}"] = float(pl == loc)
    dl = req.drop_location if req.drop_location in TOP_DROPS else "Other"
    for loc in TOP_DROPS[1:]:
        if f"drop_{loc}" in feat:
            feat[f"drop_{loc}"] = float(dl == loc)
    tp_map = {(0,6):"Night",(6,12):"Morning",(12,17):"Afternoon",(17,21):"Evening",(21,24):"Late"}
    tp = next((v for (lo,hi),v in tp_map.items() if lo <= hour < hi), "Late")
    for tper in ["Evening","Late","Morning","Night"]:
        if f"tp_{tper}" in feat:
            feat[f"tp_{tper}"] = float(tp == tper)
    return np.array([[feat.get(f, 0.0) for f in feature_names]])

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "message":    "🚖 Uber Ride Cancellation Predictor API",
        "docs":       "/docs",
        "health":     "/health",
        "model_info": "/model-info",
        "predict":    "POST /predict",
    }

@app.get("/health", tags=["Health"])
def health():
    return {
        "status":       "healthy",
        "model_loaded": MODEL_LOADED,
        "mode":         "full_model" if MODEL_LOADED else "demo",
        "version":      "1.0.0"
    }

@app.get("/model-info", tags=["Info"])
def model_info():
    return {
        "model_type": "XGBoost Classifier",
        "version":    "1.0.0",
        "features":   len(feature_names) if feature_names else 49,
        "threshold":  THRESHOLD,
        "performance": {
            "ROC_AUC": 0.9638, "F1_Score": 0.4963,
            "Precision": 0.3315, "Recall": 0.9867
        },
        "business_impact": {
            "savings_per_15k_rides": "$11,362",
            "cost_reduction": "72.1%"
        }
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: RideRequest):
    try:
        if MODEL_LOADED:
            X    = build_features(req)
            prob = float(model.predict_proba(imputer.transform(X))[0, 1])
            mode = "full_model"
        else:
            prob = demo_predict(req)
            mode = "demo"
        risk, rec = risk_label(prob)
        return PredictionResponse(
            cancellation_probability = round(prob, 4),
            predicted_cancellation   = prob >= THRESHOLD,
            risk_level               = risk,
            recommendation           = rec,
            model_version            = "XGBoost_v1.0",
            threshold_used           = THRESHOLD,
            mode                     = mode
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(requests: list[RideRequest]):
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Max 100 rides per batch.")
    out = []
    for i, req in enumerate(requests):
        try:
            if MODEL_LOADED:
                X    = build_features(req)
                prob = float(model.predict_proba(imputer.transform(X))[0, 1])
                mode = "full_model"
            else:
                prob = demo_predict(req)
                mode = "demo"
            risk, rec = risk_label(prob)
            out.append({"index": i, "cancellation_probability": round(prob,4),
                        "predicted_cancellation": prob >= THRESHOLD,
                        "risk_level": risk, "recommendation": rec, "mode": mode})
        except Exception as e:
            out.append({"index": i, "error": str(e)})
    return {"predictions": out, "count": len(out)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
