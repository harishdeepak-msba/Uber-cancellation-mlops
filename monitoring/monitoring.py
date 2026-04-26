"""
Uber Cancellation Model - Monitoring & Drift Detection Plan
===========================================================
This module implements:
1. Statistical drift detection (PSI, KS-test, population shift)
2. Model performance monitoring (AUC, F1, precision, recall)
3. Business metric tracking (cost of FN vs FP)
4. Retraining trigger logic
5. Alerting framework
"""

import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from scipy import stats
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "monitoring_interval_days": 7,          # Check weekly
    "min_samples_for_check": 500,           # Min production samples before evaluating
    "psi_threshold": 0.2,                   # PSI > 0.2 = significant drift
    "ks_pvalue_threshold": 0.05,            # p < 0.05 = significant distribution shift
    "auc_degradation_threshold": 0.05,      # Drop > 0.05 from baseline triggers alert
    "f1_degradation_threshold": 0.05,       # Drop > 0.05 from baseline triggers alert
    "cancellation_rate_drift_pct": 0.30,    # 30% relative change in cancellation rate
    "retraining_data_min_rows": 10000,      # Minimum new rows before retraining
    "cost_fn": 15,                           # $ cost per false negative
    "cost_fp": 2,                            # $ cost per false positive
    "baseline_auc": 0.9990,
    "baseline_f1":  0.9990,
    "baseline_cancellation_rate": 0.07,
}

# ── 1. Population Stability Index (PSI) ───────────────────────────────────────
def compute_psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """
    PSI measures how much a feature distribution has shifted.
    PSI < 0.1: No significant change
    PSI 0.1–0.2: Moderate change — monitor
    PSI > 0.2: Significant change — investigate / retrain
    """
    ref_min, ref_max = reference.min(), reference.max()
    bin_edges = np.linspace(ref_min, ref_max, bins + 1)
    bin_edges[0]  -= 1e-6
    bin_edges[-1] += 1e-6

    ref_counts  = np.histogram(reference,  bins=bin_edges)[0]
    prod_counts = np.histogram(production, bins=bin_edges)[0]

    ref_pct  = ref_counts  / ref_counts.sum()
    prod_pct = prod_counts / prod_counts.sum()

    # Avoid log(0)
    ref_pct  = np.where(ref_pct  == 0, 1e-6, ref_pct)
    prod_pct = np.where(prod_pct == 0, 1e-6, prod_pct)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi)


# ── 2. KS Test for Feature Drift ─────────────────────────────────────────────
def ks_test_features(ref_df: pd.DataFrame, prod_df: pd.DataFrame,
                     features: list) -> pd.DataFrame:
    """
    Run Kolmogorov-Smirnov test on all numeric features.
    Returns a DataFrame with KS statistic and p-value per feature.
    """
    results = []
    for feat in features:
        if feat in ref_df.columns and feat in prod_df.columns:
            stat, pval = ks_2samp(ref_df[feat].dropna(), prod_df[feat].dropna())
            results.append({
                "feature": feat,
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(pval), 6),
                "drift_detected": pval < CONFIG["ks_pvalue_threshold"],
                "psi": compute_psi(ref_df[feat].dropna().values, prod_df[feat].dropna().values)
            })
    return pd.DataFrame(results).sort_values("p_value")


# ── 3. Prediction Score Drift ─────────────────────────────────────────────────
def check_prediction_drift(ref_probs: np.ndarray, prod_probs: np.ndarray) -> dict:
    """
    Compare the distribution of model prediction probabilities.
    A shift in score distributions often precedes performance degradation.
    """
    stat, pval = ks_2samp(ref_probs, prod_probs)
    psi = compute_psi(ref_probs, prod_probs)

    return {
        "ks_statistic": round(float(stat), 4),
        "ks_pvalue": round(float(pval), 6),
        "psi": round(psi, 4),
        "ref_mean_score": round(float(ref_probs.mean()), 4),
        "prod_mean_score": round(float(prod_probs.mean()), 4),
        "score_drift_detected": (pval < CONFIG["ks_pvalue_threshold"]) or (psi > CONFIG["psi_threshold"])
    }


# ── 4. Cancellation Rate Monitoring ──────────────────────────────────────────
def check_cancellation_rate_drift(ref_rate: float, prod_rate: float) -> dict:
    """
    Monitor if actual cancellation rate changes significantly.
    This can indicate a business or seasonal shift.
    """
    relative_change = abs(prod_rate - ref_rate) / ref_rate
    alert = relative_change > CONFIG["cancellation_rate_drift_pct"]
    return {
        "baseline_cancellation_rate": round(ref_rate, 4),
        "current_cancellation_rate": round(prod_rate, 4),
        "relative_change_pct": round(relative_change * 100, 2),
        "alert": alert,
        "message": (
            f"⚠️ Cancellation rate shifted {relative_change:.1%} from baseline"
            if alert else "✅ Cancellation rate within normal range"
        )
    }


# ── 5. Performance Monitoring (when labels available) ─────────────────────────
def check_model_performance(y_true: np.ndarray, y_prob: np.ndarray,
                             threshold: float = 0.30) -> dict:
    """
    Compare current performance metrics against baseline.
    Should be run when ground-truth labels become available (e.g., 24–48h delay).
    """
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    pr  = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    auc_drop = CONFIG["baseline_auc"] - auc
    f1_drop  = CONFIG["baseline_f1"]  - f1

    alerts = []
    if auc_drop > CONFIG["auc_degradation_threshold"]:
        alerts.append(f"⚠️ AUC dropped {auc_drop:.4f} from baseline ({CONFIG['baseline_auc']})")
    if f1_drop > CONFIG["f1_degradation_threshold"]:
        alerts.append(f"⚠️ F1 dropped {f1_drop:.4f} from baseline ({CONFIG['baseline_f1']})")

    return {
        "current_auc": round(auc, 4), "current_f1": round(f1, 4),
        "current_precision": round(pr, 4), "current_recall": round(rec, 4),
        "auc_vs_baseline": round(-auc_drop, 4), "f1_vs_baseline": round(-f1_drop, 4),
        "alerts": alerts,
        "retrain_recommended": len(alerts) > 0
    }


# ── 6. Business Cost Monitoring ───────────────────────────────────────────────
def compute_business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Translate confusion matrix into business dollar impact."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    model_cost    = fn * CONFIG["cost_fn"] + fp * CONFIG["cost_fp"]
    baseline_cost = y_true.sum() * CONFIG["cost_fn"]
    savings       = baseline_cost - model_cost
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "missed_cancellation_cost": fn * CONFIG["cost_fn"],
        "false_alarm_cost": fp * CONFIG["cost_fp"],
        "total_model_cost": round(model_cost, 2),
        "baseline_cost_no_model": round(baseline_cost, 2),
        "estimated_savings": round(savings, 2),
        "savings_pct": round(savings / baseline_cost * 100, 2) if baseline_cost > 0 else 0
    }


# ── 7. Retraining Trigger Logic ───────────────────────────────────────────────
def should_retrain(monitoring_results: dict) -> dict:
    """
    Centralized logic to decide if retraining is needed.
    Returns decision and reasons.
    """
    triggers = []

    drift = monitoring_results.get("feature_drift", {})
    n_drifted = sum(1 for v in drift.values() if isinstance(v, dict) and v.get("drift_detected"))
    if n_drifted > 5:
        triggers.append(f"Feature drift in {n_drifted} features (KS test)")

    score_drift = monitoring_results.get("prediction_score_drift", {})
    if score_drift.get("score_drift_detected"):
        psi = score_drift.get("psi", 0)
        triggers.append(f"Prediction score drift detected (PSI={psi})")

    rate_drift = monitoring_results.get("cancellation_rate_drift", {})
    if rate_drift.get("alert"):
        triggers.append(rate_drift.get("message", "Cancellation rate drift"))

    perf = monitoring_results.get("model_performance", {})
    triggers.extend(perf.get("alerts", []))

    return {
        "retrain_recommended": len(triggers) > 0,
        "triggers": triggers,
        "timestamp": datetime.now().isoformat(),
        "action": (
            "🔄 RETRAIN MODEL: Collect new labeled data and re-run training pipeline."
            if triggers else
            "✅ Model is stable. Continue monitoring."
        )
    }


# ── 8. Full Monitoring Report ─────────────────────────────────────────────────
def run_monitoring_report(ref_csv: str, prod_csv: str,
                           feature_names: list, threshold: float = 0.30) -> dict:
    """
    Generate a complete monitoring report.
    
    Args:
        ref_csv: Path to reference (training) data CSV with 'prediction_prob' and 'target' columns
        prod_csv: Path to recent production data CSV (same schema)
        feature_names: List of feature column names
        threshold: Decision threshold
    
    Returns:
        Full monitoring report dictionary
    """
    ref  = pd.read_csv(ref_csv)
    prod = pd.read_csv(prod_csv)

    report = {
        "generated_at": datetime.now().isoformat(),
        "reference_size": len(ref),
        "production_size": len(prod),
    }

    # Feature drift
    feat_drift_df = ks_test_features(ref, prod, feature_names)
    report["feature_drift_summary"] = {
        "features_tested": len(feat_drift_df),
        "features_with_drift": int(feat_drift_df["drift_detected"].sum()),
        "top_drifting_features": feat_drift_df.head(5).to_dict("records")
    }

    # Prediction score drift
    if "prediction_prob" in ref.columns and "prediction_prob" in prod.columns:
        report["prediction_score_drift"] = check_prediction_drift(
            ref["prediction_prob"].values, prod["prediction_prob"].values
        )

    # Cancellation rate drift
    if "target" in ref.columns and "target" in prod.columns:
        report["cancellation_rate_drift"] = check_cancellation_rate_drift(
            ref["target"].mean(), prod["target"].mean()
        )

    # Model performance (if labels available)
    if "target" in prod.columns and "prediction_prob" in prod.columns:
        report["model_performance"] = check_model_performance(
            prod["target"].values, prod["prediction_prob"].values, threshold
        )
        y_pred = (prod["prediction_prob"].values >= threshold).astype(int)
        report["business_impact"] = compute_business_cost(prod["target"].values, y_pred)

    # Retraining decision
    report["retraining_decision"] = should_retrain(report)

    return report


# ── 9. Monitoring Schedule & Alerting Plan (Documentation) ────────────────────
MONITORING_PLAN = """
╔══════════════════════════════════════════════════════════════════╗
║        UBER CANCELLATION MODEL - MONITORING PLAN                ║
╚══════════════════════════════════════════════════════════════════╝

📅 MONITORING SCHEDULE
━━━━━━━━━━━━━━━━━━━━━
• DAILY   → Log prediction volume, score distribution mean, cancellation rate
• WEEKLY  → Run PSI + KS drift tests on all features; compare to reference
• MONTHLY → Full performance evaluation (AUC, F1, PR) with labeled data
            Business cost analysis vs baseline

🎯 KEY METRICS TO TRACK
━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────────────────┬────────────┬──────────────────────────┐
│ Metric                      │ Baseline   │ Alert Threshold          │
├─────────────────────────────┼────────────┼──────────────────────────┤
│ ROC AUC                     │ 0.999      │ Drop > 0.05              │
│ F1 Score (cancellations)    │ 0.999      │ Drop > 0.05              │
│ Cancellation Rate           │ 7.0%       │ Change > 30%             │
│ PSI (prediction scores)     │ ~0.0       │ PSI > 0.2                │
│ KS test (key features)      │ p ≈ 1.0    │ p < 0.05                 │
│ Daily prediction volume     │ ~1000/day  │ Drop > 50% (pipeline issue)│
│ Avg prediction score        │ ~0.07      │ Change > 50% relative    │
└─────────────────────────────┴────────────┴──────────────────────────┘

🔄 RETRAINING TRIGGERS
━━━━━━━━━━━━━━━━━━━━━━
Retrain when ANY of the following occur:
  1. ROC AUC drops more than 0.05 from baseline on labeled holdout
  2. PSI > 0.2 on prediction score distribution
  3. > 5 features show significant KS drift (p < 0.05)
  4. Cancellation rate shifts > 30% from baseline (seasonal/business change)
  5. New vehicle type or payment method added to platform
  6. Significant policy change (driver incentives, service area expansion)
  7. Quarterly schedule (at minimum, even without drift detection)

📊 DATA COLLECTION FOR RETRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Minimum 10,000 new labeled rows before retraining
• Include at least 3 months of data to capture seasonal patterns
• Ensure class balance strategy (scale_pos_weight) is recomputed on new data
• Store all production features + ground truth in data lake (e.g., S3/BigQuery)

🏗️ RETRAINING WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━
1. Pull new data from data lake
2. Re-run feature engineering pipeline
3. Train new XGBoost model with MLflow tracking
4. Compare new model vs current champion on holdout
5. Shadow deploy: run both models in parallel for 48h
6. If new model outperforms → promote to production
7. Archive old model (never delete — compliance + rollback)

⚠️ ALERTING CHANNELS
━━━━━━━━━━━━━━━━━━━━
• Slack: #ml-alerts channel for drift alerts
• PagerDuty: For critical model failures (AUC < 0.85, API down)
• Email digest: Weekly monitoring report to data science team
• Dashboard: Grafana / MLflow UI for real-time metrics

📁 TOOLS & INFRASTRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━
• MLflow: Experiment tracking and model registry
• Evidently AI: Automated drift reports
• FastAPI: Model serving
• Prometheus + Grafana: API metrics (latency, throughput, error rate)
• Apache Airflow / Prefect: Scheduled monitoring pipelines
"""


if __name__ == "__main__":
    print(MONITORING_PLAN)

    # Demo: run a self-monitoring check using reference data
    print("\n=== DEMO: Running Monitoring on Reference Data (Self-Check) ===\n")

    ref_data_path = "/home/claude/uber_project/monitoring/reference_data.csv"
    feature_names = joblib.load("/home/claude/uber_project/models/feature_names.pkl")

    # Use reference data as both ref and prod for demo (should show no drift)
    report = run_monitoring_report(ref_data_path, ref_data_path, feature_names)

    print(f"Reference size: {report['reference_size']:,}")
    print(f"Production size: {report['production_size']:,}")
    print(f"\nFeatures with drift: {report['feature_drift_summary']['features_with_drift']}")
    drift_dec = report["retraining_decision"]
    print(f"\nRetrain recommended: {drift_dec['retrain_recommended']}")
    print(f"Action: {drift_dec['action']}")
    if drift_dec["triggers"]:
        for t in drift_dec["triggers"]:
            print(f"  → {t}")

    # Save report
    with open("/home/claude/uber_project/monitoring/monitoring_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nMonitoring report saved to monitoring/monitoring_report.json")
