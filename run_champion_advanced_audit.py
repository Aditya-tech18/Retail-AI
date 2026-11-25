from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_curve
)

ROOT = Path.cwd()
SOURCE = ROOT / "reports/portfolio_audit/retrospective_scored_orders.csv"
OUT = ROOT / "reports/champion_advanced_audit"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE)

required = {"order_id", "actual_late", "predicted_risk"}
assert required.issubset(df.columns)
assert df["order_id"].is_unique

y = df["actual_late"].to_numpy(dtype=int)
p = df["predicted_risk"].to_numpy(dtype=float)

assert np.isfinite(p).all()
assert ((p >= 0) & (p <= 1)).all()
assert set(np.unique(y)).issubset({0, 1})

rng = np.random.default_rng(42)

print("\n" + "=" * 70)
print("RETAILAI | CHAMPION ADVANCED ML AUDIT")
print("=" * 70)

print("Orders:", len(df))
print("Late orders:", int(y.sum()))
print("Late prevalence:", round(y.mean(), 6))

# ==========================================================
# 1. CLASSIFICATION METRICS AT MULTIPLE CAPACITIES
# ==========================================================

def evaluate_capacity(y_true, probabilities, capacity):
    n = len(y_true)
    k = max(1, math.ceil(n * capacity))

    order = np.argsort(
        -probabilities,
        kind="stable"
    )

    prediction = np.zeros(n, dtype=int)
    prediction[order[:k]] = 1

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    ).ravel()

    return {
        "capacity": capacity,
        "flagged": int(k),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(
            y_true, prediction, zero_division=0
        ),
        "recall": recall_score(
            y_true, prediction, zero_division=0
        ),
        "f1": f1_score(
            y_true, prediction, zero_division=0
        ),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "balanced_accuracy": balanced_accuracy_score(
            y_true, prediction
        ),
        "lift": (
            precision_score(
                y_true, prediction, zero_division=0
            ) / y_true.mean()
            if y_true.mean() else np.nan
        )
    }

capacities = [
    0.01, 0.05, 0.10, 0.15,
    0.20, 0.25, 0.30, 0.40, 0.50
]

capacity_results = pd.DataFrame([
    evaluate_capacity(y, p, c)
    for c in capacities
])

capacity_results.to_csv(
    OUT / "capacity_threshold_comparison.csv",
    index=False
)

print("\nCAPACITY-BASED METRICS")
print(capacity_results.to_string(index=False))

# ==========================================================
# 2. THRESHOLD-BASED CLASSIFICATION
# ==========================================================

thresholds = np.unique(
    np.concatenate([
        np.linspace(0.05, 0.95, 19),
        [0.44, 0.50]
    ])
)

threshold_rows = []

for threshold in thresholds:
    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    threshold_rows.append({
        "threshold": float(threshold),
        "flagged": int(pred.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(
            y, pred, zero_division=0
        )
    })

threshold_results = pd.DataFrame(threshold_rows)

threshold_results.to_csv(
    OUT / "probability_threshold_comparison.csv",
    index=False
)

# ==========================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS
# ==========================================================

bootstrap_rows = []

for _ in range(1000):
    idx = rng.integers(0, len(y), len(y))
    ys = y[idx]
    ps = p[idx]

    if ys.sum() == 0 or ys.sum() == len(ys):
        continue

    result = evaluate_capacity(ys, ps, 0.20)

    result["pr_auc"] = average_precision_score(
        ys, ps
    )
    result["roc_auc"] = roc_auc_score(
        ys, ps
    )

    bootstrap_rows.append(result)

boot = pd.DataFrame(bootstrap_rows)

ci_rows = []

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "lift",
    "pr_auc",
    "roc_auc"
]:
    ci_rows.append({
        "metric": metric,
        "ci_low": boot[metric].quantile(0.025),
        "ci_high": boot[metric].quantile(0.975)
    })

pd.DataFrame(ci_rows).to_csv(
    OUT / "bootstrap_confidence_intervals.csv",
    index=False
)

# ==========================================================
# 4. CALIBRATION DIAGNOSTICS
# ==========================================================

base_rate = float(y.mean())

baseline_probabilities = np.full(
    len(y), base_rate
)

calibration_metrics = pd.DataFrame([
    {
        "model": "V1 Random Forest",
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "mean_predicted_probability": p.mean(),
        "actual_prevalence": base_rate,
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p)
    },
    {
        "model": "Constant prevalence baseline",
        "brier": brier_score_loss(
            y, baseline_probabilities
        ),
        "log_loss": log_loss(
            y, baseline_probabilities,
            labels=[0, 1]
        ),
        "mean_predicted_probability": base_rate,
        "actual_prevalence": base_rate,
        "pr_auc": base_rate,
        "roc_auc": 0.5
    }
])

calibration_metrics.to_csv(
    OUT / "calibration_diagnostics.csv",
    index=False
)

bins = np.linspace(0, 1, 11)

calibration_df = pd.DataFrame({
    "actual": y,
    "predicted": p
})

calibration_df["probability_bin"] = pd.cut(
    calibration_df["predicted"],
    bins=bins,
    include_lowest=True
)

reliability = (
    calibration_df
    .groupby("probability_bin", observed=False)
    .agg(
        orders=("actual", "size"),
        actual_late_rate=("actual", "mean"),
        mean_predicted_probability=("predicted", "mean")
    )
    .reset_index()
)

reliability.to_csv(
    OUT / "reliability_diagram_data.csv",
    index=False
)

# IMPORTANT:
# We DO NOT fit calibration on this test set.
# Calibration must be trained on a separate validation set.

# ==========================================================
# 5. BUSINESS THRESHOLD SCENARIOS
# ==========================================================

business_rows = []

for _, row in capacity_results.iterrows():
    for intervention_cost in [2, 5, 10]:
        for assumed_effectiveness in [0.10, 0.20, 0.30]:
            for avoided_delay_value in [25, 50, 100]:

                expected_prevented = (
                    row["tp"] * assumed_effectiveness
                )

                total_cost = (
                    row["flagged"] * intervention_cost
                )

                gross_value = (
                    expected_prevented * avoided_delay_value
                )

                net_value = gross_value - total_cost

                business_rows.append({
                    "capacity": row["capacity"],
                    "flagged": int(row["flagged"]),
                    "historical_late_captured": int(row["tp"]),
                    "intervention_cost": intervention_cost,
                    "assumed_effectiveness": assumed_effectiveness,
                    "avoided_delay_value": avoided_delay_value,
                    "expected_delays_prevented": expected_prevented,
                    "intervention_spend": total_cost,
                    "expected_gross_value": gross_value,
                    "expected_net_value": net_value,
                    "roi": (
                        net_value / total_cost
                        if total_cost else np.nan
                    )
                })

business = pd.DataFrame(business_rows)

business.to_csv(
    OUT / "capacity_business_scenarios.csv",
    index=False
)

# ==========================================================
# 6. MODEL CURVES
# ==========================================================

precision, recall, pr_thresholds = precision_recall_curve(
    y, p
)

pd.DataFrame({
    "precision": precision,
    "recall": recall,
    "threshold": np.append(pr_thresholds, np.nan)
}).to_csv(
    OUT / "precision_recall_curve.csv",
    index=False
)

fpr, tpr, roc_thresholds = roc_curve(y, p)

pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": roc_thresholds
}).to_csv(
    OUT / "roc_curve.csv",
    index=False
)

# ==========================================================
# 7. RISK-DECILE MONITORING
# ==========================================================

monitor = pd.DataFrame({
    "actual": y,
    "predicted": p
})

monitor["risk_decile"] = pd.qcut(
    monitor["predicted"].rank(method="first"),
    q=10,
    labels=False
) + 1

deciles = (
    monitor.groupby("risk_decile")
    .agg(
        orders=("actual", "size"),
        actual_late_rate=("actual", "mean"),
        mean_predicted_probability=("predicted", "mean")
    )
    .reset_index()
)

deciles.to_csv(
    OUT / "risk_decile_monitoring.csv",
    index=False
)

# ==========================================================
# 8. LEAKAGE AND EVALUATION STATUS
# ==========================================================

status = {
    "champion": "V1 Random Forest",
    "prediction_stage": "post_approval",
    "test_set_previously_inspected": True,
    "calibration_fitted_on_test": False,
    "calibration_status": "Diagnostics only",
    "historical_seller_features_certified": False,
    "fresh_blind_holdout_available": False,
    "business_results": "Hypothetical scenarios only",
    "recommended_next_steps": [
        "Audit feature event-time availability",
        "Fit calibration on independent validation data",
        "Validate on future untouched holdout",
        "Check subgroup performance and temporal drift",
        "Select operating capacity using validation and business costs"
    ]
}

(OUT / "evaluation_status.json").write_text(
    json.dumps(status, indent=2
))

print("\n" + "=" * 70)
print("CALIBRATION COMPARISON")
print("=" * 70)
print(calibration_metrics.to_string(index=False))

print("\n" + "=" * 70)
print("95% BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 70)
print(pd.DataFrame(ci_rows).to_string(index=False))

print("\n" + "=" * 70)
print("TOP 20% CONFUSION MATRIX")
print("=" * 70)

top20 = capacity_results.loc[
    np.isclose(capacity_results["capacity"], 0.20)
].iloc[0]

print(f"TP: {int(top20.tp)} | FP: {int(top20.fp)}")
print(f"FN: {int(top20.fn)} | TN: {int(top20.tn)}")
print(f"Accuracy: {top20.accuracy:.4f}")
print(f"Precision: {top20.precision:.4f}")
print(f"Recall: {top20.recall:.4f}")
print(f"F1: {top20.f1:.4f}")

print("\nAUDIT COMPLETE")
print("Reports saved to:", OUT)
print("Champion model unchanged.")
