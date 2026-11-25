from pathlib import Path
import json
import math
import warnings

import joblib
import nbformat
import numpy as np
import pandas as pd

from IPython.display import display
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss
)

ROOT = Path.cwd()
OUT = ROOT / "reports" / "portfolio_audit"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
BUDGET = 0.20
rng = np.random.default_rng(SEED)

print("\nRETAILAI | PORTFOLIO QUALITY AUDIT")

# --------------------------------------------------
# 1. Reconstruct canonical dataset
# --------------------------------------------------

nb = nbformat.read(
    ROOT / "notebooks/06_model_evaluation.ipynb",
    as_version=4
)

ns = {"__name__": "__main__", "display": display}

for i in range(6):
    print(f"Preparing data {i + 1}/6", flush=True)
    exec(
        compile(nb.cells[i].source, f"cell_{i+1}", "exec"),
        ns
    )

df = ns["df"].copy()

if "promised_delivery_days" not in df.columns:
    assert "promised_days" in df.columns
    df["promised_delivery_days"] = df["promised_days"]

df = df.sort_values(
    ["order_purchase_timestamp", "order_id"]
).reset_index(drop=True)

assert df["order_id"].is_unique
assert df["late_delivery"].isin([0, 1]).all()

n = len(df)
train = df.iloc[:int(n * 0.70)]
validation = df.iloc[int(n * 0.70):int(n * 0.85)]
test = df.iloc[int(n * 0.85):].copy()

champion = joblib.load(
    ROOT / "models/late_delivery_risk_model.joblib"
)

features = list(champion["feature_columns"])
missing = sorted(set(features) - set(test.columns))

if missing:
    raise ValueError(f"Missing champion features: {missing}")

def score(frame):
    X = frame[features]
    if "pipeline" in champion:
        return champion["pipeline"].predict_proba(X)[:, 1]
    transformed = champion["preprocessor"].transform(X)
    return champion["model"].predict_proba(transformed)[:, 1]

def at_budget(y, p, budget=BUDGET):
    y = np.asarray(y)
    p = np.asarray(p)
    k = max(1, math.ceil(len(y) * budget))
    chosen = np.argsort(-p, kind="stable")[:k]
    captured = int(y[chosen].sum())
    positives = int(y.sum())
    return {
        "orders_flagged": k,
        "late_captured": captured,
        "recall_at_20": captured / positives if positives else np.nan,
        "precision_at_20": captured / k,
        "lift_at_20": (
            (captured / k) / y.mean() if y.mean() else np.nan
        )
    }

# --------------------------------------------------
# 2. Feature availability / leakage checklist
# --------------------------------------------------

safe_candidates = {
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "customer_state",
    "seller_state",
    "item_count",
    "total_item_price",
    "total_freight",
    "unique_products",
    "unique_sellers",
    "promised_delivery_days",
    "promised_days",
    "approval_lag_hours",
    "approx_distance_km",
    "same_seller_customer_state",
    "total_product_weight_g",
    "total_product_volume_cm3"
}

historical_candidates = {
    "seller_prior_orders",
    "seller_prior_late_rate"
}

blocked_tokens = [
    "actual_delivery",
    "delivered_customer",
    "review_score",
    "review_comment",
    "delivery_delay",
    "days_late"
]

audit = []

for feature in features:
    if any(token in feature.lower() for token in blocked_tokens):
        status = "BLOCKED: possible post-outcome information"
    elif feature in historical_candidates:
        status = "REVIEW: historical as-of availability"
    elif feature in safe_candidates:
        status = "CONDITIONALLY AVAILABLE: verify source timing"
    else:
        status = "MANUAL REVIEW REQUIRED"

    audit.append({
        "feature": feature,
        "availability_assessment": status
    })

feature_audit = pd.DataFrame(audit)
feature_audit.to_csv(
    OUT / "feature_availability_audit.csv",
    index=False
)

print("\nFEATURE AVAILABILITY")
print(feature_audit.to_string(index=False))

# --------------------------------------------------
# 3. Historical feature temporal diagnostics
# --------------------------------------------------

temporal_checks = []

for name, frame in [
    ("train", train),
    ("validation", validation),
    ("test", test)
]:
    temporal_checks.append({
        "split": name,
        "orders": len(frame),
        "late_rate": float(frame["late_delivery"].mean()),
        "first_purchase": str(frame["order_purchase_timestamp"].min()),
        "last_purchase": str(frame["order_purchase_timestamp"].max())
    })

pd.DataFrame(temporal_checks).to_csv(
    OUT / "temporal_split_audit.csv",
    index=False
)

# A genuine as-of audit requires source-event timestamps.
# A chronological split alone cannot certify feature availability.

history_audit = {
    "status": "NOT CERTIFIED",
    "requirement": (
        "For every order, historical seller features must use "
        "only outcomes observable before that order's approval timestamp."
    ),
    "validation_gap": (
        "Historical seller aggregation implementation and event "
        "availability timestamps require independent inspection."
    ),
    "recommendation": (
        "Do not claim leakage-free historical features until "
        "an as-of join or equivalent event-time audit passes."
    )
}

(OUT / "historical_feature_audit.json").write_text(
    json.dumps(history_audit, indent=2)
)

# --------------------------------------------------
# 4. Champion performance
# --------------------------------------------------

y = test["late_delivery"].to_numpy(dtype=int)
p = score(test)

metrics = at_budget(y, p)
metrics.update({
    "test_orders": len(y),
    "late_orders": int(y.sum()),
    "late_prevalence": float(y.mean()),
    "pr_auc": float(average_precision_score(y, p)),
    "roc_auc": float(roc_auc_score(y, p)),
    "brier": float(brier_score_loss(y, p))
})

pd.DataFrame([metrics]).to_csv(
    OUT / "champion_metrics.csv",
    index=False
)

# --------------------------------------------------
# 5. Bootstrap uncertainty intervals
# --------------------------------------------------

bootstrap = []

for _ in range(1000):
    indices = rng.integers(0, len(y), len(y))
    ys = y[indices]
    ps = p[indices]

    if ys.sum() == 0 or ys.sum() == len(ys):
        continue

    result = at_budget(ys, ps)
    result["pr_auc"] = average_precision_score(ys, ps)
    bootstrap.append(result)

boot = pd.DataFrame(bootstrap)

intervals = []

for metric in [
    "recall_at_20",
    "precision_at_20",
    "lift_at_20",
    "pr_auc"
]:
    intervals.append({
        "metric": metric,
        "estimate": metrics[metric],
        "ci_low": boot[metric].quantile(0.025),
        "ci_high": boot[metric].quantile(0.975)
    })

pd.DataFrame(intervals).to_csv(
    OUT / "bootstrap_intervals.csv",
    index=False
)

# --------------------------------------------------
# 6. Calibration / ranking diagnostics
# --------------------------------------------------

calibration = pd.DataFrame({
    "actual": y,
    "predicted": p
})

calibration["risk_decile"] = pd.qcut(
    calibration["predicted"].rank(method="first"),
    q=10,
    labels=False
) + 1

calibration_summary = (
    calibration.groupby("risk_decile")
    .agg(
        orders=("actual", "size"),
        actual_late_rate=("actual", "mean"),
        mean_predicted_risk=("predicted", "mean")
    )
    .reset_index()
)

calibration_summary.to_csv(
    OUT / "risk_calibration_deciles.csv",
    index=False
)

# --------------------------------------------------
# 7. A/B experiment sample-size calculation
# --------------------------------------------------

def sample_size_two_proportions(
    control_rate,
    treatment_rate,
    alpha=0.05,
    power=0.80
):
    if not (0 < control_rate < 1 and 0 < treatment_rate < 1):
        raise ValueError("Rates must be between 0 and 1.")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    pooled = (control_rate + treatment_rate) / 2

    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power * math.sqrt(
            control_rate * (1 - control_rate)
            + treatment_rate * (1 - treatment_rate)
        )
    ) ** 2

    return math.ceil(
        numerator / (control_rate - treatment_rate) ** 2
    )

# Hypothetical experiment assumptions, not observed effects.
control_rate = float(metrics["precision_at_20"])
relative_reductions = [0.10, 0.15, 0.20, 0.25]

experiments = []

for reduction in relative_reductions:
    treatment_rate = control_rate * (1 - reduction)
    per_group = sample_size_two_proportions(
        control_rate,
        treatment_rate
    )

    experiments.append({
        "assumed_relative_reduction": reduction,
        "control_late_rate": control_rate,
        "hypothetical_treatment_late_rate": treatment_rate,
        "sample_per_group": per_group,
        "total_sample": 2 * per_group,
        "alpha": 0.05,
        "power": 0.80
    })

pd.DataFrame(experiments).to_csv(
    OUT / "ab_test_sample_sizes.csv",
    index=False
)

# --------------------------------------------------
# 8. Business simulation
# --------------------------------------------------

business_scenarios = []

flagged = metrics["orders_flagged"]
captured = metrics["late_captured"]

for intervention_cost in [2, 5, 10]:
    for success_rate in [0.10, 0.20, 0.30]:
        for cost_per_late_order in [25, 50, 100]:
            expected_prevented = captured * success_rate
            gross_benefit = (
                expected_prevented * cost_per_late_order
            )
            total_cost = flagged * intervention_cost
            net_benefit = gross_benefit - total_cost

            business_scenarios.append({
                "intervention_cost": intervention_cost,
                "assumed_success_rate": success_rate,
                "cost_per_late_order": cost_per_late_order,
                "orders_flagged": flagged,
                "late_orders_identified": captured,
                "expected_late_orders_prevented": expected_prevented,
                "gross_benefit": gross_benefit,
                "total_intervention_cost": total_cost,
                "net_benefit": net_benefit,
                "roi": (
                    net_benefit / total_cost
                    if total_cost else np.nan
                )
            })

pd.DataFrame(business_scenarios).to_csv(
    OUT / "business_scenarios.csv",
    index=False
)

# --------------------------------------------------
# 9. Reproducible scoring output
# --------------------------------------------------

scored = pd.DataFrame({
    "order_id": test["order_id"].to_numpy(),
    "purchase_timestamp": test[
        "order_purchase_timestamp"
    ].to_numpy(),
    "actual_late": y,
    "predicted_risk": p
})

scored["risk_rank"] = scored["predicted_risk"].rank(
    ascending=False,
    method="first"
)

scored["flag_for_intervention"] = (
    scored["risk_rank"] <= flagged
)

scored.to_csv(
    OUT / "retrospective_scored_orders.csv",
    index=False
)

summary = {
    "model": "V1 Random Forest",
    "prediction_stage": "post_approval",
    "test_period": "historical retrospective",
    "test_previously_inspected": True,
    "historical_features_certified": False,
    "actual_ab_test_conducted": False,
    "actual_savings_measured": False,
    "test_metrics": metrics
}

(OUT / "audit_summary.json").write_text(
    json.dumps(summary, indent=2)
)

print("\nCHAMPION METRICS")
print(pd.DataFrame([metrics]).to_string(index=False))

print("\n95% BOOTSTRAP INTERVALS")
print(pd.DataFrame(intervals).to_string(index=False))

print("\nAudit complete.")
print("Reports:", OUT)
print("Existing model and notebooks unchanged.")
