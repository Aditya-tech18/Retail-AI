from pathlib import Path
import json
import math

import joblib
import nbformat
import numpy as np
import pandas as pd

from IPython.display import display
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss
)

ROOT = Path.cwd()
OUT = ROOT / "reports/v4_calibration"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("RETAILAI V4 | VALIDATION-BASED CALIBRATION")
print("=" * 65)

# 1. Reconstruct the existing evaluation dataset.
nb = nbformat.read(
    ROOT / "notebooks/06_model_evaluation.ipynb",
    as_version=4
)

ns = {"__name__": "__main__", "display": display}

for i in range(6):
    print(f"Preparing data cell {i + 1}/6")
    exec(
        compile(nb.cells[i].source, f"cell_{i+1}", "exec"),
        ns
    )

df = ns["df"].copy()

if "promised_delivery_days" not in df.columns:
    if "promised_days" not in df.columns:
        raise ValueError("Promised delivery feature missing")
    df["promised_delivery_days"] = df["promised_days"]

df = df.sort_values(
    ["order_purchase_timestamp", "order_id"]
).reset_index(drop=True)

n = len(df)
validation = df.iloc[
    int(n * 0.70):int(n * 0.85)
].copy()
test = df.iloc[int(n * 0.85):].copy()

# 2. Load champion without modifying it.
champion_path = (
    ROOT / "models/late_delivery_risk_model.joblib"
)
champion = joblib.load(champion_path)

features = list(champion["feature_columns"])

def predict_champion(frame):
    X = frame[features]

    if "pipeline" in champion:
        scores = champion["pipeline"].predict_proba(X)[:, 1]
    else:
        transformed = champion["preprocessor"].transform(X)
        scores = champion["model"].predict_proba(transformed)[:, 1]

    return np.asarray(scores, dtype=float)

y_val = validation["late_delivery"].to_numpy(dtype=int)
y_test = test["late_delivery"].to_numpy(dtype=int)

p_val = predict_champion(validation)
p_test = predict_champion(test)

assert np.isfinite(p_val).all()
assert np.isfinite(p_test).all()
assert ((p_val >= 0) & (p_val <= 1)).all()
assert ((p_test >= 0) & (p_test <= 1)).all()

print("\nValidation:", len(y_val))
print("Test:", len(y_test))
print("Validation prevalence:", round(y_val.mean(), 5))
print("Test prevalence:", round(y_test.mean(), 5))

# 3. Chronological calibration-fit / selection split.
# Validation is not randomly shuffled.
cut = int(len(y_val) * 0.60)

y_fit, y_select = y_val[:cut], y_val[cut:]
p_fit, p_select = p_val[:cut], p_val[cut:]

EPS = 1e-6

def safe_logit(prob):
    return logit(np.clip(prob, EPS, 1 - EPS))

# Platt scaling.
platt = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=2000
)

platt.fit(
    safe_logit(p_fit).reshape(-1, 1),
    y_fit
)

def predict_platt(prob):
    return platt.predict_proba(
        safe_logit(prob).reshape(-1, 1)
    )[:, 1]

# Isotonic calibration.
isotonic = IsotonicRegression(
    y_min=0.0,
    y_max=1.0,
    out_of_bounds="clip"
)

isotonic.fit(p_fit, y_fit)

def predict_isotonic(prob):
    return np.asarray(isotonic.predict(prob))

# Constant baseline learned from calibration-fit only.
base_rate = float(y_fit.mean())

def candidate_scores(prob):
    return {
        "uncalibrated": prob,
        "platt": predict_platt(prob),
        "isotonic": predict_isotonic(prob),
        "constant_baseline": np.full(
            len(prob), base_rate
        )
    }

def evaluate(y, prob, name):
    prob = np.clip(prob, EPS, 1 - EPS)

    k = math.ceil(len(y) * 0.20)
    chosen = np.argsort(-prob, kind="stable")[:k]
    captured = int(y[chosen].sum())

    return {
        "model": name,
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(
            y, prob, labels=[0, 1]
        )),
        "mean_predicted": float(prob.mean()),
        "actual_prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "recall_at_20": (
            captured / int(y.sum())
            if y.sum() else np.nan
        ),
        "late_captured": captured
    }

# 4. Select on later validation only.
selection_rows = [
    evaluate(y_select, prob, name)
    for name, prob in candidate_scores(p_select).items()
]

selection = pd.DataFrame(selection_rows)
selection.to_csv(
    OUT / "validation_selection.csv",
    index=False
)

# Constant baseline is a benchmark, not a risk model.
# Compare actual calibration strategies against it.
eligible = selection[
    selection["model"].isin([
        "uncalibrated", "platt", "isotonic"
    ])
]

winner = eligible.sort_values(
    ["brier", "log_loss"]
).iloc[0]["model"]

# 5. Evaluate previously inspected test period.
test_candidates = candidate_scores(p_test)

test_rows = [
    evaluate(y_test, prob, name)
    for name, prob in test_candidates.items()
]

test_results = pd.DataFrame(test_rows)
test_results.to_csv(
    OUT / "test_calibration_comparison.csv",
    index=False
)

# 6. Save fitted calibration objects.
# Keep champion unchanged.
artifact = {
    "champion_path": str(champion_path),
    "feature_columns": features,
    "selected_calibration": winner,
    "platt": platt,
    "isotonic": isotonic,
    "fit_base_rate": base_rate,
    "prediction_stage": "post_approval",
    "note": (
        "Calibrators fitted on the first 60% of the "
        "validation period; selected on the later 40%. "
        "Test data was not used to fit or select them."
    )
}

joblib.dump(
    artifact,
    ROOT / "models/v4_calibration_bundle.joblib"
)

# 7. Reliability diagram data.
reliability_rows = []

for name, prob in test_candidates.items():
    if name == "constant_baseline":
        continue

    temp = pd.DataFrame({
        "actual": y_test,
        "predicted": prob
    })

    temp["bin"] = pd.cut(
        temp["predicted"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True
    )

    grouped = temp.groupby(
        "bin", observed=False
    ).agg(
        count=("actual", "size"),
        observed_rate=("actual", "mean"),
        predicted_rate=("predicted", "mean")
    ).reset_index()

    grouped["model"] = name
    grouped["bin"] = grouped["bin"].astype(str)
    reliability_rows.append(grouped)

pd.concat(
    reliability_rows,
    ignore_index=True
).to_csv(
    OUT / "reliability_curves.csv",
    index=False
)

# 8. Documentation of unresolved risks.
status = {
    "selected_using_validation": winner,
    "calibration_fit_on_test": False,
    "test_previously_inspected": True,
    "fresh_blind_test": False,
    "historical_seller_features_audited": False,
    "feature_availability_fully_certified": False,
    "model_retrained": False,
    "champion_overwritten": False,
    "selection_metric": "validation Brier score",
    "warning": (
        "Selection uses a limited validation window. "
        "The reported test period has been inspected "
        "during earlier experiments and is not a "
        "fresh blind holdout."
    )
}

(OUT / "methodology.json").write_text(
    json.dumps(status, indent=2)
)

print("\nVALIDATION SELECTION")
print(selection.to_string(index=False))

print("\nSELECTED CALIBRATION:", winner)

print("\nTEST CALIBRATION COMPARISON")
print(test_results.to_string(index=False))

print("\nSaved:", OUT)
print("Original champion unchanged.")
