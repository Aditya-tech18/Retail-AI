from pathlib import Path
from datetime import datetime
import shutil
import nbformat as nbf

ROOT = Path.cwd()
TARGET = ROOT / "notebooks/06_model_evaluation.ipynb"

if TARGET.exists():
    backup = ROOT / (
        "06_model_evaluation_backup_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".ipynb"
    )
    shutil.copy2(TARGET, backup)
    print("Backup:", backup)

cells = []

def add(title, source):
    cells.append(
        nbf.v4.new_code_cell(
            f"# {'=' * 75}\n"
            f"# {title}\n"
            f"# {'=' * 75}\n\n"
            + source.strip()
        )
    )


# ============================================================
# CELL 1 — BUSINESS PROBLEM AND EXPERIMENT DESIGN
# ============================================================

add("01 | Business Objective and Modeling Contract", r'''
"""
RETAILAI — LATE DELIVERY RISK PREDICTION

Prediction time:
    Immediately after purchase / order approval.

Target:
    1 = delivered after estimated delivery date
    0 = delivered on or before estimated delivery date

Business objective:
    Identify potentially late orders early enough to support
    fulfillment intervention or proactive communication.

Evaluation:
    - Chronological train / validation / test
    - Baseline comparison
    - PR-AUC, ROC-AUC, recall, precision
    - Threshold selection on validation only
    - Probability calibration
    - Segment-level robustness
    - Explainability and business cost analysis

Important limitations:
    Only delivered orders have observable delivery outcomes.
    Canceled and unavailable orders are excluded.
    The sample is therefore conditional on delivery completion.

    This is a historical retrospective experiment.
    No causal benefit or production readiness is claimed.
"""

from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit,
    RandomizedSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
    log_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp

from xgboost import XGBClassifier

RANDOM_STATE = 42
ROOT = Path.cwd()

if not (ROOT / "datasets").exists():
    ROOT = ROOT.parent

DATA = ROOT / "datasets/processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

REPORTS.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

print("RetailAI | Late Delivery Risk Modeling")
print("Random seed:", RANDOM_STATE)
print("Project:", ROOT)
''')


# ============================================================
# CELL 2 — DATA LOADING AND SCHEMA VALIDATION
# ============================================================

add("02 | Data Contracts and Source Validation", r'''
orders = pd.read_parquet(DATA / "orders.parquet")
items = pd.read_parquet(DATA / "order_items.parquet")
customers = pd.read_parquet(DATA / "customers.parquet")

required_orders = {
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
}

required_items = {
    "order_id",
    "order_item_id",
    "price",
    "freight_value",
    "product_id",
    "seller_id"
}

required_customers = {
    "customer_id",
    "customer_state"
}

for name, frame, required in [
    ("orders", orders, required_orders),
    ("items", items, required_items),
    ("customers", customers, required_customers)
]:
    missing = required - set(frame.columns)
    assert not missing, f"{name}: missing columns {missing}"

assert orders["order_id"].is_unique
assert customers["customer_id"].is_unique
assert orders["order_id"].notna().all()

for col in [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]:
    orders[col] = pd.to_datetime(
        orders[col], errors="coerce"
    )

for col in ["price", "freight_value"]:
    items[col] = pd.to_numeric(
        items[col], errors="coerce"
    )

assert items["price"].dropna().ge(0).all()
assert items["freight_value"].dropna().ge(0).all()

quality_report = pd.DataFrame([
    {
        "table": name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum())
    }
    for name, frame in [
        ("orders", orders),
        ("items", items),
        ("customers", customers)
    ]
])

display(quality_report)
print("Source contracts passed.")
''')


# ============================================================
# CELL 3 — TARGET ENGINEERING AND LEAKAGE AUDIT
# ============================================================

add("03 | Target Engineering and Leakage Prevention", r'''
# Aggregate order items before joining to order-level data.
item_features = (
    items.groupby("order_id", as_index=False)
    .agg(
        item_count=("order_item_id", "size"),
        total_item_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        unique_products=("product_id", "nunique"),
        unique_sellers=("seller_id", "nunique")
    )
)

data = (
    orders.merge(
        item_features,
        on="order_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        customers[
            ["customer_id", "customer_state"]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )
)

assert len(data) == len(orders)
assert data["order_id"].is_unique

# Define the observable outcome population.
data = data.loc[
    data["order_status"].eq("delivered")
].copy()

data = data.dropna(
    subset=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
)

data["actual_delivery_days"] = (
    data["order_delivered_customer_date"]
    - data["order_purchase_timestamp"]
).dt.total_seconds() / 86400

data["promised_delivery_days"] = (
    data["order_estimated_delivery_date"]
    - data["order_purchase_timestamp"]
).dt.total_seconds() / 86400

data = data.loc[
    data["actual_delivery_days"].ge(0)
    & data["promised_delivery_days"].gt(0)
].copy()

data["late_delivery"] = (
    data["order_delivered_customer_date"]
    > data["order_estimated_delivery_date"]
).astype(int)

# Candidate features available at purchase or approval time.
data["purchase_month"] = (
    data["order_purchase_timestamp"].dt.month
)

data["purchase_weekday"] = (
    data["order_purchase_timestamp"].dt.dayofweek
)

data["purchase_hour"] = (
    data["order_purchase_timestamp"].dt.hour
)

data["approval_lag_hours"] = (
    data["order_approved_at"]
    - data["order_purchase_timestamp"]
).dt.total_seconds() / 3600

# Only retain nonnegative approval lags.
data.loc[
    data["approval_lag_hours"] < 0,
    "approval_lag_hours"
] = np.nan

# Approval lag is valid only for a post-approval prediction.
# Missing approval timestamps remain missing.

numeric_features = [
    "promised_delivery_days",
    "approval_lag_hours",
    "item_count",
    "total_item_price",
    "total_freight",
    "unique_products",
    "unique_sellers",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour"
]

categorical_features = [
    "customer_state"
]

feature_columns = (
    numeric_features + categorical_features
)

forbidden_features = {
    "order_delivered_customer_date",
    "actual_delivery_days",
    "late_delivery",
    "order_delivered_carrier_date",
    "order_status",
    "order_id",
    "customer_id"
}

assert not (
    forbidden_features & set(feature_columns)
), "Target leakage detected."

assert data["late_delivery"].nunique() == 2

print("Modeling population:", len(data))
print("Late delivery rate:", data["late_delivery"].mean())
print("Features:", feature_columns)
print("Leakage audit: PASSED")

display(
    data[
        feature_columns + ["late_delivery"]
    ].head()
)
''')


# ============================================================
# CELL 4 — CHRONOLOGICAL SPLIT
# ============================================================

add("04 | Chronological Train, Validation and Test Split", r'''
"""
A random split can make a historical forecasting
experiment unrealistically optimistic.

Use:
    Earliest 70% -> training
    Next 15%     -> validation
    Latest 15%   -> test

Test remains untouched until final evaluation.
"""

data = data.sort_values(
    ["order_purchase_timestamp", "order_id"]
).reset_index(drop=True)

n = len(data)
train_end = int(n * 0.70)
valid_end = int(n * 0.85)

train_df = data.iloc[:train_end].copy()
valid_df = data.iloc[train_end:valid_end].copy()
test_df = data.iloc[valid_end:].copy()

assert (
    train_df["order_purchase_timestamp"].max()
    <= valid_df["order_purchase_timestamp"].min()
)

assert (
    valid_df["order_purchase_timestamp"].max()
    <= test_df["order_purchase_timestamp"].min()
)

assert set(train_df["order_id"]).isdisjoint(
    valid_df["order_id"]
)

assert set(valid_df["order_id"]).isdisjoint(
    test_df["order_id"]
)

X_train = train_df[feature_columns]
X_valid = valid_df[feature_columns]
X_test = test_df[feature_columns]

y_train = train_df["late_delivery"]
y_valid = valid_df["late_delivery"]
y_test = test_df["late_delivery"]

split_report = pd.DataFrame([
    {
        "split": name,
        "rows": len(frame),
        "late_rate": frame["late_delivery"].mean(),
        "start": frame["order_purchase_timestamp"].min(),
        "end": frame["order_purchase_timestamp"].max()
    }
    for name, frame in [
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df)
    ]
])

display(split_report)

assert all(
    y.nunique() == 2
    for y in [y_train, y_valid, y_test]
)

print("Chronological split validated.")
''')


# ============================================================
# CELL 5 — PREPROCESSING PIPELINE
# ============================================================

add("05 | Reproducible Feature Preprocessing", r'''
"""
Preprocessing is fitted on training data only.

Numeric:
    median imputation + standardization

Categorical:
    most frequent imputation + one-hot encoding

Unknown categories are handled safely at inference.
"""

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop",
    sparse_threshold=0
)

# Fit only to training data.
X_train_processed = preprocessor.fit_transform(
    X_train
)

X_valid_processed = preprocessor.transform(
    X_valid
)

X_test_processed = preprocessor.transform(
    X_test
)

feature_names = (
    preprocessor.get_feature_names_out()
)

assert X_train_processed.shape[1] == len(feature_names)
assert X_valid_processed.shape[1] == len(feature_names)
assert X_test_processed.shape[1] == len(feature_names)

assert np.isfinite(X_train_processed).all()
assert np.isfinite(X_valid_processed).all()
assert np.isfinite(X_test_processed).all()

print("Training matrix:", X_train_processed.shape)
print("Validation matrix:", X_valid_processed.shape)
print("Test matrix:", X_test_processed.shape)
print("Encoded features:", len(feature_names))
''')


# ============================================================
# CELL 6 — BASELINE AND MODEL COMPARISON
# ============================================================

add("06 | Baseline, Logistic Regression, Random Forest, XGBoost", r'''
"""
The baseline prevents a sophisticated model from being
celebrated when it fails to beat a simple reference.

Model selection uses validation metrics only.
"""

models = {
    "Dummy Baseline": DummyClassifier(
        strategy="prior"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE
    )
}

fitted_models = {}
validation_predictions = {}
comparison_rows = []

for name, model in models.items():
    print("Training:", name)

    model.fit(X_train_processed, y_train)

    probabilities = model.predict_proba(
        X_valid_processed
    )[:, 1]

    fitted_models[name] = model
    validation_predictions[name] = probabilities

    comparison_rows.append({
        "model": name,
        "pr_auc": average_precision_score(
            y_valid, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_valid, probabilities
        ),
        "brier_score": brier_score_loss(
            y_valid, probabilities
        ),
        "precision_at_0_5": precision_score(
            y_valid,
            probabilities >= 0.5,
            zero_division=0
        ),
        "recall_at_0_5": recall_score(
            y_valid,
            probabilities >= 0.5,
            zero_division=0
        )
    })

model_comparison = (
    pd.DataFrame(comparison_rows)
    .sort_values(
        ["pr_auc", "brier_score"],
        ascending=[False, True]
    )
)

display(model_comparison)

print(
    "Validation prevalence:",
    round(y_valid.mean(), 4)
)
''')


# ============================================================
# CELL 7 — HYPERPARAMETER TUNING
# ============================================================

add("07 | Time-Aware Hyperparameter Optimization", r'''
"""
Tune only within the training period.

TimeSeriesSplit preserves chronological ordering
inside the training sample.

The untouched validation set remains available
for model selection and threshold optimization.
"""

xgb_search_space = {
    "n_estimators": [150, 250, 350, 500],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.025, 0.05, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.75, 0.9, 1.0],
    "colsample_bytree": [0.75, 0.9, 1.0],
    "reg_lambda": [1, 5, 10]
}

cv = TimeSeriesSplit(n_splits=3)

search = RandomizedSearchCV(
    estimator=XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=2,
        random_state=RANDOM_STATE
    ),
    param_distributions=xgb_search_space,
    n_iter=12,
    scoring="average_precision",
    cv=cv,
    refit=True,
    n_jobs=2,
    random_state=RANDOM_STATE,
    verbose=1,
    error_score="raise"
)

search.fit(
    X_train_processed,
    y_train
)

tuned_xgb = search.best_estimator_

tuned_prob = tuned_xgb.predict_proba(
    X_valid_processed
)[:, 1]

tuned_result = {
    "model": "Tuned XGBoost",
    "pr_auc": average_precision_score(
        y_valid, tuned_prob
    ),
    "roc_auc": roc_auc_score(
        y_valid, tuned_prob
    ),
    "brier_score": brier_score_loss(
        y_valid, tuned_prob
    ),
    "precision_at_0_5": precision_score(
        y_valid, tuned_prob >= 0.5,
        zero_division=0
    ),
    "recall_at_0_5": recall_score(
        y_valid, tuned_prob >= 0.5,
        zero_division=0
    )
}

model_comparison = pd.concat(
    [
        model_comparison,
        pd.DataFrame([tuned_result])
    ],
    ignore_index=True
).sort_values(
    ["pr_auc", "brier_score"],
    ascending=[False, True]
)

fitted_models["Tuned XGBoost"] = tuned_xgb
validation_predictions["Tuned XGBoost"] = tuned_prob

display(model_comparison)
print("Best CV parameters:", search.best_params_)
print("Best training CV PR-AUC:", search.best_score_)
''')


# ============================================================
# CELL 8 — VALIDATION METRICS
# ============================================================

add("08 | Precision-Recall and ROC Diagnostics", r'''
"""
PR-AUC is useful when positive cases are less common.
ROC-AUC measures ranking discrimination.

Neither metric alone determines business usefulness.
"""

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for name, probabilities in validation_predictions.items():
    precision, recall, _ = precision_recall_curve(
        y_valid, probabilities
    )

    fpr, tpr, _ = roc_curve(
        y_valid, probabilities
    )

    ap = average_precision_score(
        y_valid, probabilities
    )

    auc = roc_auc_score(
        y_valid, probabilities
    )

    axes[0].plot(
        recall, precision,
        label=f"{name} ({ap:.3f})"
    )

    axes[1].plot(
        fpr, tpr,
        label=f"{name} ({auc:.3f})"
    )

axes[0].axhline(
    y_valid.mean(),
    linestyle="--",
    color="gray",
    label="Prevalence"
)

axes[0].set_title("Validation Precision-Recall")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")

axes[1].plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray"
)

axes[1].set_title("Validation ROC")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")

for ax in axes:
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

selected_name = model_comparison.iloc[0]["model"]
selected_model = fitted_models[selected_name]

valid_prob = validation_predictions[selected_name]

print("Selected using validation PR-AUC:", selected_name)
''')


# ============================================================
# CELL 9 — BUSINESS COST AND THRESHOLD
# ============================================================

add("09 | Decision Threshold and Business Cost", r'''
"""
Illustrative decision costs, not observed company costs.

False negative:
    Late delivery missed -> potential service failure.

False positive:
    On-time delivery incorrectly flagged -> unnecessary action.

We select a threshold using validation data only.
"""

FALSE_NEGATIVE_COST = 100.0
FALSE_POSITIVE_COST = 15.0

thresholds = np.linspace(0.05, 0.95, 181)

threshold_rows = []

for threshold in thresholds:
    predicted = (
        valid_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_valid, predicted,
        labels=[0, 1]
    ).ravel()

    cost = (
        fn * FALSE_NEGATIVE_COST
        + fp * FALSE_POSITIVE_COST
    )

    threshold_rows.append({
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision_score(
            y_valid, predicted,
            zero_division=0
        ),
        "recall": recall_score(
            y_valid, predicted,
            zero_division=0
        ),
        "estimated_cost": cost
    })

threshold_table = pd.DataFrame(threshold_rows)

best_threshold_row = threshold_table.loc[
    threshold_table["estimated_cost"].idxmin()
]

selected_threshold = float(
    best_threshold_row["threshold"]
)

display(
    threshold_table.nsmallest(
        10, "estimated_cost"
    )
)

plt.figure(figsize=(11, 5))
plt.plot(
    threshold_table["threshold"],
    threshold_table["estimated_cost"]
)
plt.axvline(
    selected_threshold,
    linestyle="--",
    label=f"Selected = {selected_threshold:.2f}"
)
plt.xlabel("Classification Threshold")
plt.ylabel("Illustrative Decision Cost")
plt.title("Validation Cost vs Decision Threshold")
plt.legend()
plt.tight_layout()
plt.show()

print("Selected threshold:", selected_threshold)
print(
    "Warning: Costs are hypothetical. "
    "Replace with actual business estimates."
)
''')


# ============================================================
# CELL 10 — CALIBRATION
# ============================================================

add("10 | Probability Calibration and Reliability", r'''
"""
A model with strong ranking may still produce
poorly calibrated probabilities.

Calibration matters when decisions are based on
risk probabilities or expected costs.
"""

prob_true, prob_pred = calibration_curve(
    y_valid,
    valid_prob,
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(8, 6))

plt.plot(
    [0, 1], [0, 1],
    "--", label="Perfect calibration"
)

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label=selected_name
)

plt.xlabel("Predicted Late-Delivery Probability")
plt.ylabel("Observed Late-Delivery Frequency")
plt.title("Validation Reliability Diagram")
plt.legend()
plt.tight_layout()
plt.show()

calibration_metrics = pd.DataFrame([{
    "model": selected_name,
    "brier_score": brier_score_loss(
        y_valid, valid_prob
    ),
    "log_loss": log_loss(
        y_valid, valid_prob
    ),
    "mean_predicted_risk": valid_prob.mean(),
    "observed_late_rate": y_valid.mean()
}])

display(calibration_metrics)

print(
    "Calibration assessed, not automatically corrected. "
    "A calibrated production model would require "
    "a separate calibration design."
)
''')


# ============================================================
# CELL 11 — FINAL HELD-OUT TEST
# ============================================================

add("11 | Final Held-Out Test Evaluation", r'''
"""
The test set is evaluated once after model
and threshold selection.

Do not adjust the model or threshold using test results.
"""

test_prob = selected_model.predict_proba(
    X_test_processed
)[:, 1]

test_pred = (
    test_prob >= selected_threshold
).astype(int)

test_metrics = pd.DataFrame([{
    "model": selected_name,
    "threshold": selected_threshold,
    "test_rows": len(y_test),
    "positive_rate": y_test.mean(),
    "pr_auc": average_precision_score(
        y_test, test_prob
    ),
    "roc_auc": roc_auc_score(
        y_test, test_prob
    ),
    "precision": precision_score(
        y_test, test_pred,
        zero_division=0
    ),
    "recall": recall_score(
        y_test, test_pred,
        zero_division=0
    ),
    "f1": f1_score(
        y_test, test_pred,
        zero_division=0
    ),
    "brier_score": brier_score_loss(
        y_test, test_prob
    )
}])

display(test_metrics)

matrix = confusion_matrix(
    y_test, test_pred,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["On time", "Late"],
    yticklabels=["On time", "Late"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Held-Out Test Confusion Matrix")
plt.tight_layout()
plt.show()

print(
    classification_report(
        y_test,
        test_pred,
        zero_division=0
    )
)
''')


# ============================================================
# CELL 12 — SEGMENT-LEVEL ERROR ANALYSIS
# ============================================================

add("12 | Geographic and Temporal Robustness", r'''
"""
Global model metrics can hide weak performance
in specific states or purchase periods.

Report segment sample sizes to avoid
overinterpreting small groups.
"""

diagnostics = test_df[
    [
        "order_id",
        "customer_state",
        "purchase_month",
        "promised_delivery_days"
    ]
].copy()

diagnostics["actual"] = y_test.to_numpy()
diagnostics["predicted"] = test_pred
diagnostics["probability"] = test_prob

def segment_metrics(frame):
    y = frame["actual"]
    p = frame["probability"]
    pred = frame["predicted"]

    return pd.Series({
        "orders": len(frame),
        "actual_late_rate": y.mean(),
        "predicted_late_rate": pred.mean(),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "pr_auc": (
            average_precision_score(y, p)
            if y.nunique() == 2 else np.nan
        )
    })

state_diagnostics = (
    diagnostics.groupby("customer_state")
    .apply(
        segment_metrics,
        include_groups=False
    )
    .reset_index()
)

state_diagnostics = state_diagnostics.loc[
    state_diagnostics["orders"] >= 100
].sort_values("recall")

month_diagnostics = (
    diagnostics.groupby("purchase_month")
    .apply(
        segment_metrics,
        include_groups=False
    )
    .reset_index()
)

display(state_diagnostics)
display(month_diagnostics)

print(
    "Segment diagnostics are descriptive. "
    "Differences may reflect prevalence and sample size."
)
''')


# ============================================================
# CELL 13 — SHAP EXPLAINABILITY
# ============================================================

add("13 | SHAP Feature Attribution", r'''
"""
SHAP provides model-level feature attribution.

Important:
    Explanations describe model behavior.
    They do not prove causal effects.
"""

import shap

# Limit computation to a reproducible sample.
rng = np.random.default_rng(RANDOM_STATE)

sample_size = min(
    500,
    len(X_test_processed)
)

sample_idx = rng.choice(
    len(X_test_processed),
    size=sample_size,
    replace=False
)

X_explain = X_test_processed[sample_idx]

if isinstance(
    selected_model,
    (XGBClassifier, RandomForestClassifier)
):
    explainer = shap.TreeExplainer(
        selected_model
    )

    shap_values = explainer.shap_values(
        X_explain
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

elif isinstance(
    selected_model,
    LogisticRegression
):
    explainer = shap.LinearExplainer(
        selected_model,
        X_train_processed
    )

    shap_values = explainer.shap_values(
        X_explain
    )

else:
    # A dummy baseline has no meaningful feature attribution.
    shap_values = np.zeros_like(X_explain)

assert shap_values.shape == X_explain.shape

importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": (
        np.abs(shap_values).mean(axis=0)
    )
}).sort_values(
    "mean_absolute_shap",
    ascending=False
)

display(importance.head(20))

plt.figure(figsize=(10, 7))
sns.barplot(
    data=importance.head(15),
    x="mean_absolute_shap",
    y="feature"
)
plt.title("Top Features by Mean Absolute SHAP")
plt.xlabel("Mean |SHAP Value|")
plt.ylabel("")
plt.tight_layout()
plt.show()

print(
    "Feature attribution is not causal evidence."
)
''')


# ============================================================
# CELL 14 — MODEL EXPORT AND GOVERNANCE
# ============================================================

add("14 | Model Artifact, Reproducibility and Governance", r'''
"""
Export:
    - Fitted preprocessing pipeline
    - Selected estimator
    - Selected decision threshold
    - Feature schema
    - Evaluation metadata

The artifact is for offline demonstration.
Deployment requires additional monitoring,
security and data-quality controls.
"""

artifact = {
    "preprocessor": preprocessor,
    "model": selected_model,
    "model_name": selected_name,
    "threshold": selected_threshold,
    "feature_columns": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "random_state": RANDOM_STATE,
    "prediction_stage": "post_approval",
    "target": "late_delivery"
}

model_path = (
    MODELS / "late_delivery_risk_model.joblib"
)

joblib.dump(artifact, model_path)

# Verify the serialized artifact.
loaded = joblib.load(model_path)

sample_raw = X_test.iloc[:10]

sample_processed = (
    loaded["preprocessor"].transform(sample_raw)
)

reloaded_predictions = (
    loaded["model"].predict_proba(
        sample_processed
    )[:, 1]
)

assert np.allclose(
    reloaded_predictions,
    test_prob[:10]
)

model_comparison.to_csv(
    REPORTS / "model_comparison.csv",
    index=False
)

threshold_table.to_csv(
    REPORTS / "model_threshold_analysis.csv",
    index=False
)

test_metrics.to_csv(
    REPORTS / "model_test_metrics.csv",
    index=False
)

state_diagnostics.to_csv(
    REPORTS / "model_state_diagnostics.csv",
    index=False
)

month_diagnostics.to_csv(
    REPORTS / "model_month_diagnostics.csv",
    index=False
)

importance.to_csv(
    REPORTS / "model_shap_importance.csv",
    index=False
)

metadata = {
    "model": selected_name,
    "target": "late_delivery",
    "threshold": selected_threshold,
    "test_pr_auc": float(
        test_metrics.iloc[0]["pr_auc"]
    ),
    "test_roc_auc": float(
        test_metrics.iloc[0]["roc_auc"]
    ),
    "test_recall": float(
        test_metrics.iloc[0]["recall"]
    ),
    "test_precision": float(
        test_metrics.iloc[0]["precision"]
    ),
    "train_rows": len(train_df),
    "validation_rows": len(valid_df),
    "test_rows": len(test_df),
    "known_limitations": [
        "Delivered-order selection bias",
        "Historical Brazilian e-commerce dataset",
        "No causal intervention evaluation",
        "No production monitoring",
        "Illustrative business cost assumptions"
    ]
}

metadata_path = (
    MODELS / "late_delivery_model_metadata.json"
)

metadata_path.write_text(
    json.dumps(metadata, indent=2)
)

print("=" * 65)
print("RETAILAI MODEL EVALUATION COMPLETE")
print("=" * 65)
print("Selected model:", selected_name)
print("Decision threshold:", selected_threshold)
print("Model artifact:", model_path)
print("Metadata:", metadata_path)
print("Test PR-AUC:", metadata["test_pr_auc"])
print("Test recall:", metadata["test_recall"])
print("Test precision:", metadata["test_precision"])
print()
print("STATUS: OFFLINE VALIDATION COMPLETE")
print("NOT CLAIMED: PRODUCTION DEPLOYMENT")
''')

assert len(cells) == 14

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (RetailAI NLP)",
            "language": "python",
            "name": "retailai-nlp"
        }
    }
)

nbf.write(notebook, TARGET)

print("Notebook:", TARGET)
print("Total cells:", len(cells))
print("Ready for execution.")
