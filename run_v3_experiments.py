from pathlib import Path
import json
import warnings

import joblib
import nbformat
import numpy as np
import pandas as pd

from IPython.display import display
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path.cwd()
REPORT = ROOT / "reports" / "v3_experiments"
REPORT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
ALERT_BUDGET = 0.20

print("\n" + "=" * 75)
print("RETAILAI V3 — ADVANCED MODELING EXPERIMENT")
print("=" * 75)

# ============================================================
# STEP 1: RECONSTRUCT DATA FROM EXISTING NOTEBOOK
# ============================================================

notebook = nbformat.read(
    ROOT / "notebooks/06_model_evaluation.ipynb",
    as_version=4
)

namespace = {
    "__name__": "__main__",
    "display": display
}

for i in range(6):
    print(f"\nExecuting data preparation cell {i+1}/6")
    exec(
        compile(
            notebook.cells[i].source,
            f"cell_{i+1}",
            "exec"
        ),
        namespace
    )

df = namespace["df"].copy()

df["promised_delivery_days"] = df["promised_days"]

df = df.sort_values(
    ["order_purchase_timestamp", "order_id"]
).reset_index(drop=True)

assert df["order_id"].is_unique
assert df["late_delivery"].notna().all()

# ============================================================
# STEP 2: TEMPORAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

print("\nTEMPORAL SPLITS")
for name, frame in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test)
]:
    print(
        name,
        "rows:", len(frame),
        "late rate:", round(frame.late_delivery.mean(), 4)
    )

# ============================================================
# STEP 3: FEATURE GROUPS
# ============================================================

baseline_features = [
    "promised_delivery_days",
    "approval_lag_hours",
    "item_count",
    "total_item_price",
    "total_freight",
    "unique_products",
    "unique_sellers",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "customer_state"
]

product_geo_candidates = [
    "approx_distance_km",
    "total_product_weight_g",
    "total_product_volume_cm3",
    "seller_state",
    "same_seller_customer_state"
]

seller_history_candidates = [
    "seller_prior_orders",
    "seller_prior_late_rate"
]

product_geo_features = [
    f for f in product_geo_candidates
    if f in df.columns
]

seller_history_features = [
    f for f in seller_history_candidates
    if f in df.columns
]

feature_groups = {
    "baseline": baseline_features,
    "baseline_plus_geo": (
        baseline_features + product_geo_features
    ),
    "baseline_plus_seller_history": (
        baseline_features + seller_history_features
    ),
    "all_features": (
        baseline_features
        + product_geo_features
        + seller_history_features
    )
}

for name, features in feature_groups.items():
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name}: missing features {missing}"
        )

print("\nFEATURE ABLATION GROUPS")
for name, features in feature_groups.items():
    print(name, ":", len(features), "features")

# ============================================================
# STEP 4: EVALUATION FUNCTIONS
# ============================================================

def recall_at_budget(y, scores, budget=0.20):
    y = np.asarray(y)
    scores = np.asarray(scores)

    k = int(np.ceil(len(y) * budget))

    selected = np.argsort(
        -scores,
        kind="stable"
    )[:k]

    tp = int(y[selected].sum())
    positives = int(y.sum())

    precision = tp / k
    recall = tp / positives if positives else np.nan
    prevalence = y.mean()

    return {
        "recall_at_20": recall,
        "precision_at_20": precision,
        "lift_at_20": (
            precision / prevalence
            if prevalence > 0 else np.nan
        ),
        "late_captured": tp,
        "orders_flagged": k
    }


def evaluate(y, probabilities):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)

    result = recall_at_budget(y, probabilities)

    result.update({
        "pr_auc": average_precision_score(
            y, probabilities
        ),
        "roc_auc": roc_auc_score(
            y, probabilities
        ),
        "brier": brier_score_loss(
            y, probabilities
        )
    })

    return result



def build_preprocessor(features, frame):

    # Explicit categorical columns prevent state codes
    # such as "SP", "RJ" and "MG" being treated as numbers.
    categorical_candidates = {
        "customer_state",
        "seller_state",
        "product_category_name",
        "purchase_weekday",
        "purchase_month"
    }

    categorical = [
        col for col in features
        if (
            col in categorical_candidates
            or not pd.api.types.is_numeric_dtype(frame[col])
        )
    ]

    numeric = [
        col for col in features
        if col not in categorical
    ]

    print(
        "Categorical:", categorical,
        "| Numeric:", numeric,
        flush=True
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        ))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical)
    ])


# ============================================================
# STEP 5: MODEL CANDIDATES
# ============================================================

def make_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=12,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_lambda=10,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=5,
            learning_rate=0.045,
            min_samples_leaf=40,
            l2_regularization=10,
            random_state=RANDOM_STATE
        )
    }


# ============================================================
# STEP 6: ROLLING TEMPORAL VALIDATION
# ============================================================

# Combine train + validation for temporal CV diagnostics.
# Do not use test for hyperparameter selection.

development = pd.concat(
    [train, validation],
    ignore_index=True
)

tscv = TimeSeriesSplit(n_splits=3)

cv_results = []

for group_name, features in feature_groups.items():

    for model_name, estimator in make_models().items():

        print(
            f"\nTemporal CV: {group_name} | {model_name}",
            flush=True
        )

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            tscv.split(development),
            start=1
        ):
            fold_train = development.iloc[train_idx]
            fold_val = development.iloc[val_idx]

            preprocessor = build_preprocessor(
                features,
                fold_train
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", clone(estimator))
            ])

            pipeline.fit(
                fold_train[features],
                fold_train["late_delivery"]
            )

            predictions = pipeline.predict_proba(
                fold_val[features]
            )[:, 1]

            metrics = evaluate(
                fold_val["late_delivery"],
                predictions
            )

            metrics.update({
                "feature_group": group_name,
                "model": model_name,
                "fold": fold,
                "train_rows": len(fold_train),
                "validation_rows": len(fold_val),
                "validation_prevalence": (
                    fold_val["late_delivery"].mean()
                )
            })

            cv_results.append(metrics)
            fold_scores.append(metrics["recall_at_20"])

            print(
                f"  Fold {fold}: "
                f"Recall@20={metrics['recall_at_20']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}"
            )

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    REPORT / "temporal_cv_folds.csv",
    index=False
)

cv_summary = (
    cv_df
    .groupby(
        ["feature_group", "model"],
        as_index=False
    )
    .agg(
        mean_recall_at_20=("recall_at_20", "mean"),
        std_recall_at_20=("recall_at_20", "std"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_brier=("brier", "mean")
    )
)

cv_summary["stability_score"] = (
    cv_summary["mean_recall_at_20"]
    - 0.5 * cv_summary["std_recall_at_20"].fillna(0)
)

cv_summary = cv_summary.sort_values(
    ["stability_score", "mean_pr_auc"],
    ascending=False
)

cv_summary.to_csv(
    REPORT / "temporal_cv_summary.csv",
    index=False
)

print("\n" + "=" * 75)
print("TEMPORAL CV SUMMARY")
print("=" * 75)
print(cv_summary.to_string(index=False))

# ============================================================
# STEP 7: SELECT TOP CONFIGURATIONS
# ============================================================

top_configs = cv_summary.head(3)

validation_results = []
trained_candidates = {}

for _, config in top_configs.iterrows():

    group_name = config["feature_group"]
    model_name = config["model"]
    features = feature_groups[group_name]

    pipeline = Pipeline([
        (
            "preprocessor",
            build_preprocessor(features, train)
        ),
        (
            "model",
            clone(make_models()[model_name])
        )
    ])

    pipeline.fit(
        train[features],
        train["late_delivery"]
    )

    probabilities = pipeline.predict_proba(
        validation[features]
    )[:, 1]

    metrics = evaluate(
        validation["late_delivery"],
        probabilities
    )

    metrics.update({
        "feature_group": group_name,
        "model": model_name
    })

    validation_results.append(metrics)

    trained_candidates[
        (group_name, model_name)
    ] = (pipeline, features)

validation_df = pd.DataFrame(
    validation_results
).sort_values(
    ["recall_at_20", "pr_auc"],
    ascending=False
)

validation_df.to_csv(
    REPORT / "validation_top_candidates.csv",
    index=False
)

print("\n" + "=" * 75)
print("TOP CANDIDATES — VALIDATION")
print("=" * 75)
print(validation_df.to_string(index=False))

# ============================================================
# STEP 8: SAVE CHALLENGER WITHOUT TOUCHING CHAMPION
# ============================================================

best = validation_df.iloc[0]

best_group = best["feature_group"]
best_model_name = best["model"]

best_pipeline, best_features = trained_candidates[
    (best_group, best_model_name)
]

challenger_path = (
    ROOT / "models" / "late_delivery_risk_v3_challenger.joblib"
)

joblib.dump({
    "pipeline": best_pipeline,
    "feature_columns": best_features,
    "model_name": best_model_name,
    "feature_group": best_group,
    "prediction_stage": "post_approval",
    "selection_metric": "validation_recall_at_20",
    "status": "experimental_challenger"
}, challenger_path)

# ============================================================
# STEP 9: REPORT TEST PERFORMANCE
# ============================================================

# This is retrospective diagnostics only:
# previous V1/V2 test results have already been inspected.

test_probabilities = best_pipeline.predict_proba(
    test[best_features]
)[:, 1]

test_metrics = evaluate(
    test["late_delivery"],
    test_probabilities
)

test_report = pd.DataFrame([{
    "model": best_model_name,
    "feature_group": best_group,
    "test_rows": len(test),
    "test_prevalence": test["late_delivery"].mean(),
    **test_metrics
}])

test_report.to_csv(
    REPORT / "challenger_test_metrics.csv",
    index=False
)

print("\n" + "=" * 75)
print("V3 CHALLENGER — RETROSPECTIVE TEST")
print("=" * 75)
print(test_report.to_string(index=False))

# ============================================================
# STEP 10: CHAMPION VS CHALLENGER
# ============================================================

champion = pd.read_csv(
    ROOT / "reports/model_v1_v2_common_cohort_audit.csv"
)

champion = champion.loc[
    champion["model"] == "V1 Random Forest"
].iloc[0]

print("\n" + "=" * 75)
print("CHAMPION VS CHALLENGER")
print("=" * 75)

print(
    "V1 Recall@20:",
    round(champion["recall_at_20"] * 100, 2),
    "%"
)

print(
    "V3 Recall@20:",
    round(test_metrics["recall_at_20"] * 100, 2),
    "%"
)

print(
    "V1 PR-AUC:",
    round(champion["pr_auc"], 4)
)

print(
    "V3 PR-AUC:",
    round(test_metrics["pr_auc"], 4)
)

print("\nIMPORTANT:")
print("Champion model remains unchanged.")
print("V3 is an experimental challenger.")
print("Do not promote based on repeatedly inspected test data.")

print("\nReports saved:", REPORT)
print("Challenger saved:", challenger_path)
print("\nV3 EXPERIMENT COMPLETE")
