from pathlib import Path
from functools import lru_cache
import json
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(os.environ.get("RETAILAI_PROJECT", Path(__file__).resolve().parents[2]))
BASE = Path(__file__).resolve().parents[1]
FRONTEND = BASE / "frontend"

app = FastAPI(title="RetailAI 3.0", version="3.0")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


def locate(name):
    matches = [
        p for p in ROOT.rglob(name)
        if "dashboard_interview" not in p.parts
        and ".venv" not in p.parts
        and "node_modules" not in p.parts
    ]
    if not matches:
        raise FileNotFoundError(f"Dataset missing: {name}")
    return min(matches, key=lambda p: len(p.parts))


@lru_cache(maxsize=12)
def table(name):
    return pd.read_csv(locate(name), low_memory=False)


def numeric(value):
    if pd.isna(value):
        return None
    return float(value)


def safe_json(data):
    return json.loads(json.dumps(data, default=str))


@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "project": str(ROOT),
        "mode": "historical_analysis",
    }


@app.get("/api/overview")
def overview():
    orders = table("olist_orders_dataset.csv").copy()
    payments = table("olist_order_payments_dataset.csv")
    items = table("olist_order_items_dataset.csv")
    products = table("olist_products_dataset.csv")
    translations = table("product_category_name_translation.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )

    # Payments can contain multiple rows per order.
    revenue = payments.groupby("order_id")["payment_value"].sum()
    total_revenue = float(revenue.sum())
    total_orders = int(orders["order_id"].nunique())

    monthly = (
        orders.assign(
            month=orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
        )
        .merge(revenue.rename("revenue"), on="order_id", how="left")
        .groupby("month", as_index=False)["revenue"].sum()
        .sort_values("month")
    )
    monthly = monthly[
        monthly["month"].str.match(r"^20\d{2}-\d{2}$", na=False)
    ]

    categories = (
        items[["order_id", "product_id", "price"]]
        .merge(
            products[["product_id", "product_category_name"]],
            on="product_id",
            how="left",
        )
        .merge(
            translations,
            on="product_category_name",
            how="left",
        )
    )
    categories["category"] = categories[
        "product_category_name_english"
    ].fillna("Other")

    top_categories = (
        categories.groupby("category")["price"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
    )

    delivered = orders[orders["order_status"] == "delivered"].copy()
    actual = pd.to_datetime(
        delivered["order_delivered_customer_date"], errors="coerce"
    )
    estimated = pd.to_datetime(
        delivered["order_estimated_delivery_date"], errors="coerce"
    )
    valid = actual.notna() & estimated.notna()
    late_rate = float((actual[valid] > estimated[valid]).mean()) if valid.any() else None

    return safe_json({
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "average_order_value": total_revenue / total_orders,
        "historical_late_rate": late_rate,
        "monthly": monthly.tail(30).to_dict("records"),
        "categories": [
            {"name": k, "revenue": float(v)}
            for k, v in top_categories.items()
        ],
        "currency": "BRL",
        "source": "Olist historical dataset",
    })


@app.get("/api/customers")
def customers():
    orders = table("olist_orders_dataset.csv").copy()
    customers_df = table("olist_customers_dataset.csv")
    payments = table("olist_order_payments_dataset.csv")
    reviews = table("olist_order_reviews_dataset.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )

    df = orders[
        ["order_id", "customer_id", "order_purchase_timestamp"]
    ].merge(
        customers_df[
            ["customer_id", "customer_unique_id"]
        ],
        on="customer_id",
        how="left",
    )

    pay = payments.groupby("order_id")["payment_value"].sum()
    df = df.merge(pay.rename("spend"), on="order_id", how="left")
    df["spend"] = df["spend"].fillna(0)

    snapshot = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("spend", "sum"),
    ).dropna()

    rfm["recency"] = (
        snapshot - rfm["last_purchase"]
    ).dt.days

    # Explicit, interpretable rule-based segments.
    # These are not claimed to be trained K-means clusters.
    high_value = rfm["monetary"].quantile(0.75)
    recent = rfm["recency"].quantile(0.50)

    conditions = [
        (rfm["frequency"] > 1) & (rfm["monetary"] >= high_value),
        (rfm["frequency"] > 1),
        (rfm["frequency"] == 1) & (rfm["recency"] > recent),
    ]
    labels = [
        "Best Customers",
        "Regular Shoppers",
        "At Risk",
    ]

    rfm["segment"] = np.select(
        conditions, labels, default="New Customers"
    )

    segment_data = (
        rfm.groupby("segment")
        .agg(
            customers=("frequency", "size"),
            average_spend=("monetary", "mean"),
            revenue=("monetary", "sum"),
        )
        .reset_index()
    )

    review_scores = (
        reviews["review_score"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return safe_json({
        "unique_customers": int(len(rfm)),
        "repeat_rate": float((rfm["frequency"] > 1).mean()),
        "average_review": float(reviews["review_score"].mean()),
        "segments": segment_data.to_dict("records"),
        "review_scores": [
            {"score": str(k), "count": int(v)}
            for k, v in review_scores.items()
        ],
        "segment_method": "Rule-based RFM; historical data",
    })


@app.get("/api/predictions")
def predictions():
    audit = ROOT / "reports" / "champion_advanced_audit"
    capacity_path = audit / "capacity_threshold_comparison.csv"
    calibration_path = audit / "calibration_diagnostics.csv"
    ci_path = audit / "bootstrap_confidence_intervals.csv"

    if not all(p.exists() for p in [
        capacity_path, calibration_path, ci_path
    ]):
        raise HTTPException(
            status_code=404,
            detail="Champion audit reports not found. Run the existing audit first.",
        )

    capacity = pd.read_csv(capacity_path)
    calibration = pd.read_csv(calibration_path)
    intervals = pd.read_csv(ci_path)

    active = capacity.iloc[
        (capacity["capacity"] - 0.20).abs().argmin()
    ]

    model = calibration[
        calibration["model"].str.contains(
            "Random Forest", case=False, na=False
        )
    ].iloc[0]

    return safe_json({
        "evaluated_orders": int(
            active["tp"] + active["fp"] + active["fn"] + active["tn"]
        ),
        "late_orders": int(active["tp"] + active["fn"]),
        "champion": "V1 Random Forest",
        "calibration": "Platt scaling evaluated separately",
        "recall_at_20": float(active["recall"]),
        "precision_at_20": float(active["precision"]),
        "lift_at_20": float(active["lift"]),
        "pr_auc": float(model["pr_auc"]),
        "roc_auc": float(model["roc_auc"]),
        "confusion": {
            "tp": int(active["tp"]),
            "fp": int(active["fp"]),
            "fn": int(active["fn"]),
            "tn": int(active["tn"]),
        },
        "capacity": capacity.to_dict("records"),
        "confidence_intervals": intervals.to_dict("records"),
        "calibration_comparison": calibration.to_dict("records"),
        "mode": "retrospective_test_evaluation",
    })


@app.get("/api/decisions")
def decisions(
    capacity: float = 0.20,
    cost: float = 8.0,
    success: float = 0.35,
    value: float = 45.0,
):
    if not (0.01 <= capacity <= 0.50):
        raise HTTPException(400, "Capacity must be between 1% and 50%")
    if cost < 0 or value < 0 or not 0 <= success <= 1:
        raise HTTPException(400, "Invalid simulation assumptions")

    audit = ROOT / "reports" / "champion_advanced_audit"
    path = audit / "capacity_threshold_comparison.csv"
    if not path.exists():
        raise HTTPException(404, "Capacity audit report missing")

    df = pd.read_csv(path)
    row = df.iloc[(df["capacity"] - capacity).abs().argmin()]

    flagged = int(row["flagged"])
    captured = int(row["tp"])
    prevented = captured * success
    intervention_cost = flagged * cost
    gross_benefit = prevented * value
    net_benefit = gross_benefit - intervention_cost

    return safe_json({
        "capacity": float(row["capacity"]),
        "flagged": flagged,
        "captured": captured,
        "expected_prevented": prevented,
        "intervention_cost": intervention_cost,
        "gross_benefit": gross_benefit,
        "net_benefit": net_benefit,
        "roi": (
            net_benefit / intervention_cost
            if intervention_cost else None
        ),
        "break_even_success": (
            intervention_cost / (captured * value)
            if captured * value else None
        ),
        "note": "Scenario estimate, not measured savings or causal impact.",
    })

# ============================================================



# ============================================================
# RetailAI 3.0 — Premium Data and Customer Reviews
# ============================================================

@app.get("/api/reviews")
def reviews_api(search: str = "", limit: int = 30):
    from fastapi import HTTPException

    if len(search) > 100:
        raise HTTPException(400, "Search too long")

    limit = max(1, min(int(limit), 100))

    reviews = table("olist_order_reviews_dataset.csv").copy()
    orders = table("olist_orders_dataset.csv")[
        [
            "order_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ].copy()

    for column in [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        orders[column] = pd.to_datetime(
            orders[column], errors="coerce"
        )

    reviews = reviews.merge(orders, on="order_id", how="left")
    reviews["review_score"] = pd.to_numeric(
        reviews["review_score"], errors="coerce"
    )
    reviews = reviews.dropna(subset=["review_score"])

    reviews["sentiment"] = np.select(
        [
            reviews["review_score"] >= 4,
            reviews["review_score"] <= 2,
        ],
        ["Positive", "Negative"],
        default="Neutral",
    )

    counts = reviews["sentiment"].value_counts()
    total = len(reviews)

    sentiments = [
        {
            "label": label,
            "count": int(counts.get(label, 0)),
            "pct": round(100 * counts.get(label, 0) / total, 2)
            if total else 0,
        }
        for label in ["Positive", "Neutral", "Negative"]
    ]

    reviews["month"] = (
        reviews["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        reviews.groupby(["month", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    trend = []
    for month, row in monthly.iterrows():
        if not str(month).startswith("20"):
            continue
        n = row.sum()
        if n:
            trend.append({
                "month": month,
                "positive": round(
                    100 * row.get("Positive", 0) / n, 2
                ),
                "negative": round(
                    100 * row.get("Negative", 0) / n, 2
                ),
            })

    valid = (
        reviews["order_delivered_customer_date"].notna()
        & reviews["order_estimated_delivery_date"].notna()
    )

    reviews["late"] = (
        reviews["order_delivered_customer_date"]
        > reviews["order_estimated_delivery_date"]
    )

    delivery = []
    low_rates = {}

    for late, label in [(False, "On-time"), (True, "Late")]:
        subset = reviews[valid & (reviews["late"] == late)]
        dist = subset["review_score"].value_counts(
            normalize=True
        )

        low_rates[label] = (
            float((subset["review_score"] <= 2).mean())
            if len(subset) else None
        )

        delivery.append({
            "name": label,
            "count": len(subset),
            "values": [
                round(100 * float(dist.get(i, 0)), 2)
                for i in range(1, 6)
            ],
        })

    ontime_rate = low_rates["On-time"]
    late_rate = low_rates["Late"]

    ratio = (
        late_rate / ontime_rate
        if ontime_rate and late_rate is not None
        else None
    )

    comments = reviews["review_comment_message"].fillna(
        ""
    ).astype(str)

    mask = comments.str.strip().ne("")

    if search:
        mask &= comments.str.contains(
            search, case=False, regex=False
        )

    explorer = reviews.loc[mask, [
        "review_score",
        "review_comment_message",
        "sentiment",
        "month",
    ]].head(limit)

    return safe_json({
        "total_reviews": total,
        "average_rating": float(
            reviews["review_score"].mean()
        ),
        "sentiment": sentiments,
        "trend": trend,
        "delivery_ratings": delivery,
        "late_low_rating_ratio": ratio,
        "review_explorer": [
            {
                "rating": int(r.review_score),
                "text": r.review_comment_message,
                "sentiment": r.sentiment,
                "month": r.month,
            }
            for r in explorer.itertuples(index=False)
        ],
        "method": (
            "Sentiment is derived from star ratings: "
            "4-5 positive, 3 neutral, 1-2 negative."
        ),
        "source": "Olist historical customer reviews",
    })


@app.get("/api/premium-data")
def premium_data_api():
    import inspect
    from fastapi.encoders import jsonable_encoder

    sections = {
        "overview": overview,
        "customers": customers,
        "predictions": predictions,
        "decisions": decisions,
    }

    result = {}
    errors = {}

    for name, function in sections.items():
        try:
            signature = inspect.signature(function)
            kwargs = {
                key: parameter.default
                for key, parameter in signature.parameters.items()
                if parameter.default is not inspect.Parameter.empty
            }

            result[name] = jsonable_encoder(
                function(**kwargs)
            )
        except Exception as exc:
            result[name] = None
            errors[name] = str(exc)

    result["metadata"] = {
        "source": "Local Olist dataset and audit reports",
        "errors": errors,
        "verified": not bool(errors),
    }

    return result


from backend.reviews_api import router as reviews_router
app.include_router(reviews_router)


from backend.reviews_verified import router as verified_reviews_router
app.include_router(verified_reviews_router)
