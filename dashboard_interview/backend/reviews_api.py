from fastapi import APIRouter, Query
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

router = APIRouter()
ROOT = Path(__file__).resolve().parents[2]

@lru_cache(maxsize=1)
def review_data():
    raw = ROOT / "datasets" / "raw"
    reviews = pd.read_csv(
        raw / "olist_order_reviews_dataset.csv",
        low_memory=False
    )
    orders = pd.read_csv(
        raw / "olist_orders_dataset.csv",
        usecols=[
            "order_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )

    for col in [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]:
        orders[col] = pd.to_datetime(
            orders[col], errors="coerce"
        )

    df = reviews.merge(orders, on="order_id", how="left")
    df["review_score"] = pd.to_numeric(
        df["review_score"], errors="coerce"
    )
    df = df.dropna(subset=["review_score"]).copy()
    df["sentiment"] = np.select(
        [
            df.review_score >= 4,
            df.review_score <= 2
        ],
        ["Positive", "Negative"],
        default="Neutral"
    )
    df["month"] = (
        df.order_purchase_timestamp
        .dt.to_period("M")
        .astype(str)
    )
    df["late"] = (
        df.order_delivered_customer_date
        > df.order_estimated_delivery_date
    )
    df["valid_delivery"] = (
        df.order_delivered_customer_date.notna()
        & df.order_estimated_delivery_date.notna()
    )
    df["review_comment_message"] = (
        df["review_comment_message"]
        .fillna("")
        .astype(str)
    )
    return df

@router.get("/api/reviews")
def reviews_api(
    search: str = Query("", max_length=100),
    limit: int = Query(25, ge=1, le=100)
):
    df = review_data()
    n = len(df)

    sentiments = []
    for label in ["Positive", "Neutral", "Negative"]:
        count = int((df.sentiment == label).sum())
        sentiments.append({
            "label": label,
            "count": count,
            "pct": round(count / n * 100, 2) if n else 0
        })

    rating_counts = (
        df.review_score.value_counts()
        .reindex(range(1, 6), fill_value=0)
    )

    valid_months = df[
        df.month.str.match(r"^20\d\d-\d\d$")
    ]

    trend_table = (
        valid_months.groupby(["month", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    trend = []
    for month, row in trend_table.iterrows():
        total = int(row.sum())
        if total:
            trend.append({
                "month": month,
                "positive": round(
                    row.get("Positive", 0) / total * 100, 2
                ),
                "negative": round(
                    row.get("Negative", 0) / total * 100, 2
                )
            })

    delivery = []
    low_rates = {}

    for late, label in [
        (False, "On-time"),
        (True, "Late")
    ]:
        group = df[
            df.valid_delivery & (df.late == late)
        ]
        dist = (
            group.review_score.value_counts(
                normalize=True
            )
        )
        low_rates[label] = (
            float((group.review_score <= 2).mean())
            if len(group) else None
        )
        delivery.append({
            "name": label,
            "count": int(len(group)),
            "values": [
                round(float(dist.get(i, 0)) * 100, 2)
                for i in range(1, 6)
            ]
        })

    a = low_rates["On-time"]
    b = low_rates["Late"]
    ratio = b / a if a and b is not None else None

    comments = df[
        df.review_comment_message.str.strip().ne("")
    ]

    if search:
        comments = comments[
            comments.review_comment_message.str.contains(
                search, case=False, regex=False
            )
        ]

    comments = comments.head(limit)

    explorer = [
        {
            "rating": int(row.review_score),
            "text": row.review_comment_message,
            "sentiment": row.sentiment,
            "month": row.month
        }
        for row in comments.itertuples(index=False)
    ]

    return {
        "total_reviews": n,
        "average_rating": round(
            float(df.review_score.mean()), 3
        ),
        "sentiment": sentiments,
        "rating_distribution": [
            {"score": i, "count": int(rating_counts[i])}
            for i in range(1, 6)
        ],
        "trend": trend,
        "delivery_ratings": delivery,
        "late_low_rating_ratio": (
            round(ratio, 3) if ratio is not None else None
        ),
        "review_explorer": explorer,
        "method": (
            "Rating-derived sentiment: "
            "4-5 positive, 3 neutral, 1-2 negative"
        )
    }
