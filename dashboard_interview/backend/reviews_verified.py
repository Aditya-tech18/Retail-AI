from pathlib import Path
from functools import lru_cache
from fastapi import APIRouter, Query
import pandas as pd
import numpy as np

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets" / "raw"
OUT = Path(__file__).resolve().parents[1] / "reports" / "reviews_verified"
OUT.mkdir(parents=True, exist_ok=True)

def load_source():
    reviews = pd.read_csv(
        RAW / "olist_order_reviews_dataset.csv",
        low_memory=False
    )
    orders = pd.read_csv(
        RAW / "olist_orders_dataset.csv",
        usecols=[
            "order_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )

    # One record per review_id. Olist can contain
    # multiple review records for some orders.
    reviews = reviews.drop_duplicates(
        subset=["review_id"], keep="last"
    ).copy()

    for c in [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]:
        orders[c] = pd.to_datetime(
            orders[c], errors="coerce"
        )

    orders = orders.drop_duplicates("order_id")
    df = reviews.merge(
        orders, on="order_id",
        how="left", validate="many_to_one"
    )

    df["review_score"] = pd.to_numeric(
        df.review_score, errors="coerce"
    )
    df = df[df.review_score.between(1, 5)].copy()

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
        .dt.to_period("M").astype(str)
    )

    df["has_delivery_dates"] = (
        df.order_delivered_customer_date.notna()
        & df.order_estimated_delivery_date.notna()
    )

    df["late"] = (
        df.order_delivered_customer_date
        > df.order_estimated_delivery_date
    )

    df["review_comment_message"] = (
        df.review_comment_message
        .fillna("").astype(str)
    )

    return df

@lru_cache(maxsize=1)
def analyze():
    df = load_source()
    n = len(df)

    ratings = [
        {
            "score": score,
            "count": int(
                (df.review_score == score).sum()
            )
        }
        for score in range(1, 6)
    ]

    sentiment = [
        {
            "label": label,
            "count": int((df.sentiment == label).sum()),
            "pct": round(
                100 * (df.sentiment == label).mean(), 2
            )
        }
        for label in ["Positive", "Neutral", "Negative"]
    ]

    # Only months with sufficient observations
    # appear in the primary trend chart.
    valid = df[
        df.month.str.match(r"^20\d{2}-\d{2}$")
    ]

    monthly = []
    for month, group in valid.groupby("month"):
        count = len(group)
        monthly.append({
            "month": month,
            "count": count,
            "positive": round(
                100 * (group.sentiment == "Positive").mean(), 2
            ),
            "negative": round(
                100 * (group.sentiment == "Negative").mean(), 2
            )
        })

    # Monthly trend: require >=100 reviews.
    # Threshold is displayed transparently.
    trend = [
        row for row in monthly
        if row["count"] >= 100
    ]

    delivered = df[df.has_delivery_dates].copy()
    delivery_ratings = []
    low_rates = {}

    for is_late, label in [
        (False, "On-time"),
        (True, "Late")
    ]:
        group = delivered[
            delivered.late == is_late
        ]

        counts = group.review_score.value_counts()
        total = len(group)

        low = int((group.review_score <= 2).sum())
        low_rates[label] = (
            low / total if total else None
        )

        delivery_ratings.append({
            "name": label,
            "count": total,
            "low_rating_count": low,
            "values": [
                round(
                    100 * counts.get(score, 0) / total,
                    2
                ) if total else 0
                for score in range(1, 6)
            ]
        })

    a = low_rates["On-time"]
    b = low_rates["Late"]

    ratio = (
        round(b / a, 3)
        if a is not None and a > 0
        and b is not None else None
    )

    # Save reproducible analysis outputs.
    pd.DataFrame(ratings).to_csv(
        OUT / "rating_distribution.csv", index=False
    )
    pd.DataFrame(sentiment).to_csv(
        OUT / "sentiment_distribution.csv", index=False
    )
    pd.DataFrame(monthly).to_csv(
        OUT / "monthly_review_trend_all.csv", index=False
    )
    pd.DataFrame(trend).to_csv(
        OUT / "monthly_review_trend_chart.csv", index=False
    )
    pd.DataFrame(delivery_ratings).to_csv(
        OUT / "delivery_rating_comparison.csv",
        index=False
    )

    return {
        "total_reviews": n,
        "average_rating": round(
            float(df.review_score.mean()), 4
        ),
        "sentiment": sentiment,
        "rating_distribution": ratings,
        "trend": trend,
        "delivery_ratings": delivery_ratings,
        "late_low_rating_ratio": ratio,
        "comments_available": int(
            df.review_comment_message.str.strip().ne("").sum()
        ),
        "methodology": {
            "unit": "unique review_id",
            "sentiment": "4-5 positive, 3 neutral, 1-2 negative",
            "trend_min_monthly_reviews": 100,
            "delivery_comparison": (
                "Reviews with both actual and estimated "
                "delivery timestamps"
            ),
            "causal_claim": False
        }
    }

@router.get("/api/reviews-v2")
def reviews_v2(
    search: str = Query("", max_length=100),
    limit: int = Query(8, ge=1, le=30)
):
    result = dict(analyze())
    df = load_source()

    comments = df[
        df.review_comment_message.str.strip().ne("")
    ]

    if search:
        comments = comments[
            comments.review_comment_message.str.contains(
                search, case=False, regex=False
            )
        ]

    # Stable ordering, not cherry-picked examples.
    comments = comments.sort_values(
        ["review_id"]
    ).head(limit)

    result["review_explorer"] = [
        {
            "rating": int(row.review_score),
            "text": row.review_comment_message,
            "sentiment": row.sentiment,
            "month": row.month
        }
        for row in comments.itertuples(index=False)
    ]

    return result

if __name__ == "__main__":
    from pprint import pprint
    result = analyze()

    print("\nVERIFIED CUSTOMER REVIEW ANALYSIS")
    print("=" * 48)
    print("Total reviews:", result["total_reviews"])
    print("Average rating:", result["average_rating"])
    print("Sentiment:")
    pprint(result["sentiment"])
    print("Star distribution:")
    pprint(result["rating_distribution"])
    print("Monthly trend points:", len(result["trend"]))
    print("Delivery comparison:")
    pprint(result["delivery_ratings"])
    print(
        "Low-rating risk ratio:",
        result["late_low_rating_ratio"]
    )
    print("Output:", OUT)
