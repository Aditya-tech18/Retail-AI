from pathlib import Path
import nbformat as nbf

root = Path.cwd()
notebooks = root / "notebooks"
notebooks.mkdir(exist_ok=True)

cells = []

def add(code):
    cells.append(nbf.v4.new_code_cell(code.strip()))

# CELL 1 — Data loading and validation
add("""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()
if not (ROOT / "datasets").exists():
    ROOT = ROOT.parent

RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets" / "processed"
REPORTS = ROOT / "reports"
OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

reviews = pd.read_csv(RAW / "olist_order_reviews_dataset.csv")
orders = pd.read_csv(RAW / "olist_orders_dataset.csv")
items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
products = pd.read_csv(RAW / "olist_products_dataset.csv")
sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
translation = pd.read_csv(RAW / "product_category_name_translation.csv")

reviews["review_text_clean"] = (
    reviews["review_comment_title"].fillna("").astype(str).str.strip()
    + " "
    + reviews["review_comment_message"].fillna("").astype(str).str.strip()
).str.normalize("NFKC").str.replace(r"\\s+", " ", regex=True).str.strip()

review_nlp = reviews.loc[
    reviews["review_text_clean"].ne("")
].drop_duplicates(["review_id", "order_id"]).copy()

assert review_nlp["review_text_clean"].ne("").all()
print("Reviews for NLP:", len(review_nlp))
""")

# CELL 2 — Sentiment inference with checkpoint
add("""
from transformers import pipeline
from tqdm.auto import tqdm

checkpoint = OUT / "review_sentiment.parquet"
model_name = "clapAI/roberta-base-multilingual-sentiment"

required = {"review_id", "order_id", "sentiment", "sentiment_confidence"}

if checkpoint.exists():
    cached = pd.read_parquet(checkpoint)
    valid = (
        required.issubset(cached.columns)
        and len(cached) == len(review_nlp)
        and cached[["review_id", "order_id"]].reset_index(drop=True).equals(
            review_nlp[["review_id", "order_id"]].reset_index(drop=True)
        )
        and cached["sentiment"].notna().all()
        and cached["sentiment_confidence"].between(0, 1).all()
    )
else:
    valid = False

if valid:
    review_nlp = cached.copy()
    print("Reused validated sentiment checkpoint.")
else:
    classifier = pipeline(
        "text-classification",
        model=model_name,
        device=-1
    )

    texts = review_nlp["review_text_clean"].tolist()
    predictions = []

    for start in tqdm(range(0, len(texts), 32)):
        batch = texts[start:start + 32]
        result = classifier(
            batch,
            batch_size=32,
            truncation=True,
            max_length=256
        )
        predictions.extend(result)

    assert len(predictions) == len(review_nlp)

    review_nlp["sentiment"] = [
        p["label"].lower() for p in predictions
    ]
    review_nlp["sentiment_confidence"] = [
        p["score"] for p in predictions
    ]

    review_nlp.to_parquet(checkpoint, index=False)
    print("Saved sentiment checkpoint.")

print(review_nlp["sentiment"].value_counts())
""")

# CELL 3 — Sentiment quality and complaint detection
add("""
import re

# Weak-label taxonomy: interpretable, not a validated classifier.
complaint_patterns = {
    "delivery": r"atras|demor|entreg|prazo|chegou|receb|transport",
    "packaging": r"embalag|pacote|caixa|embrulh|proteção",
    "product_quality": r"quebrad|defeit|danific|estragad|qualidade|funciona|travando",
    "wrong_or_missing_item": r"errad|faltando|faltou|diferente|não veio|nao veio",
    "refund_return": r"reembols|devolu|estorn|troca|dinheiro de volta",
    "service": r"atendimento|suporte|vendedor|resposta|contato"
}

# Complaint detection is restricted to low-rated reviews to
# reduce false complaints from positive mentions.
low_rating = review_nlp["review_score"].le(2)
negative_text = review_nlp["sentiment"].eq("negative")
complaint_candidate = low_rating | negative_text

for topic, pattern in complaint_patterns.items():
    review_nlp["complaint_" + topic] = (
        complaint_candidate
        & review_nlp["review_text_clean"].str.contains(
            pattern, case=False, regex=True, na=False
        )
    )

topic_cols = ["complaint_" + t for t in complaint_patterns]
review_nlp["has_detected_complaint"] = review_nlp[topic_cols].any(axis=1)

review_nlp["low_rating"] = low_rating

print("Sentiment by rating:")
print(pd.crosstab(
    review_nlp["review_score"],
    review_nlp["sentiment"],
    normalize="index"
).round(3))

print("\\nComplaint topic counts:")
print(review_nlp[topic_cols].sum().sort_values(ascending=False))
""")

# CELL 4 — Safe business data integration
add("""
# Keep review-level data at its original grain.
review_business = review_nlp.merge(
    orders[["order_id", "customer_id", "order_status"]],
    on="order_id",
    how="left",
    validate="many_to_one"
)

review_business = review_business.merge(
    customers[
        ["customer_id", "customer_unique_id",
         "customer_city", "customer_state"]
    ],
    on="customer_id",
    how="left",
    validate="many_to_one"
)

# One order may contain several products and sellers.
# Build order-level product context without multiplying reviews.
item_context = items.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

item_context = item_context.merge(
    translation,
    on="product_category_name",
    how="left",
    validate="many_to_one"
)

order_context = item_context.groupby("order_id").agg(
    item_count=("order_item_id", "count"),
    unique_products=("product_id", "nunique"),
    unique_sellers=("seller_id", "nunique"),
    unique_categories=("product_category_name_english", "nunique")
).reset_index()

review_business = review_business.merge(
    order_context,
    on="order_id",
    how="left",
    validate="many_to_one"
)

assert len(review_business) == len(review_nlp)

# Attribute a review to a product/seller only when the
# order has exactly one unique product and one unique seller.
single_context = item_context.groupby("order_id").agg(
    product_id=("product_id", "first"),
    seller_id=("seller_id", "first"),
    category=("product_category_name_english", "first")
).reset_index()

review_business = review_business.merge(
    single_context,
    on="order_id",
    how="left",
    validate="many_to_one"
)

ambiguous = (
    review_business["unique_products"].ne(1)
    | review_business["unique_sellers"].ne(1)
)

review_business.loc[
    ambiguous, ["product_id", "seller_id", "category"]
] = None

print("Business dataset:", review_business.shape)
print("Reviews with unambiguous product/seller attribution:",
      review_business["product_id"].notna().sum())
""")

# CELL 5 — Business KPIs
add("""
# Minimum sample size avoids ranking tiny groups.
MIN_REVIEWS = 30

category_kpis = (
    review_business.dropna(subset=["category"])
    .groupby("category")
    .agg(
        review_count=("review_id", "size"),
        negative_rate=("low_rating", "mean"),
        average_rating=("review_score", "mean"),
        detected_complaint_rate=("has_detected_complaint", "mean")
    )
    .reset_index()
)

category_kpis = category_kpis.loc[
    category_kpis["review_count"] >= MIN_REVIEWS
].sort_values(
    ["negative_rate", "review_count"],
    ascending=[False, False]
)

seller_kpis = (
    review_business.dropna(subset=["seller_id"])
    .groupby("seller_id")
    .agg(
        review_count=("review_id", "size"),
        negative_rate=("low_rating", "mean"),
        average_rating=("review_score", "mean")
    )
    .reset_index()
)

seller_kpis = seller_kpis.loc[
    seller_kpis["review_count"] >= MIN_REVIEWS
].sort_values(
    ["negative_rate", "review_count"],
    ascending=[False, False]
)

print("Top categories by low-rating rate:")
display(category_kpis.head(10))

print("Top sellers by low-rating rate:")
display(seller_kpis.head(10))
""")

# CELL 6 — Exports and final checks
add("""
final_path = OUT / "review_product_intelligence.parquet"
review_business.to_parquet(final_path, index=False)

category_path = REPORTS / "review_category_kpis.csv"
seller_path = REPORTS / "review_seller_kpis.csv"
topic_path = REPORTS / "review_complaint_topics.csv"

category_kpis.to_csv(category_path, index=False)
seller_kpis.to_csv(seller_path, index=False)

topic_summary = pd.DataFrame({
    "topic": list(complaint_patterns),
    "matched_reviews": [
        int(review_business["complaint_" + topic].sum())
        for topic in complaint_patterns
    ]
})
topic_summary.to_csv(topic_path, index=False)

assert len(pd.read_parquet(final_path)) == len(review_nlp)

print("PIPELINE COMPLETE")
print("Reviews:", len(review_business))
print("Saved:", final_path)
print("Saved:", category_path)
print("Saved:", seller_path)
print("Saved:", topic_path)
print("Note: Complaint topics are heuristic signals, not verified ground truth.")
""")

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

destination = notebooks / "04_review_product_intelligence_final.ipynb"
nbf.write(notebook, destination)
print("Created:", destination)
print("Total cells:", len(cells))
