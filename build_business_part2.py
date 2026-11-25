from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
import shutil

ROOT = Path.cwd()
PATH = ROOT / "notebooks/05_business_analytics.ipynb"

if not PATH.exists():
    raise FileNotFoundError(PATH)

nb = nbf.read(PATH, as_version=4)

if len(nb.cells) != 7:
    raise ValueError(
        f"Expected 7 existing cells, found {len(nb.cells)}. "
        "Notebook unchanged."
    )

# Keep the backup outside notebooks to avoid duplicate Notebook 05 files.
BACKUP = ROOT / "05_business_analytics_before_part2_backup.ipynb"
if not BACKUP.exists():
    shutil.copy2(PATH, BACKUP)

def add(title, code):
    nb.cells.append(
        nbf.v4.new_code_cell(
            f"# {'=' * 72}\n"
            f"# {title}\n"
            f"# {'=' * 72}\n\n"
            + code.strip()
        )
    )


# ============================================================
# CELL 8 — COHORT RETENTION
# ============================================================

add("08 | Customer Cohort Retention and Observation Windows", r'''
"""
Business question:
Do customers return after their first purchase?

A cohort is defined by the customer's first observed
eligible purchase month.

We avoid counting future, unobservable months as churn.
"""

cohort_orders = eligible.dropna(
    subset=["customer_unique_id", "purchase_timestamp"]
).copy()

cohort_orders["order_month"] = (
    cohort_orders["purchase_timestamp"].dt.to_period("M")
)

first_purchase = (
    cohort_orders.groupby("customer_unique_id")["order_month"]
    .min()
    .rename("cohort_month")
)

cohort_orders = cohort_orders.join(
    first_purchase, on="customer_unique_id"
)

cohort_orders["months_since_first"] = (
    (cohort_orders["order_month"].dt.year -
     cohort_orders["cohort_month"].dt.year) * 12
    + cohort_orders["order_month"].dt.month
    - cohort_orders["cohort_month"].dt.month
)

assert cohort_orders["months_since_first"].ge(0).all()

cohort_sizes = (
    cohort_orders.groupby("cohort_month")["customer_unique_id"]
    .nunique()
)

cohort_active = (
    cohort_orders.groupby(
        ["cohort_month", "months_since_first"]
    )["customer_unique_id"]
    .nunique()
    .unstack(fill_value=0)
)

cohort_retention = cohort_active.div(
    cohort_sizes, axis=0
)

# Months after the last fully observed calendar month
# must be missing, not zero retention.
observation_end = (
    cohort_orders["order_month"].max() - 1
)

for cohort in cohort_retention.index:
    for age in cohort_retention.columns:
        if cohort + int(age) > observation_end:
            cohort_retention.loc[cohort, age] = np.nan

# Limit visualization to the first 12 months.
retention_12 = cohort_retention.reindex(
    columns=range(12)
)

display(
    (retention_12.head(12) * 100).round(2)
)

plt.figure(figsize=(15, 8))
sns.heatmap(
    retention_12.tail(18) * 100,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    mask=retention_12.tail(18).isna(),
    cbar_kws={"label": "Retention (%)"}
)

plt.title("Customer Cohort Retention — First 12 Months")
plt.xlabel("Months Since First Purchase")
plt.ylabel("Acquisition Cohort")
plt.tight_layout()
plt.show()

print("Cohort sizes:", len(cohort_sizes))
print("Last fully observed month:", observation_end)
print(
    "Interpretation: same-month repeat activity is "
    "included in month zero; future months are censored."
)
''')


# ============================================================
# CELL 9 — PRODUCT CATEGORY ECONOMICS
# ============================================================

add("09 | Category Economics and Revenue Concentration", r'''
"""
Category performance is based on merchandise value.

We do not join payment totals to individual item rows,
because that would duplicate order-level payments.

Category sales are not profit: product cost, refunds,
fees and fulfillment costs are unavailable.
"""

category_items = items.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

translation_path = ROOT / "datasets/raw/product_category_name_translation.csv"

if translation_path.exists():
    translation = pd.read_csv(translation_path)

    assert translation[
        "product_category_name"
    ].is_unique, "Translation table contains duplicate keys."

    category_items = category_items.merge(
        translation,
        on="product_category_name",
        how="left",
        validate="many_to_one"
    )

    category_items["category"] = (
        category_items["product_category_name_english"]
        .fillna(category_items["product_category_name"])
    )
else:
    category_items["category"] = (
        category_items["product_category_name"]
    )

category_items["category"] = (
    category_items["category"].fillna("unknown")
)

eligible_order_ids = set(eligible["order_id"])

category_items = category_items.loc[
    category_items["order_id"].isin(eligible_order_ids)
].copy()

category_kpis = (
    category_items.groupby("category", as_index=False)
    .agg(
        merchandise_brl=("price", "sum"),
        freight_brl=("freight_value", "sum"),
        units=("order_item_id", "size"),
        orders=("order_id", "nunique"),
        products=("product_id", "nunique"),
        sellers=("seller_id", "nunique")
    )
)

category_kpis["merchandise_share_pct"] = (
    100 * category_kpis["merchandise_brl"]
    / category_kpis["merchandise_brl"].sum()
)

category_kpis["avg_item_price_brl"] = (
    category_kpis["merchandise_brl"]
    / category_kpis["units"]
)

category_kpis = category_kpis.sort_values(
    "merchandise_brl", ascending=False
)

assert np.isclose(
    category_kpis["merchandise_brl"].sum(),
    category_items["price"].sum()
)

display(category_kpis.head(15))

plt.figure(figsize=(12, 7))
sns.barplot(
    data=category_kpis.head(12),
    x="merchandise_brl",
    y="category"
)
plt.title("Top Categories by Merchandise Value")
plt.xlabel("Merchandise Value (BRL)")
plt.ylabel("")
plt.tight_layout()
plt.show()

top_10_category_share = (
    category_kpis.head(10)["merchandise_brl"].sum()
    / category_kpis["merchandise_brl"].sum()
    * 100
)

print(
    f"Top 10 category concentration: "
    f"{top_10_category_share:.2f}%"
)
''')


# ============================================================
# CELL 10 — GEOGRAPHIC PERFORMANCE
# ============================================================

add("10 | Geographic Business Performance", r'''
"""
Business question:
Where are order volume and customer value concentrated?

Customer geography is taken from the customer table.
This represents customer location, not seller location.
"""

state_kpis = (
    eligible.groupby("customer_state", dropna=False)
    .agg(
        orders=("order_id", "nunique"),
        customers=("customer_unique_id", "nunique"),
        recorded_payment_brl=(
            "recorded_payment_brl", "sum"
        ),
        paid_orders=("recorded_payment_brl", "count")
    )
    .reset_index()
)

state_kpis["aov_brl"] = (
    state_kpis["recorded_payment_brl"]
    / state_kpis["paid_orders"].replace(0, np.nan)
)

state_kpis["revenue_share_pct"] = (
    100 * state_kpis["recorded_payment_brl"]
    / state_kpis["recorded_payment_brl"].sum()
)

state_kpis = state_kpis.sort_values(
    "recorded_payment_brl", ascending=False
)

assert state_kpis["orders"].sum() == len(eligible)

display(state_kpis.head(15))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(
    data=state_kpis.head(12),
    x="recorded_payment_brl",
    y="customer_state",
    ax=axes[0]
)
axes[0].set_title("Top States by Payment Volume")
axes[0].set_xlabel("BRL")
axes[0].set_ylabel("")

# Minimum sample size to avoid ranking tiny markets by AOV.
large_states = state_kpis.loc[
    state_kpis["paid_orders"] >= 100
].sort_values("aov_brl", ascending=False)

sns.barplot(
    data=large_states.head(12),
    x="aov_brl",
    y="customer_state",
    ax=axes[1]
)
axes[1].set_title("AOV by State (100+ Paid Orders)")
axes[1].set_xlabel("BRL")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()
''')


# ============================================================
# CELL 11 — DELIVERY PERFORMANCE
# ============================================================

add("11 | Delivery Performance and Satisfaction", r'''
"""
Business question:
Are late deliveries associated with lower customer ratings?

Delivery performance is measured only for orders with
actual and estimated delivery timestamps.

Review ratings are aggregated to order level to avoid
duplicating orders with multiple review records.
"""

delivery = eligible.loc[
    eligible["order_status"].eq("delivered")
].copy()

delivery["delivery_delay_days"] = (
    (
        delivery["delivered_timestamp"]
        - delivery["estimated_delivery_timestamp"]
    ).dt.total_seconds() / 86400
)

delivery["actual_delivery_days"] = (
    (
        delivery["delivered_timestamp"]
        - delivery["purchase_timestamp"]
    ).dt.total_seconds() / 86400
)

delivery = delivery.dropna(
    subset=[
        "delivery_delay_days",
        "actual_delivery_days"
    ]
)

delivery = delivery.loc[
    delivery["actual_delivery_days"] >= 0
].copy()

delivery["late_delivery"] = (
    delivery["delivery_delay_days"] > 0
)

# All available ratings, including reviews without text.
rating_order = (
    reviews_raw.dropna(subset=["review_score"])
    .groupby("order_id", as_index=False)
    .agg(
        average_review_score=("review_score", "mean"),
        review_records=("review_score", "size")
    )
)

delivery = delivery.merge(
    rating_order,
    on="order_id",
    how="left",
    validate="one_to_one"
)

delivery["low_rating"] = (
    delivery["average_review_score"] <= 2
)

rated_delivery = delivery.dropna(
    subset=["average_review_score"]
).copy()

delivery_kpis = (
    rated_delivery.groupby("late_delivery")
    .agg(
        orders=("order_id", "nunique"),
        avg_rating=("average_review_score", "mean"),
        low_rating_rate=("low_rating", "mean"),
        avg_delivery_days=("actual_delivery_days", "mean"),
        median_delay_days=("delivery_delay_days", "median")
    )
    .reset_index()
)

display(delivery_kpis)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.barplot(
    data=delivery_kpis,
    x="late_delivery",
    y="avg_rating",
    ax=axes[0]
)
axes[0].set_title("Average Rating: On-Time vs Late")
axes[0].set_xlabel("Late Delivery")
axes[0].set_ylabel("Average Review Score")

sns.barplot(
    data=delivery_kpis,
    x="late_delivery",
    y="low_rating_rate",
    ax=axes[1]
)
axes[1].set_title("Low-Rating Rate by Delivery Status")
axes[1].set_xlabel("Late Delivery")
axes[1].set_ylabel("Rate")

plt.tight_layout()
plt.show()

print("Delivered orders with usable ratings:", len(rated_delivery))
print("Association does not establish causality.")
''')


# ============================================================
# CELL 12 — NLP INTELLIGENCE AND MODEL QUALITY
# ============================================================

add("12 | NLP Review Intelligence and Model Diagnostics", r'''
"""
NLP results were generated in Notebook 04.

Quality safeguards:
- Compare predicted sentiment with review ratings.
- Measure text coverage.
- Do not treat keyword matches as verified complaints.
- Do not infer seller-wide performance from text reviews alone.
"""

nlp = pd.read_parquet(
    DATA / "review_product_intelligence.parquet"
)

assert {
    "review_id", "order_id", "sentiment",
    "review_score", "sentiment_confidence"
}.issubset(nlp.columns)

nlp = nlp.drop_duplicates(
    ["review_id", "order_id"]
).copy()

nlp["rating_polarity"] = np.select(
    [
        nlp["review_score"].le(2),
        nlp["review_score"].ge(4)
    ],
    ["negative", "positive"],
    default="neutral"
)

nlp["sentiment_disagreement"] = (
    nlp["sentiment"] != nlp["rating_polarity"]
)

nlp_summary = (
    nlp.groupby("sentiment")
    .agg(
        reviews=("review_id", "size"),
        avg_rating=("review_score", "mean"),
        avg_confidence=("sentiment_confidence", "mean"),
        disagreement_rate=(
            "sentiment_disagreement", "mean"
        )
    )
    .reset_index()
)

text_coverage = (
    nlp["order_id"].nunique()
    / reviews_raw["order_id"].nunique()
)

display(nlp_summary)

print(f"Order-level text coverage: {text_coverage:.2%}")

complaint_columns = [
    col for col in nlp.columns
    if col.startswith("complaint_")
]

complaint_counts = (
    nlp[complaint_columns]
    .fillna(False)
    .astype(bool)
    .sum()
    .sort_values(ascending=False)
)

complaint_summary = complaint_counts.rename_axis(
    "complaint_type"
).reset_index(name="matched_reviews")

display(complaint_summary)

plt.figure(figsize=(11, 5))
sns.barplot(
    data=complaint_summary,
    x="matched_reviews",
    y="complaint_type"
)
plt.title("Heuristic Complaint Topic Matches")
plt.xlabel("Matched Text Reviews")
plt.ylabel("")
plt.tight_layout()
plt.show()

print(
    "Warning: Rating-sentiment disagreement is a "
    "diagnostic proxy, not validated model accuracy."
)
print(
    "Keyword matches may include negations or praise "
    "and require manual precision checks."
)
''')


# ============================================================
# CELL 13 — STATISTICAL HYPOTHESIS TESTING
# ============================================================

add("13 | Hypothesis Test: Late Delivery and Low Ratings", r'''
"""
H0: Late delivery and low ratings are independent.
H1: They are associated.

Methods:
- 2x2 contingency table
- Chi-square test
- Risk difference
- Risk ratio
- Approximate 95% confidence interval for risk difference

Observational association, not a causal treatment effect.
"""

from scipy.stats import chi2_contingency

test_data = rated_delivery[
    ["late_delivery", "low_rating"]
].dropna().copy()

contingency = pd.crosstab(
    test_data["late_delivery"],
    test_data["low_rating"]
).reindex(
    index=[False, True],
    columns=[False, True],
    fill_value=0
)

display(contingency)

if (contingency.values == 0).all() or (
    contingency.sum(axis=1) == 0
).any() or (
    contingency.sum(axis=0) == 0
).any():
    raise ValueError(
        "Insufficient variation for the planned hypothesis test."
    )

chi2, p_value, degrees_freedom, expected = (
    chi2_contingency(contingency, correction=False)
)

on_time_n = contingency.loc[False].sum()
late_n = contingency.loc[True].sum()

on_time_risk = (
    contingency.loc[False, True] / on_time_n
)

late_risk = (
    contingency.loc[True, True] / late_n
)

risk_difference = late_risk - on_time_risk

risk_ratio = (
    late_risk / on_time_risk
    if on_time_risk > 0 else np.nan
)

standard_error = np.sqrt(
    late_risk * (1 - late_risk) / late_n
    + on_time_risk * (1 - on_time_risk) / on_time_n
)

ci_low = risk_difference - 1.96 * standard_error
ci_high = risk_difference + 1.96 * standard_error

statistical_results = pd.DataFrame([{
    "late_orders": int(late_n),
    "on_time_orders": int(on_time_n),
    "late_low_rating_rate": late_risk,
    "on_time_low_rating_rate": on_time_risk,
    "risk_difference": risk_difference,
    "risk_ratio": risk_ratio,
    "ci95_low": ci_low,
    "ci95_high": ci_high,
    "chi_square": chi2,
    "degrees_freedom": degrees_freedom,
    "p_value": p_value
}])

display(statistical_results)

print(
    f"Low-rating risk difference: "
    f"{risk_difference:.2%}"
)
print(
    f"Approx. 95% CI: [{ci_low:.2%}, {ci_high:.2%}]"
)
print(f"Chi-square p-value: {p_value:.6g}")

if p_value < 0.05:
    print(
        "Statistically significant association "
        "at the 5% threshold."
    )
else:
    print(
        "Insufficient evidence of association "
        "at the 5% threshold."
    )

print(
    "Caveat: confounding by seller, product, "
    "geography and delivery distance is possible."
)
''')


# ============================================================
# CELL 14 — EXECUTIVE RECOMMENDATIONS
# ============================================================

add("14 | Evidence-Based Business Recommendations", r'''
"""
Prioritize recommendations using observed evidence.

Do not claim hypothetical intervention benefits
as achieved revenue or causal impact.
"""

recommendations = []

# Delivery recommendation
if late_n > 0:
    recommendations.append({
        "priority": 1,
        "business_area": "Delivery experience",
        "evidence": (
            f"{int(late_n):,} rated late deliveries; "
            f"low-rating risk difference "
            f"{risk_difference:.1%}"
        ),
        "recommended_action": (
            "Investigate late-delivery drivers by "
            "seller, geography and product category."
        ),
        "validation_plan": (
            "Run a controlled delivery-alert or "
            "fulfillment improvement experiment."
        ),
        "caveat": "Observed association is not causal."
    })

# Customer retention recommendation
repeat_rate = (
    100 * repeat_customers / purchasing_customers
    if purchasing_customers else np.nan
)

recommendations.append({
    "priority": 2,
    "business_area": "Customer retention",
    "evidence": (
        f"Observed repeat purchase rate: "
        f"{repeat_rate:.2f}%"
    ),
    "recommended_action": (
        "Target high-value customers with "
        "personalized second-purchase campaigns."
    ),
    "validation_plan": (
        "Randomized holdout test measuring "
        "incremental repeat orders and contribution."
    ),
    "caveat": (
        "Historical RFM segmentation is descriptive; "
        "campaign uplift has not been established."
    )
})

# Category concentration recommendation
recommendations.append({
    "priority": 3,
    "business_area": "Category portfolio",
    "evidence": (
        f"Top 10 categories contribute "
        f"{top_10_category_share:.2f}% "
        f"of merchandise value."
    ),
    "recommended_action": (
        "Review concentration risk and prioritize "
        "high-demand categories for deeper margin analysis."
    ),
    "validation_plan": (
        "Add costs, refunds and seller economics "
        "before reallocating investment."
    ),
    "caveat": (
        "Merchandise value does not measure profitability."
    )
})

executive_recommendations = pd.DataFrame(
    recommendations
).sort_values("priority")

display(executive_recommendations)

print("EXECUTIVE SUMMARY")
print("=" * 60)

for row in executive_recommendations.itertuples():
    print(
        f"\nPriority {row.priority}: "
        f"{row.business_area}"
    )
    print("Evidence:", row.evidence)
    print("Action:", row.recommended_action)
    print("Validation:", row.validation_plan)
    print("Limitation:", row.caveat)
''')


# ============================================================
# CELL 15 — REPRODUCIBLE OUTPUTS AND AUDIT
# ============================================================

add("15 | Export, Data Contracts and Final Audit", r'''
"""
All outputs are generated from validated upstream tables.

A dashboard can consume these exports without
repeating the notebook's analytical joins.
"""

exports = {
    "business_executive_kpis.csv": kpi_table,
    "business_revenue_reconciliation.csv": reconciliation,
    "business_monthly_growth.csv": monthly.reset_index().assign(
        purchase_month=lambda d: d["purchase_month"].astype(str)
    ),
    "business_growth_decomposition.csv":
        valid_growth.reset_index().assign(
            purchase_month=lambda d: d["purchase_month"].astype(str)
        ),
    "business_customer_segments.csv": segment_summary,
    "business_cohort_retention.csv":
        retention_12.reset_index().assign(
            cohort_month=lambda d: d["cohort_month"].astype(str)
        ),
    "business_category_kpis.csv": category_kpis,
    "business_geography_kpis.csv": state_kpis,
    "business_delivery_kpis.csv": delivery_kpis,
    "business_nlp_diagnostics.csv": nlp_summary,
    "business_complaint_topics.csv": complaint_summary,
    "business_statistical_test.csv": statistical_results,
    "business_executive_recommendations.csv":
        executive_recommendations
}

for filename, frame in exports.items():
    output_path = REPORTS / filename
    frame.to_csv(output_path, index=False)
    assert output_path.exists()
    print(f"Saved: {filename} | {len(frame):,} rows")

# Export order-grain fact table for future dashboard.
fact_path = DATA / "business_order_fact.parquet"

fact_to_export = eligible.copy()
fact_to_export["purchase_month"] = (
    fact_to_export["purchase_month"].astype(str)
)

fact_to_export.to_parquet(fact_path, index=False)

assert fact_to_export["order_id"].is_unique
assert len(fact_to_export) == len(eligible)

# Final financial and row-grain contracts
assert np.isclose(
    fact["recorded_payment_brl"].sum(),
    payments["payment_value"].sum()
)

assert segment_summary["customers"].sum() == (
    customer_spend["customer_unique_id"].nunique()
)

print("\n" + "=" * 65)
print("RETAILAI BUSINESS ANALYTICS — COMPLETE")
print("=" * 65)
print("Notebook cells: 15")
print("Business CSV reports:", len(exports))
print("Order fact table:", fact_path)
print("Currency: BRL")
print("Financial metrics: historical, not net profit")
print("Statistical results: observational, not causal")
print("Dashboard datasets: READY")
''')

assert len(nb.cells) == 15
nbf.write(nb, PATH)

print("15 cells saved to:", PATH)
print("Executing all 15 cells in the same notebook...")

# Execute and persist results in the SAME notebook.
client = NotebookClient(
    nb,
    timeout=900,
    kernel_name="retailai-nlp",
    resources={"metadata": {"path": str(ROOT)}}
)

try:
    client.execute()
except Exception:
    # Preserve executed outputs up to the failing cell.
    nbf.write(nb, PATH)
    print("Execution failed. Partial outputs saved.")
    raise
else:
    nbf.write(nb, PATH)
    print("\nSUCCESS: ALL 15 CELLS EXECUTED")
    print("Notebook:", PATH)
