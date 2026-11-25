from pathlib import Path
import nbformat as nbf

ROOT = Path.cwd()
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)

cells = []

def add_cell(title, code):
    cell = nbf.v4.new_code_cell(
        f"# {'=' * 70}\n"
        f"# {title}\n"
        f"# {'=' * 70}\n\n"
        + code.strip()
    )
    cells.append(cell)


# ============================================================
# CELL 1 — BUSINESS CONTEXT AND ANALYTICAL FRAMEWORK
# ============================================================

add_cell("01 | Business Context and KPI Framework", r'''
"""
RETAILAI — E-COMMERCE BUSINESS INTELLIGENCE

Business questions
------------------
1. What drives recorded marketplace payment volume?
2. Are growth changes caused by order volume or order value?
3. Which customer segments contribute the most revenue?
4. How does customer retention vary across cohorts?
5. Where do delivery problems and customer dissatisfaction overlap?
6. Which interventions should the business test first?

Metric definitions
------------------
Recorded payment volume:
    Sum of payment_value for eligible orders.

Eligible orders:
    Orders excluding canceled and unavailable statuses.

Average Order Value (AOV):
    Recorded payment volume / eligible orders with payments.

Repeat Purchase Rate:
    Customers with 2+ eligible orders / purchasing customers.

Important:
    Recorded payments are not net revenue, profit, or GMV.
    The historical Olist dataset is not live business data.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
if not (ROOT / "datasets").exists():
    ROOT = ROOT.parent

DATA = ROOT / "datasets" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110
})

pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:,.2f}".format)

print("RetailAI | Business Intelligence")
print("Project root:", ROOT)
print("Currency: BRL")
print("Analytical framework initialized.")
''')


# ============================================================
# CELL 2 — DATA LOADING AND QUALITY CONTRACTS
# ============================================================

add_cell("02 | Data Loading and Quality Contracts", r'''
required_files = {
    "orders": "orders.parquet",
    "items": "order_items.parquet",
    "payments": "order_payments.parquet",
    "customers": "customers.parquet",
    "products": "products.parquet",
    "reviews": "order_reviews.parquet",
}

datasets = {}

for name, filename in required_files.items():
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset: {path}")

    datasets[name] = pd.read_parquet(path)

orders = datasets["orders"].copy()
items = datasets["items"].copy()
payments = datasets["payments"].copy()
customers = datasets["customers"].copy()
products = datasets["products"].copy()
reviews_raw = datasets["reviews"].copy()

segments = pd.read_csv(DATA / "customer_segments.csv")
retention = pd.read_csv(
    DATA / "customer_retention_strategies.csv"
)

def validate_primary_key(frame, key, name):
    assert key in frame.columns, f"{name}: missing {key}"
    assert frame[key].notna().all(), f"{name}: null {key}"
    assert frame[key].is_unique, f"{name}: duplicate {key}"

validate_primary_key(orders, "order_id", "orders")
validate_primary_key(customers, "customer_id", "customers")
validate_primary_key(products, "product_id", "products")
validate_primary_key(
    segments, "customer_unique_id", "segments"
)
validate_primary_key(
    retention, "customer_unique_id", "retention"
)

for frame, name in [
    (items, "items"),
    (payments, "payments")
]:
    assert frame["order_id"].notna().all(), (
        f"{name}: null order_id"
    )

for col in ["price", "freight_value"]:
    items[col] = pd.to_numeric(items[col], errors="coerce")

payments["payment_value"] = pd.to_numeric(
    payments["payment_value"], errors="coerce"
)

assert (items["price"].dropna() >= 0).all()
assert (items["freight_value"].dropna() >= 0).all()
assert (payments["payment_value"].dropna() >= 0).all()

quality = []

for name, frame in datasets.items():
    quality.append({
        "dataset": name,
        "rows": len(frame),
        "columns": frame.shape[1],
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_full_rows": int(frame.duplicated().sum())
    })

quality_report = pd.DataFrame(quality)

display(quality_report)

print("Primary-key and numeric integrity checks passed.")
print("Customer segments:", len(segments))
print("Retention records:", len(retention))
''')


# ============================================================
# CELL 3 — REVENUE FACT TABLE AND RECONCILIATION
# ============================================================

add_cell("03 | Order-Level Fact Table and Revenue Reconciliation", r'''
"""
Critical modeling decision:
Payments and order items are both one-to-many against orders.

Joining them directly creates a many-to-many multiplication.

Solution:
1. Aggregate payments to order_id.
2. Aggregate items to order_id.
3. Join both aggregates to the order dimension.
4. Validate row counts and monetary reconciliation.
"""

payment_agg = (
    payments.groupby("order_id", as_index=False)
    .agg(
        recorded_payment_brl=("payment_value", "sum"),
        payment_records=("payment_value", "size"),
        payment_methods=("payment_type", "nunique")
    )
)

item_agg = (
    items.groupby("order_id", as_index=False)
    .agg(
        merchandise_value_brl=("price", "sum"),
        freight_value_brl=("freight_value", "sum"),
        item_count=("order_item_id", "size"),
        unique_products=("product_id", "nunique"),
        unique_sellers=("seller_id", "nunique")
    )
)

fact = (
    orders
    .merge(
        payment_agg,
        on="order_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        item_agg,
        on="order_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
                "customer_state",
                "customer_city"
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )
)

assert len(fact) == len(orders)
assert fact["order_id"].is_unique

source_payment_total = payments["payment_value"].sum()
fact_payment_total = fact["recorded_payment_brl"].sum()

assert np.isclose(
    source_payment_total,
    fact_payment_total,
    rtol=1e-10
), "Payment reconciliation failed."

fact["purchase_timestamp"] = pd.to_datetime(
    fact["order_purchase_timestamp"], errors="coerce"
)

fact["delivered_timestamp"] = pd.to_datetime(
    fact["order_delivered_customer_date"], errors="coerce"
)

fact["estimated_delivery_timestamp"] = pd.to_datetime(
    fact["order_estimated_delivery_date"], errors="coerce"
)

fact["purchase_month"] = (
    fact["purchase_timestamp"].dt.to_period("M")
)

fact["eligible"] = ~fact["order_status"].isin(
    ["canceled", "unavailable"]
)

eligible = fact.loc[fact["eligible"]].copy()

reconciliation = pd.DataFrame([{
    "source_payment_total_brl": source_payment_total,
    "fact_payment_total_brl": fact_payment_total,
    "difference_brl": (
        source_payment_total - fact_payment_total
    ),
    "source_orders": len(orders),
    "fact_orders": len(fact),
    "eligible_orders": len(eligible),
    "eligible_without_payment": int(
        eligible["recorded_payment_brl"].isna().sum()
    )
}])

display(reconciliation)

print("Order-level fact table successfully validated.")
''')


# ============================================================
# CELL 4 — EXECUTIVE KPI SCORECARD
# ============================================================

add_cell("04 | Executive KPI Scorecard", r'''
"""
These KPIs form the executive overview of the dashboard.

All revenue-related metrics use the same eligible-order
population to maintain consistent denominators.
"""

customer_frequency = (
    eligible.dropna(subset=["customer_unique_id"])
    .groupby("customer_unique_id")["order_id"]
    .nunique()
)

purchasing_customers = len(customer_frequency)

repeat_customers = int(
    customer_frequency.ge(2).sum()
)

recorded_volume = eligible[
    "recorded_payment_brl"
].sum()

paid_order_count = int(
    eligible["recorded_payment_brl"].notna().sum()
)

kpis = {
    "Recorded Payment Volume (BRL)": recorded_volume,
    "Eligible Orders": len(eligible),
    "Purchasing Customers": purchasing_customers,
    "Average Payment / Paid Order (BRL)":
        recorded_volume / paid_order_count
        if paid_order_count else np.nan,
    "Repeat Purchase Rate (%)":
        100 * repeat_customers / purchasing_customers
        if purchasing_customers else np.nan,
    "Items / Order":
        eligible["item_count"].mean()
}

kpi_table = pd.DataFrame(
    [{"metric": k, "value": v} for k, v in kpis.items()]
)

display(kpi_table)

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
axes = axes.flatten()

for ax, (metric, value) in zip(axes, kpis.items()):
    ax.axis("off")

    if "Rate" in metric:
        formatted = f"{value:,.2f}%"
    elif "Volume" in metric or "(BRL)" in metric:
        formatted = f"R$ {value:,.2f}"
    elif "Items" in metric:
        formatted = f"{value:,.2f}"
    else:
        formatted = f"{value:,.0f}"

    ax.text(
        0.5, 0.60, formatted,
        ha="center", va="center",
        fontsize=19, fontweight="bold",
        transform=ax.transAxes
    )

    ax.text(
        0.5, 0.30, metric,
        ha="center", va="center",
        fontsize=10,
        transform=ax.transAxes
    )

plt.suptitle(
    "RetailAI | Executive Business Scorecard",
    fontsize=17, fontweight="bold"
)
plt.tight_layout()
plt.show()

print("Note: Recorded payment volume is not net profit.")
''')


# ============================================================
# CELL 5 — MONTHLY GROWTH AND TEMPORAL VALIDITY
# ============================================================

add_cell("05 | Monthly Revenue Growth and Trend Diagnostics", r'''
"""
Monthly business performance.

Exclude partial boundary months from growth interpretation.
Keep them visible in the raw monthly table.
"""

monthly = (
    eligible.dropna(subset=["purchase_timestamp"])
    .groupby("purchase_month")
    .agg(
        orders=("order_id", "nunique"),
        recorded_volume_brl=(
            "recorded_payment_brl", "sum"
        ),
        paying_orders=(
            "recorded_payment_brl", "count"
        ),
        customers=("customer_unique_id", "nunique")
    )
    .sort_index()
)

monthly["average_payment_brl"] = (
    monthly["recorded_volume_brl"]
    / monthly["paying_orders"].replace(0, np.nan)
)

monthly["revenue_mom_pct"] = (
    monthly["recorded_volume_brl"].pct_change() * 100
)

monthly["order_mom_pct"] = (
    monthly["orders"].pct_change() * 100
)

# A month is considered fully observable only when
# the source date range covers its first and last day.
min_date = eligible["purchase_timestamp"].min()
max_date = eligible["purchase_timestamp"].max()

monthly["complete_calendar_month"] = [
    (
        period.start_time >= min_date.normalize()
        and period.end_time.normalize()
        <= max_date.normalize()
    )
    for period in monthly.index
]

monthly["reliable_mom_comparison"] = (
    monthly["complete_calendar_month"]
    & monthly["complete_calendar_month"].shift(
        1, fill_value=False
    )
)

monthly["reportable_mom_pct"] = (
    monthly["revenue_mom_pct"].where(
        monthly["reliable_mom_comparison"]
    )
)

display(monthly.tail(12))

plot_monthly = monthly.loc[
    monthly["complete_calendar_month"]
]

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

axes[0].plot(
    plot_monthly.index.astype(str),
    plot_monthly["recorded_volume_brl"],
    marker="o", linewidth=2
)
axes[0].set_title("Monthly Recorded Payment Volume")
axes[0].set_ylabel("BRL")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(
    monthly.index.astype(str),
    monthly["reportable_mom_pct"]
)
axes[1].axhline(0, linewidth=1)
axes[1].set_title(
    "Month-over-Month Growth — Complete Months Only"
)
axes[1].set_ylabel("Growth (%)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

print(
    "Months with reliable MoM comparisons:",
    int(monthly["reliable_mom_comparison"].sum())
)
''')


# ============================================================
# CELL 6 — GROWTH DECOMPOSITION
# ============================================================

add_cell("06 | Growth Decomposition: Order Volume vs Order Value", r'''
"""
Business identity:

Recorded payment volume
    = Paying order count × Average payment per paying order

Exact period-over-period decomposition:

Change in volume
    = Change in orders × Previous AOV
    + Change in AOV × Previous orders
    + Interaction term

The interaction term is retained to avoid attribution errors.
"""

growth = monthly[
    ["paying_orders", "average_payment_brl",
     "recorded_volume_brl", "complete_calendar_month"]
].copy()

growth["previous_orders"] = growth[
    "paying_orders"
].shift(1)

growth["previous_aov"] = growth[
    "average_payment_brl"
].shift(1)

growth["order_volume_effect_brl"] = (
    (growth["paying_orders"] - growth["previous_orders"])
    * growth["previous_aov"]
)

growth["order_value_effect_brl"] = (
    growth["previous_orders"]
    * (
        growth["average_payment_brl"]
        - growth["previous_aov"]
    )
)

growth["interaction_effect_brl"] = (
    (growth["paying_orders"] - growth["previous_orders"])
    * (
        growth["average_payment_brl"]
        - growth["previous_aov"]
    )
)

growth["actual_change_brl"] = (
    growth["recorded_volume_brl"].diff()
)

growth["explained_change_brl"] = (
    growth["order_volume_effect_brl"]
    + growth["order_value_effect_brl"]
    + growth["interaction_effect_brl"]
)

growth["valid_comparison"] = (
    growth["complete_calendar_month"]
    & growth["complete_calendar_month"].shift(
        1, fill_value=False
    )
    & growth["average_payment_brl"].notna()
    & growth["previous_aov"].notna()
)

valid_growth = growth.loc[
    growth["valid_comparison"]
].copy()

assert np.allclose(
    valid_growth["actual_change_brl"],
    valid_growth["explained_change_brl"],
    atol=0.01
), "Growth decomposition does not reconcile."

display(
    valid_growth[
        [
            "actual_change_brl",
            "order_volume_effect_brl",
            "order_value_effect_brl",
            "interaction_effect_brl"
        ]
    ].tail(10)
)

if not valid_growth.empty:
    last = valid_growth.iloc[-1]

    effects = pd.Series({
        "Order volume": last["order_volume_effect_brl"],
        "Average order value": last["order_value_effect_brl"],
        "Interaction": last["interaction_effect_brl"]
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    effects.plot.bar(ax=ax)

    ax.axhline(0, linewidth=1)
    ax.set_title(
        "Latest Complete-Month Growth Decomposition"
    )
    ax.set_ylabel("Contribution to Change (BRL)")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.show()

print("Exact growth decomposition validated.")
''')


# ============================================================
# CELL 7 — CUSTOMER SEGMENT ECONOMICS
# ============================================================

add_cell("07 | Customer Segment Economics and Revenue Concentration", r'''
"""
Customer segmentation should answer:

- Which segments generate the most recorded volume?
- Which segments have the highest average customer value?
- Is revenue concentrated among a small number of customers?
- Are existing RFM labels consistent with observed spending?

Important:
The segmentation dataset is derived from historical orders.
This is descriptive analysis, not out-of-sample prediction.
"""

customer_spend = (
    eligible.dropna(subset=["customer_unique_id"])
    .groupby("customer_unique_id", as_index=False)
    .agg(
        observed_orders=("order_id", "nunique"),
        observed_payment_brl=(
            "recorded_payment_brl", "sum"
        )
    )
)

customer_spend = customer_spend.merge(
    segments[
        [
            "customer_unique_id",
            "customer_segment",
            "RFM_score",
            "cluster"
        ]
    ],
    on="customer_unique_id",
    how="left",
    validate="one_to_one"
)

customer_spend["customer_segment"] = (
    customer_spend["customer_segment"]
    .fillna("Unsegmented")
)

segment_summary = (
    customer_spend.groupby("customer_segment")
    .agg(
        customers=("customer_unique_id", "nunique"),
        orders=("observed_orders", "sum"),
        recorded_volume_brl=(
            "observed_payment_brl", "sum"
        ),
        avg_customer_spend_brl=(
            "observed_payment_brl", "mean"
        ),
        median_customer_spend_brl=(
            "observed_payment_brl", "median"
        ),
        repeat_customers=(
            "observed_orders",
            lambda s: int((s >= 2).sum())
        )
    )
    .reset_index()
)

segment_summary["revenue_share_pct"] = (
    100 * segment_summary["recorded_volume_brl"]
    / segment_summary["recorded_volume_brl"].sum()
)

segment_summary["repeat_rate_pct"] = (
    100 * segment_summary["repeat_customers"]
    / segment_summary["customers"]
)

segment_summary = segment_summary.sort_values(
    "recorded_volume_brl", ascending=False
)

assert (
    segment_summary["customers"].sum()
    == len(customer_spend)
)

assert np.isclose(
    segment_summary["recorded_volume_brl"].sum(),
    customer_spend["observed_payment_brl"].sum()
)

display(segment_summary)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(
    data=segment_summary,
    x="recorded_volume_brl",
    y="customer_segment",
    ax=axes[0]
)

axes[0].set_title("Recorded Payment Volume by Segment")
axes[0].set_xlabel("BRL")
axes[0].set_ylabel("")

sns.barplot(
    data=segment_summary,
    x="repeat_rate_pct",
    y="customer_segment",
    ax=axes[1]
)

axes[1].set_title("Repeat Purchase Rate by Segment")
axes[1].set_xlabel("Repeat Customers (%)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

# Revenue concentration analysis
ranked_customers = customer_spend.sort_values(
    "observed_payment_brl", ascending=False
)

top_10_count = max(
    1, int(np.ceil(len(ranked_customers) * 0.10))
)

top_10_share = (
    100
    * ranked_customers.head(top_10_count)[
        "observed_payment_brl"
    ].sum()
    / ranked_customers["observed_payment_brl"].sum()
)

print(
    f"Top 10% customer revenue concentration: "
    f"{top_10_share:.2f}%"
)

print(
    "Unsegmented customers:",
    customer_spend["customer_segment"]
    .eq("Unsegmented").sum()
)

print(
    "Business interpretation: prioritize segments based "
    "on both value and repeat behavior, not revenue alone."
)
''')


# ============================================================
# SAVE NOTEBOOK
# ============================================================

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

destination = NOTEBOOKS / "05_business_analytics_interview.ipynb"

if destination.exists():
    raise FileExistsError(
        f"{destination} already exists. "
        "Existing notebook preserved; do not overwrite it."
    )

nbf.write(notebook, destination)

print("\nNotebook created:", destination)
print("Cells:", len(cells))
print("PART 1 READY")
