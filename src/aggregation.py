import polars as pl
# ---------- Helper ----------
def safe_div(a, b):
    return a / b if b != 0 else 0

# ---------- Monthly Trends ----------
def monthly_sales_summary(df: pl.DataFrame):
    df = df.with_columns(
        pl.col("Order Date").str.strptime(pl.Date, strict=False)
    )

    result = (
        df.with_columns([
            pl.col("Order Date").dt.year().alias("year"),
            pl.col("Order Date").dt.month().alias("month")
        ])
        .drop_nulls(["year", "month"])
        .group_by(["year", "month"])
        .agg([
            pl.sum("Sales").alias("sales"),
            pl.sum("Profit").alias("profit"),
            pl.mean("Discount").alias("discount")
        ])
        .sort(["year", "month"])
    )

    rows = result.to_dicts()

    texts = []
    for i, r in enumerate(rows):
        margin = safe_div(r["profit"], r["sales"])

        trend = ""
        if i > 0:
            prev = rows[i - 1]
            if r["sales"] > prev["sales"]:
                trend = "increased compared to previous month"
            else:
                trend = "decreased compared to previous month"

        texts.append(
            f"In {r['year']}-{r['month']:02d}, sales were {r['sales']:.2f} and profit was {r['profit']:.2f}. "
            f"Profit margin was {margin:.2%} and average discount was {r['discount']:.2f}. "
            f"Sales {trend}."
        )

    return texts


# ---------- Category + Subcategory ----------
def category_summary(df: pl.DataFrame):
    result = (
        df.group_by(["Category", "Sub-Category"])
        .agg([
            pl.sum("Sales").alias("sales"),
            pl.sum("Profit").alias("profit"),
            pl.mean("Discount").alias("discount"),
            pl.count().alias("orders")
        ])
        .sort("sales", descending=True)
    )

    return [
        f"{r['Category']} -> {r['Sub-Category']} generated {r['sales']:.2f} sales across {r['orders']} orders. "
        f"Profit was {r['profit']:.2f} with margin {safe_div(r['profit'], r['sales']):.2%}. "
        f"Average discount {r['discount']:.2f}."
        for r in result.to_dicts()
    ]


# ---------- Regional + State ----------
def regional_summary(df: pl.DataFrame):
    result = (
        df.group_by(["Region", "State"])
        .agg([
            pl.sum("Sales").alias("sales"),
            pl.sum("Profit").alias("profit")
        ])
        .sort("sales", descending=True)
    )

    top = result.head(5).to_dicts()

    texts = [
        f"State {r['State']} in {r['Region']} generated {r['sales']:.2f} sales and {r['profit']:.2f} profit."
        for r in top
    ]

    return texts


# ---------- Product Deep Analysis ----------
def product_performance(df: pl.DataFrame, n: int = 10):
    result = (
        df.group_by("Product Name")
        .agg([
            pl.sum("Sales").alias("sales"),
            pl.sum("Profit").alias("profit"),
            pl.mean("Discount").alias("discount"),
            pl.count().alias("orders")
        ])
        .with_columns([
            (pl.col("profit") / pl.col("sales")).alias("margin")
        ])
        .sort("sales", descending=True)
    )

    rows = result.to_dicts()

    texts = []

    for r in rows[:n]:
        performance = "high performing"
        if r["margin"] < 0:
            performance = "loss-making"
        elif r["margin"] < 0.05:
            performance = "low margin"

        texts.append(
            f"Product {r['Product Name']} is {performance}. "
            f"It generated {r['sales']:.2f} sales across {r['orders']} orders. "
            f"Profit was {r['profit']:.2f} with margin {r['margin']:.2%}. "
            f"Average discount {r['discount']:.2f}."
        )

    return texts

def business_insights(df):
    high_discount = df.filter(pl.col("Discount") > 0.3)
    low_discount = df.filter(pl.col("Discount") <= 0.3)

    high_margin = (high_discount["Profit"].sum() / high_discount["Sales"].sum())
    low_margin = (low_discount["Profit"].sum() / low_discount["Sales"].sum())

    return [f"""
    Business insight:
    High discounts (>30%) result in profit margin {high_margin:.2%},
    while lower discounts result in margin {low_margin:.2%}.
    This suggests that heavy discounting {'reduces' if high_margin < low_margin else 'does not reduce'} profitability.
    """]

def global_summary(df):
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    avg_discount = df["Discount"].mean()

    best_category = (
        df.group_by("Category")
        .agg(pl.sum("Sales").alias("sales"))
        .sort("sales", descending=True)
        .row(0)[0]
    )

    return [f"""
    Overall dataset summary:
    Total sales are {total_sales:.2f} and total profit is {total_profit:.2f}.
    Average discount is {avg_discount:.2f}.
    The best performing category by sales is {best_category}.
    """]

# ---------- Discount Impact ----------
def discount_analysis(df: pl.DataFrame):
    result = (
        df.with_columns([
            (pl.col("Profit") / pl.col("Sales")).alias("margin")
        ])
        .group_by("Discount")
        .agg([
            pl.mean("margin").alias("avg_margin"),
            pl.count().alias("orders")
        ])
        .sort("Discount")
    )

    return [
        f"At discount level {r['Discount']:.2f}, average profit margin is {r['avg_margin']:.2%} across {r['orders']} orders."
        for r in result.to_dicts()
    ]


# ---------- Seasonality Analysis ----------
def seasonality_analysis(df: pl.DataFrame):
    df = df.with_columns(
        pl.col("Order Date").str.strptime(pl.Date, strict=False)
    )
    
    # Aggregate by month (across all years)
    monthly = (
        df.with_columns(pl.col("Order Date").dt.month().alias("month"))
        .drop_nulls("month")
        .group_by("month")
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit"),
            pl.count().alias("order_count")
        ])
        .sort("month")
    )
    
    rows = monthly.to_dicts()
    
    # Find peak and low months
    sorted_by_sales = sorted(rows, key=lambda x: x["total_sales"], reverse=True)
    peak_months = sorted_by_sales[:3]
    low_months = sorted_by_sales[-3:]
    
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    peak_str = ", ".join([f"{month_names[m['month']]} ({m['total_sales']:.0f})" for m in peak_months])
    low_str = ", ".join([f"{month_names[m['month']]} ({m['total_sales']:.0f})" for m in low_months])
    
    avg_sales = sum(r["total_sales"] for r in rows) / len(rows)
    seasonality_ratio = max(r["total_sales"] for r in rows) / min(r["total_sales"] for r in rows)
    
    return [
        f"""SEASONALITY ANALYSIS: Strong seasonality detected (peak/low ratio: {seasonality_ratio:.1f}x). 
Peak sales months: {peak_str}. 
Low sales months: {low_str}. 
Average monthly sales: {avg_sales:.0f}. 
This indicates clear seasonal patterns in customer purchasing behavior."""
    ]