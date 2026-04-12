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


# ---------- Sub-Category Profit Margins ----------
def subcategory_profit_margins(df: pl.DataFrame):
    result = (
        df.group_by("Sub-Category")
        .agg([
            pl.sum("Sales").alias("sales"),
            pl.sum("Profit").alias("profit"),
            pl.count().alias("order_count")
        ])
        .with_columns([
            (pl.col("profit") / pl.col("sales")).alias("margin")
        ])
        .sort("margin", descending=True)
    )
    
    rows = result.to_dicts()
    
    # Find top and bottom performers
    top_performers = rows[:5]
    bottom_performers = rows[-3:]
    
    top_str = ", ".join([f"{r['Sub-Category']} ({r['margin']:.2%})" for r in top_performers])
    bottom_str = ", ".join([f"{r['Sub-Category']} ({r['margin']:.2%})" for r in bottom_performers])
    
    texts = [
        f"""PROFIT MARGIN ANALYSIS BY SUB-CATEGORY:
Top performers (highest profit margins): {top_str}.
Bottom performers (lowest profit margins): {bottom_str}."""
    ]
    
    # Add individual sub-category insights
    for r in rows:
        performance = "highly profitable" if r["margin"] > 0.15 else "profitable" if r["margin"] > 0.05 else "low margin" if r["margin"] > 0 else "loss-making"
        texts.append(
            f"{r['Sub-Category']} has a profit margin of {r['margin']:.2%} on {r['order_count']} orders "
            f"({r['sales']:.0f} sales, {r['profit']:.0f} profit). This sub-category is {performance}."
        )
    
    return texts


# ---------- Discount Frequency Analysis ----------
def discount_frequency_analysis(df: pl.DataFrame):
    # Get total sales per product
    product_totals = (
        df.group_by("Product Name")
        .agg(pl.count().alias("total_sales"))
    )
    
    # Get discounted sales per product
    discounted_sales = (
        df.filter(pl.col("Discount") > 0)
        .group_by("Product Name")
        .agg(pl.count().alias("discount_count"))
    )
    
    # Combine and calculate percentage
    result = (
        product_totals.join(discounted_sales, on="Product Name", how="left")
        .with_columns([
            pl.col("discount_count").fill_null(0),
            (pl.col("discount_count").fill_null(0) / pl.col("total_sales")).alias("discount_percentage")
        ])
        .sort("discount_count", descending=True)
    )
    
    rows = result.to_dicts()
    
    # Find products most frequently sold at discount
    frequent_discount = [r for r in rows if r["discount_percentage"] > 0.5][:5]
    frequent_str = ", ".join([f"{r['Product Name']} ({r['discount_percentage']:.1%})" for r in frequent_discount])
    
    texts = [
        f"""DISCOUNT FREQUENCY ANALYSIS:
Products most frequently sold at discount (>50% of sales): {frequent_str}."""
    ]
    
    # Add detailed insights for products with significant discount activity
    for r in rows[:10]:  # Top 10 most discounted products
        if r["discount_count"] > 0:
            texts.append(
                f"{r['Product Name']} was sold at a discount {r['discount_count']} times "
                f"out of {r['total_sales']} total sales ({r['discount_percentage']:.1%} discount rate). "
                f"This product has {'high' if r['discount_percentage'] > 0.5 else 'moderate' if r['discount_percentage'] > 0.2 else 'low'} discount frequency."
            )
    
    return texts


# ---------- Regional Sales Performance ----------
def regional_performance_analysis(df: pl.DataFrame):
    result = (
        df.group_by("Region")
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit"),
            pl.count().alias("order_count"),
            pl.mean("Discount").alias("avg_discount")
        ])
        .with_columns([
            (pl.col("total_profit") / pl.col("total_sales")).alias("profit_margin")
        ])
        .sort("total_sales", descending=True)
    )
    
    rows = result.to_dicts()
    
    # Find best performing region
    best_region = rows[0]
    
    texts = [
        f"""REGIONAL SALES PERFORMANCE ANALYSIS:
Best performing region: {best_region['Region']} with {best_region['total_sales']:.0f} in total sales 
and profit margin of {best_region['profit_margin']:.2%}."""
    ]
    
    # Add insights for each region
    for r in rows:
        texts.append(
            f"{r['Region']} generated {r['total_sales']:.0f} in sales across {r['order_count']} orders, "
            f"with total profit of {r['total_profit']:.0f} (margin: {r['profit_margin']:.2%}). "
            f"Average discount in this region: {r['avg_discount']:.2f}."
        )
    
    return texts


# ---------- State Sales Performance Comparison ----------
def state_performance_comparison(df: pl.DataFrame):
    result = (
        df.group_by("State")
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit"),
            pl.count().alias("order_count"),
            pl.mean("Discount").alias("avg_discount")
        ])
        .with_columns([
            (pl.col("total_profit") / pl.col("total_sales")).alias("profit_margin")
        ])
        .sort("total_sales", descending=True)
    )
    
    rows = result.to_dicts()
    
    # Find top and bottom performing states
    top_states = rows[:5]
    bottom_states = rows[-3:]
    
    top_str = ", ".join([f"{s['State']} ({s['total_sales']:.0f})" for s in top_states])
    bottom_str = ", ".join([f"{s['State']} ({s['total_sales']:.0f})" for s in bottom_states])
    
    avg_sales = sum(r["total_sales"] for r in rows) / len(rows)
    
    texts = [
        f"""SALES PERFORMANCE COMPARISON ACROSS STATES:
Top performing states (highest sales): {top_str}.
Bottom performing states (lowest sales): {bottom_str}.
Average state sales: {avg_sales:.0f}."""
    ]
    
    # Add detailed insights for each state
    for r in rows:
        performance = "top performer" if r["total_sales"] > avg_sales * 1.5 else "above average" if r["total_sales"] > avg_sales else "below average"
        texts.append(
            f"{r['State']} generated {r['total_sales']:.0f} in sales across {r['order_count']} orders, "
            f"with total profit of {r['total_profit']:.0f} (margin: {r['profit_margin']:.2%}). "
            f"This state is {performance} with average discount of {r['avg_discount']:.2f}."
        )
    
    return texts


# ---------- City Performance Analysis ----------
def city_performance_analysis(df: pl.DataFrame):
    result = (
        df.group_by("City")
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit"),
            pl.count().alias("order_count"),
            pl.mean("Discount").alias("avg_discount")
        ])
        .with_columns([
            (pl.col("total_profit") / pl.col("total_sales")).alias("profit_margin")
        ])
        .sort("total_sales", descending=True)
    )
    
    rows = result.to_dicts()
    
    # Filter for cities with meaningful profit: positive margin AND substantial absolute profit
    # Calculate 50th percentile of profits for meaningful threshold
    all_profits = sorted([r["total_profit"] for r in rows], reverse=True)
    min_profit_threshold = all_profits[len(all_profits) // 2] if all_profits else 0
    
    profitable_cities = [r for r in rows if r["profit_margin"] > 0 and r["total_profit"] >= min_profit_threshold]
    profitable_cities_sorted = sorted(profitable_cities, key=lambda x: x["profit_margin"], reverse=True)
    
    # Get top performers by profit margin
    top_cities = profitable_cities_sorted[:10]
    
    top_str = ", ".join([f"{c['City']} (margin: {c['profit_margin']:.2%})" for c in top_cities])
    
    avg_margin = sum(r["profit_margin"] for r in profitable_cities) / len(profitable_cities) if profitable_cities else 0
    
    texts = [
        f"""TOP PERFORMING CITIES ANALYSIS (By Profitability):
Only cities with positive profit margins and substantial profit generation (top 50% of profits) are considered. 
Top performing cities (ranked by profit margin): {top_str}.
Average profit margin across qualifying cities: {avg_margin:.2%}."""
    ]
    
    # Add detailed insights for top performing cities
    for r in top_cities:
        performance = "exceptional" if r["profit_margin"] > 0.20 else "outstanding" if r["profit_margin"] > 0.15 else "very strong" if r["profit_margin"] > 0.10 else "strong"
        texts.append(
            f"CITY: {r['City']} is a {performance} performer with {r['total_sales']:.0f} in sales, "
            f"generating {r['total_profit']:.0f} profit with margin of {r['profit_margin']:.2%} across {r['order_count']} orders. "
            f"Average discount: {r['avg_discount']:.2f}."
        )
    
    return texts