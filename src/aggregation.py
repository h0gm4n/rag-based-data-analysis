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
    
    top_performers = rows[:5]
    bottom_performers = rows[-3:]
    
    top_str = ", ".join([f"{r['Sub-Category']} ({r['margin']:.2%})" for r in top_performers])
    bottom_str = ", ".join([f"{r['Sub-Category']} ({r['margin']:.2%})" for r in bottom_performers])
    
    texts = [
        f"""PROFIT MARGIN ANALYSIS BY SUB-CATEGORY:
Top performers (highest profit margins): {top_str}.
Bottom performers (lowest profit margins): {bottom_str}."""
    ]
    
    for r in rows:
        performance = "highly profitable" if r["margin"] > 0.15 else "profitable" if r["margin"] > 0.05 else "low margin" if r["margin"] > 0 else "loss-making"
        texts.append(
            f"{r['Sub-Category']} has a profit margin of {r['margin']:.2%} on {r['order_count']} orders "
            f"({r['sales']:.0f} sales, {r['profit']:.0f} profit). This sub-category is {performance}."
        )
    
    return texts


# ---------- Discount Frequency Analysis ----------
def discount_frequency_analysis(df: pl.DataFrame):
    product_totals = (
        df.group_by("Product Name")
        .agg(pl.count().alias("total_sales"))
    )
    
    discounted_sales = (
        df.filter(pl.col("Discount") > 0)
        .group_by("Product Name")
        .agg(pl.count().alias("discount_count"))
    )
    
    result = (
        product_totals.join(discounted_sales, on="Product Name", how="left")
        .with_columns([
            pl.col("discount_count").fill_null(0),
            (pl.col("discount_count").fill_null(0) / pl.col("total_sales")).alias("discount_percentage")
        ])
        .sort("discount_count", descending=True)
    )
    
    rows = result.to_dicts()
    
    frequent_discount = [r for r in rows if r["discount_percentage"] > 0.5][:5]
    frequent_str = ", ".join([f"{r['Product Name']} ({r['discount_percentage']:.1%})" for r in frequent_discount])
    
    texts = [
        f"""DISCOUNT FREQUENCY ANALYSIS:
Products most frequently sold at discount (>50% of sales): {frequent_str}."""
    ]
    
    for r in rows[:10]:
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
    
    best_region = rows[0]
    
    texts = [
        f"""REGIONAL SALES PERFORMANCE ANALYSIS:
Best performing region: {best_region['Region']} with {best_region['total_sales']:.0f} in total sales 
and profit margin of {best_region['profit_margin']:.2%}."""
    ]
    
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
    

    all_profits = sorted([r["total_profit"] for r in rows], reverse=True)
    min_profit_threshold = all_profits[len(all_profits) // 2] if all_profits else 0
    
    profitable_cities = [r for r in rows if r["profit_margin"] > 0 and r["total_profit"] >= min_profit_threshold]
    profitable_cities_sorted = sorted(profitable_cities, key=lambda x: x["profit_margin"], reverse=True)

    top_cities = profitable_cities_sorted[:10]
    
    top_str = ", ".join([f"{c['City']} (margin: {c['profit_margin']:.2%})" for c in top_cities])
    
    avg_margin = sum(r["profit_margin"] for r in profitable_cities) / len(profitable_cities) if profitable_cities else 0
    
    texts = [
        f"""TOP PERFORMING CITIES ANALYSIS (By Profitability):
Only cities with positive profit margins and substantial profit generation (top 50% of profits) are considered. 
Top performing cities (ranked by profit margin): {top_str}.
Average profit margin across qualifying cities: {avg_margin:.2%}."""
    ]

    for r in top_cities:
        performance = "exceptional" if r["profit_margin"] > 0.20 else "outstanding" if r["profit_margin"] > 0.15 else "very strong" if r["profit_margin"] > 0.10 else "strong"
        texts.append(
            f"CITY: {r['City']} is a {performance} performer with {r['total_sales']:.0f} in sales, "
            f"generating {r['total_profit']:.0f} profit with margin of {r['profit_margin']:.2%} across {r['order_count']} orders. "
            f"Average discount: {r['avg_discount']:.2f}."
        )
    
    return texts


# ---------- Category Trends Comparison ----------
def category_trends_comparison(df: pl.DataFrame):
    df_with_date = df.with_columns(
        pl.col("Order Date").str.strptime(pl.Date, strict=False)
    )
    
    filtered_df = df_with_date.filter(
        pl.col("Category").is_in(["Technology", "Furniture"])
    )
    
    # Monthly trends by category
    monthly_trends = (
        filtered_df.with_columns(pl.col("Order Date").dt.truncate("1mo").alias("Month"))
        .drop_nulls("Month")
        .group_by(["Month", "Category"])
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit"),
            pl.count().alias("order_count")
        ])
        .sort("Month")
    )
    
    overall_comparison = (
        filtered_df.group_by("Category")
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit"),
            pl.count().alias("order_count"),
            pl.mean("Discount").alias("avg_discount")
        ])
        .with_columns([
            (pl.col("total_profit") / pl.col("total_sales")).alias("profit_margin")
        ])
    )
    
    overall_dicts = overall_comparison.to_dicts()
    tech = next((cat for cat in overall_dicts if cat["Category"] == "Technology"), None)
    furniture = next((cat for cat in overall_dicts if cat["Category"] == "Furniture"), None)
    
    texts = []
    
    if tech and furniture:
        tech_higher_sales = tech["total_sales"] > furniture["total_sales"]
        tech_higher_profit = tech["profit_margin"] > furniture["profit_margin"]
        
        better_sales_cat = "Technology" if tech_higher_sales else "Furniture"
        better_profit_cat = "Technology" if tech_higher_profit else "Furniture"
        
        texts.append(
            f"""TECHNOLOGY VS FURNITURE SALES TRENDS COMPARISON:
Overall Performance: {better_sales_cat} has higher total sales ({max(tech['total_sales'], furniture['total_sales']):.0f} vs {min(tech['total_sales'], furniture['total_sales']):.0f}).
Profitability: {better_profit_cat} has better profit margin ({max(tech['profit_margin'], furniture['profit_margin']):.2%} vs {min(tech['profit_margin'], furniture['profit_margin']):.2%})."""
        )
        
        texts.append(
            f"TECHNOLOGY: Total sales {tech['total_sales']:.0f}, profit {tech['total_profit']:.0f} ({tech['profit_margin']:.2%} margin) "
            f"across {tech['order_count']} orders. Average discount: {tech['avg_discount']:.2f}."
        )
        
        texts.append(
            f"FURNITURE: Total sales {furniture['total_sales']:.0f}, profit {furniture['total_profit']:.0f} ({furniture['profit_margin']:.2%} margin) "
            f"across {furniture['order_count']} orders. Average discount: {furniture['avg_discount']:.2f}."
        )
        
        monthly_dicts = monthly_trends.to_dicts()
        if monthly_dicts:
            tech_recent = [m for m in monthly_dicts if m["Category"] == "Technology"][-1:]
            furn_recent = [m for m in monthly_dicts if m["Category"] == "Furniture"][-1:]
            
            if tech_recent and furn_recent:
                tech_trend = tech_recent[0]
                furn_trend = furn_recent[0]
                
                texts.append(
                    f"RECENT TREND (Latest Month): Technology {tech_trend['total_sales']:.0f} sales vs Furniture {furn_trend['total_sales']:.0f} sales. "
                    f"Technology profit margin trend: {(tech_trend['total_profit'] / tech_trend['total_sales']):.2%}, "
                    f"Furniture profit margin trend: {(furn_trend['total_profit'] / furn_trend['total_sales']):.2%}."
                )
    
    return texts


# ---------- Regional Profit Comparison: West vs East ----------
def region_profit_comparison(df: pl.DataFrame):
    region_profits = (
        df.filter(pl.col("Region").is_in(["West", "East"]))
        .group_by("Region")
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit"),
            pl.count().alias("order_count"),
            pl.mean("Discount").alias("avg_discount"),
            pl.mean("Sales").alias("avg_order_value")
        ])
        .with_columns([
            (pl.col("total_profit") / pl.col("total_sales")).alias("profit_margin")
        ])
    )
    
    region_dicts = region_profits.to_dicts()
    west = next((r for r in region_dicts if r["Region"] == "West"), None)
    east = next((r for r in region_dicts if r["Region"] == "East"), None)
    
    texts = []
    
    if west and east:
        profit_difference = west["total_profit"] - east["total_profit"]
        margin_difference = west["profit_margin"] - east["profit_margin"]
        
        west_higher_profit = profit_difference > 0
        west_higher_margin = margin_difference > 0
        
        better_region = "West" if west_higher_profit else "East"
        better_margin_region = "West" if west_higher_margin else "East"
        
        texts.append(
            f"""WEST VS EAST REGIONS: PROFIT COMPARISON:
Total Profit Comparison: {better_region} region generated higher profit ({max(west['total_profit'], east['total_profit']):.0f} vs {min(west['total_profit'], east['total_profit']):.0f}). 
Difference: {abs(profit_difference):.0f}.
Profit Margin Comparison: {better_margin_region} region has better profit margin ({max(west['profit_margin'], east['profit_margin']):.2%} vs {min(west['profit_margin'], east['profit_margin']):.2%})."""
        )
        
        texts.append(
            f"WEST REGION: Total profit {west['total_profit']:.0f} from {west['total_sales']:.0f} in sales "
            f"(margin: {west['profit_margin']:.2%}) across {west['order_count']} orders. "
            f"Average order value: {west['avg_order_value']:.2f}, Average discount: {west['avg_discount']:.2f}."
        )
        
        texts.append(
            f"EAST REGION: Total profit {east['total_profit']:.0f} from {east['total_sales']:.0f} in sales "
            f"(margin: {east['profit_margin']:.2%}) across {east['order_count']} orders. "
            f"Average order value: {east['avg_order_value']:.2f}, Average discount: {east['avg_discount']:.2f}."
        )
        
        west_efficiency = west["total_profit"] / west["order_count"]
        east_efficiency = east["total_profit"] / east["order_count"]
        
        texts.append(
            f"Profit per Order: {better_region} region generates more profit per order. "
            f"West: {west_efficiency:.2f} per order, East: {east_efficiency:.2f} per order."
        )
    
    return texts