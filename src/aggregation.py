import polars as pl


def monthly_sales_summary(df: pl.DataFrame):
    df = df.with_columns(
        pl.col("Order Date").str.strptime(pl.Date, strict=False)
    )

    result = (
        df.with_columns([
            pl.col("Order Date").dt.year().alias("year"),
            pl.col("Order Date").dt.month().alias("month")
        ])
        .groupby(["year", "month"])
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit"),
            pl.mean("Discount").alias("avg_discount")
        ])
        .sort(["year", "month"])
    )

    return [
        f"In {r['year']}-{r['month']:02d}, total sales were {r['total_sales']:.2f}, profit was {r['total_profit']:.2f}, and average discount was {r['avg_discount']:.2f}."
        for r in result.to_dicts()
    ]


def category_summary(df: pl.DataFrame):
    result = (
        df.groupby("Category")
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit"),
            pl.mean("Discount").alias("avg_discount")
        ])
        .sort("total_sales", descending=True)
    )

    return [
        f"Category {r['Category']} generated {r['total_sales']:.2f} in sales and {r['total_profit']:.2f} in profit, with avg discount {r['avg_discount']:.2f}."
        for r in result.to_dicts()
    ]


def regional_summary(df: pl.DataFrame):
    result = (
        df.groupby("Region")
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit")
        ])
        .sort("total_sales", descending=True)
    )

    return [
        f"Region {r['Region']} has total sales {r['total_sales']:.2f} and profit {r['total_profit']:.2f}."
        for r in result.to_dicts()
    ]


def top_products(df: pl.DataFrame, n: int = 5):
    result = (
        df.groupby("Product Name")
        .agg([
            pl.sum("Sales").alias("total_sales"),
            pl.sum("Profit").alias("total_profit")
        ])
        .sort("total_sales", descending=True)
        .head(n)
    )

    return [
        f"Top product: {r['Product Name']} with sales {r['total_sales']:.2f} and profit {r['total_profit']:.2f}."
        for r in result.to_dicts()
    ]