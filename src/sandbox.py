import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Revenue by product category
df_full = pl.read_csv("data/superstore.csv", encoding="latin1")
category_revenue = df_full.group_by("Category").agg(
    pl.col("Sales").sum().alias("Revenue")
).sort("Revenue", descending=True)

with pl.Config(tbl_rows=100):
    print("\nRevenue by Product Category:")
    print(category_revenue)

# Profit margins by product sub-category
subcategory_margins = df_full.group_by("Sub-Category").agg(
    (pl.col("Profit").sum() / pl.col("Sales").sum()).alias("Profit Margin")
).sort("Profit Margin", descending=True)

with pl.Config(tbl_rows=100):
    print("\nProfit Margins by Product Sub-Category:")
    print(subcategory_margins)

products = df_full.group_by("Product Name").len()

with pl.Config(tbl_rows=100):
    print(products)

# Count times each product was sold at discount
discount_sales = df_full.filter(pl.col("Discount") > 0).group_by("Product Name").len().sort("len", descending=True)

with pl.Config(tbl_rows=100):
    print("\nDiscount Sales Count by Product:")
    print(discount_sales)

# Sales performance by region
region_performance = df_full.group_by("Region").agg(
    pl.col("Sales").sum().alias("Total Sales"),
    pl.col("Profit").sum().alias("Total Profit"),
    pl.count().alias("Order Count")
).with_columns(
    (pl.col("Total Profit") / pl.col("Total Sales")).alias("Profit Margin")
).sort("Total Sales", descending=True)

with pl.Config(tbl_rows=100):
    print("\nSales Performance by Region:")
    print(region_performance)

# Sales performance by state
state_performance = df_full.group_by("State").agg(
    pl.col("Sales").sum().alias("Total Sales"),
    pl.col("Profit").sum().alias("Total Profit"),
    pl.count().alias("Order Count")
).with_columns(
    (pl.col("Total Profit") / pl.col("Total Sales")).alias("Profit Margin")
).sort("Total Sales", descending=True)

with pl.Config(tbl_rows=100):
    print("\nSales Performance Comparison Across States:")
    print(state_performance)

# Sales performance by city
city_performance = df_full.group_by("City").agg(
    pl.col("Sales").sum().alias("Total Sales"),
    pl.col("Profit").sum().alias("Total Profit"),
    pl.count().alias("Order Count")
).with_columns(
    (pl.col("Total Profit") / pl.col("Total Sales")).alias("Profit Margin")
).sort("Total Sales", descending=True)

with pl.Config(tbl_rows=100):
    print("\nTop Performing Cities:")
    print(city_performance)

# Technology vs Furniture sales trends
df_full = df_full.with_columns(
    pl.col("Order Date").str.strptime(pl.Date, "%m/%d/%Y")
)

category_trends = df_full.with_columns(
    pl.col("Order Date").dt.truncate("1mo").alias("Month")
).filter(
    pl.col("Category").is_in(["Technology", "Furniture"])
).group_by(["Month", "Category"]).agg(
    pl.col("Sales").sum().alias("Total Sales"),
    pl.col("Profit").sum().alias("Total Profit")
).sort("Month")

with pl.Config(tbl_rows=100):
    print("\nTechnology vs Furniture Sales Trends:")
    print(category_trends)