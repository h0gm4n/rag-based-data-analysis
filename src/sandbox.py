import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

df = pl.read_csv("data/superstore.csv", encoding="latin1")
df = df.with_columns(
    pl.col("Order Date").str.strptime(pl.Date, "%m/%d/%Y")
)
df = df.sort("Order Date")

df = df.select(["Order Date", "Sales"])

with pl.Config(tbl_rows=100):
    print(df)

# Aggregate sales by month
df_monthly = df.with_columns(
    pl.col("Order Date").dt.truncate("1mo").alias("Month")
).group_by("Month").agg(
    pl.col("Sales").sum().alias("Total Sales")
).sort("Month")

with pl.Config(tbl_rows=100):
    print(df_monthly.sort("Total Sales"))

# Create line graph with monthly aggregated data
plt.figure(figsize=(12, 6))
plt.plot(df_monthly["Month"], df_monthly["Total Sales"], linewidth=2, marker='o', markersize=4, label='Monthly Sales')

# Add trend line
x_numeric = mdates.date2num(df_monthly["Month"].to_list())
y_values = df_monthly["Total Sales"].to_numpy()
z = np.polyfit(x_numeric, y_values, 1)
p = np.poly1d(z)
plt.plot(x_numeric, p(x_numeric), "r--", linewidth=2, label='Trend Line')

plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Sales Trend Over 4-Year Period (Monthly Aggregation)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
