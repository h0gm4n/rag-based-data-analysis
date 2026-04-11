# src/load_data.py
import polars as pl

def load_data(path: str):
    df = pl.read_csv(path)
    return df

def row_to_text(row):
    return f"""
    Order from {row['Order Date']}:
    Customer: {row['Customer Name']} ({row['Segment']})
    Product: {row['Product Name']} ({row['Category']} - {row['Sub-Category']})
    Location: {row['City']}, {row['State']}, {row['Region']}
    Sales: ${row['Sales']}, Profit: ${row['Profit']}, Discount: {row['Discount']}
    """

