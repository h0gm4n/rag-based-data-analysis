def row_to_text(row: dict) -> str:
    return f"""
Order Date: {row.get('Order Date')}
Customer: {row.get('Customer Name')} ({row.get('Segment')})
Product: {row.get('Product Name')} ({row.get('Category')} - {row.get('Sub-Category')})
Location: {row.get('City')}, {row.get('State')}, {row.get('Region')}
Sales: {row.get('Sales')}
Profit: {row.get('Profit')}
Discount: {row.get('Discount')}
"""


def dataframe_to_texts(df):
    return [row_to_text(r) for r in df.to_dicts()]