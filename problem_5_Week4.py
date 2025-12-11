"""
Problem 5: List the Products Ordered in a Period
LeetCode #1327

Write a solution to get the names of products that have at least 100 units 
ordered in February 2020 and their amount.

Table: Products
+------------------+---------+
| Column Name      | Type    |
+------------------+---------+
| product_id       | int     |
| product_name     | varchar |
| product_category | varchar |
+------------------+---------+
product_id is the primary key.

Table: Orders
+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| product_id    | int     |
| order_date    | date    |
| unit          | int     |
+---------------+---------+
There is no primary key. This table may have duplicate rows.
"""

import pandas as pd

def list_products(products: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """
    Solution:
    1. Convert order_date to datetime
    2. Filter orders for February 2020
    3. Group by product_id and sum units
    4. Filter products with total units >= 100
    5. Join with products table to get product names
    """
    # Convert order_date to datetime
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    
    # Filter for February 2020
    feb_2020_orders = orders[
        (orders['order_date'].dt.year == 2020) & 
        (orders['order_date'].dt.month == 2)
    ]
    
    # Group by product_id and sum units
    product_units = feb_2020_orders.groupby('product_id')['unit'].sum().reset_index()
    
    # Filter products with at least 100 units
    product_units = product_units[product_units['unit'] >= 100]
    
    # Join with products table to get product names
    result = products.merge(product_units, on='product_id')[['product_name', 'unit']]
    
    return result


# Test case
if __name__ == "__main__":
    # Sample data
    products_data = {
        'product_id': [1, 2, 3],
        'product_name': ['Leetcode Solutions', 'Jewels of Stringology', 'HP'],
        'product_category': ['Book', 'Book', 'Laptop']
    }
    orders_data = {
        'product_id': [1, 1, 2, 2, 3],
        'order_date': ['2020-02-05', '2020-02-10', '2020-01-18', '2020-02-11', '2020-02-17'],
        'unit': [60, 70, 30, 80, 2]
    }
    products_df = pd.DataFrame(products_data)
    orders_df = pd.DataFrame(orders_data)
    print(list_products(products_df, orders_df))
