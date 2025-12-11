"""
Problem 2: Fix Names in a Table
LeetCode #1667

Write a solution to fix the names so that only the first character is uppercase 
and the rest are lowercase. Return the result table ordered by user_id.

Table: Users
+----------------+---------+
| Column Name    | Type    |
+----------------+---------+
| user_id        | int     |
| name           | varchar |
+----------------+---------+
user_id is the primary key (column with unique values).
"""

import pandas as pd

def fix_names(users: pd.DataFrame) -> pd.DataFrame:
    """
    Solution:
    1. Use str.capitalize() to format names (first letter uppercase, rest lowercase)
    2. Sort by user_id
    3. Return user_id and name columns
    """
    # Fix the name format
    users['name'] = users['name'].str.capitalize()
    
    # Sort by user_id
    result = users.sort_values('user_id')[['user_id', 'name']]
    
    return result


# Test case
if __name__ == "__main__":
    # Sample data
    data = {
        'user_id': [1, 2, 3],
        'name': ['aLice', 'bOB', 'CHARLIE']
    }
    df = pd.DataFrame(data)
    print(fix_names(df))
