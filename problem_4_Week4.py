"""
Problem 4: Second Highest Salary
LeetCode #176

Write a solution to find the second highest distinct salary from the Employee table.
If there is no second highest salary, return null.

Table: Employee
+-------------+------+
| Column Name | Type |
+-------------+------+
| id          | int  |
| salary      | int  |
+-------------+------+
id is the primary key.
"""

import pandas as pd

def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:
    """
    Solution:
    1. Get distinct salaries
    2. Sort in descending order
    3. Get the second value if it exists, otherwise return None
    4. Return as DataFrame with column name 'SecondHighestSalary'
    """
    # Get unique salaries and sort in descending order
    distinct_salaries = employee['salary'].drop_duplicates().sort_values(ascending=False)
    
    # Check if there's a second highest salary
    if len(distinct_salaries) < 2:
        return pd.DataFrame({'SecondHighestSalary': [None]})
    
    # Get the second highest salary
    second_highest = distinct_salaries.iloc[1]
    
    return pd.DataFrame({'SecondHighestSalary': [second_highest]})


# Test case
if __name__ == "__main__":
    # Sample data with second highest
    data1 = {
        'id': [1, 2, 3],
        'salary': [100, 200, 300]
    }
    df1 = pd.DataFrame(data1)
    print("Test 1:", second_highest_salary(df1))
    
    # Sample data without second highest
    data2 = {
        'id': [1],
        'salary': [100]
    }
    df2 = pd.DataFrame(data2)
    print("Test 2:", second_highest_salary(df2))
