"""
Problem 6: Replace Employee ID With The Unique Identifier
LeetCode #1378

Write a solution to show the unique ID of each user. If a user does not have 
a unique ID, replace it with null.

Table: Employees
+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| id            | int     |
| name          | varchar |
+---------------+---------+
id is the primary key.

Table: EmployeeUNI
+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| id            | int     |
| unique_id     | int     |
+---------------+---------+
(id, unique_id) is the primary key.
Each row contains the id and unique_id of an employee.
"""

import pandas as pd

def replace_employee_id(employees: pd.DataFrame, employee_uni: pd.DataFrame) -> pd.DataFrame:
    """
    Solution:
    1. Perform a LEFT JOIN between Employees and EmployeeUNI on id
    2. This ensures all employees are included
    3. Employees without unique_id will have null values
    4. Return unique_id and name columns
    """
    # Left join to keep all employees
    result = employees.merge(employee_uni, on='id', how='left')
    
    # Select required columns
    result = result[['unique_id', 'name']]
    
    return result


# Test case
if __name__ == "__main__":
    # Sample data
    employees_data = {
        'id': [1, 7, 11, 90, 3],
        'name': ['Alice', 'Bob', 'Meir', 'Winston', 'Jonathan']
    }
    employee_uni_data = {
        'id': [3, 11, 90],
        'unique_id': [1, 2, 3]
    }
    employees_df = pd.DataFrame(employees_data)
    employee_uni_df = pd.DataFrame(employee_uni_data)
    print(replace_employee_id(employees_df, employee_uni_df))
