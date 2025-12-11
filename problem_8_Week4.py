"""
Problem 8: Project Employees I
LeetCode #1075

Write a solution that reports the average experience years of all the employees 
for each project, rounded to 2 digits.

Table: Project
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| project_id  | int     |
| employee_id | int     |
+-------------+---------+
(project_id, employee_id) is the primary key.

Table: Employee
+------------------+---------+
| Column Name      | Type    |
+------------------+---------+
| employee_id      | int     |
| name             | varchar |
| experience_years | int     |
+------------------+---------+
employee_id is the primary key.
"""

import pandas as pd

def project_employees_i(project: pd.DataFrame, employee: pd.DataFrame) -> pd.DataFrame:
    """
    Solution:
    1. Join Project with Employee on employee_id
    2. Group by project_id
    3. Calculate the average experience_years for each project
    4. Round to 2 decimal places
    """
    # Join the two tables
    merged = project.merge(employee, on='employee_id')
    
    # Group by project_id and calculate average experience years
    result = merged.groupby('project_id')['experience_years'].mean().round(2).reset_index()
    
    # Rename column to match expected output
    result.columns = ['project_id', 'average_years']
    
    return result


# Test case
if __name__ == "__main__":
    # Sample data
    project_data = {
        'project_id': [1, 1, 1, 2, 2],
        'employee_id': [1, 2, 3, 1, 4]
    }
    employee_data = {
        'employee_id': [1, 2, 3, 4],
        'name': ['Khaled', 'Ali', 'John', 'Doe'],
        'experience_years': [3, 2, 1, 2]
    }
    project_df = pd.DataFrame(project_data)
    employee_df = pd.DataFrame(employee_data)
    print(project_employees_i(project_df, employee_df))
