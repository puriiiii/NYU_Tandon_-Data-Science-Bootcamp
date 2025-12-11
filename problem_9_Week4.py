"""
Problem 9: Department Top Three Salaries
LeetCode #185

A company's executives are interested in seeing who earns the most money in each 
department. A high earner is an employee who has a salary in the top three unique 
salaries for that department.

Write a solution to find the employees who are high earners in each department.

Table: Employee
+--------------+---------+
| Column Name  | Type    |
+--------------+---------+
| id           | int     |
| name         | varchar |
| salary       | int     |
| departmentId | int     |
+--------------+---------+
id is the primary key.

Table: Department
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| name        | varchar |
+-------------+---------+
id is the primary key.
"""

import pandas as pd

def top_three_salaries(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:
    """
    Solution:
    1. Join Employee with Department tables
    2. Use dense_rank to rank salaries within each department
    3. Filter for employees with rank <= 3 (top 3 unique salaries)
    4. Return Department, Employee, and Salary columns
    """
    # Join employee and department tables
    merged = employee.merge(
        department, 
        left_on='departmentId', 
        right_on='id', 
        suffixes=('_emp', '_dept')
    )
    
    # Rank salaries within each department using dense_rank
    # dense_rank handles ties properly (same salary = same rank)
    merged['rank'] = merged.groupby('departmentId')['salary'].rank(
        method='dense', 
        ascending=False
    )
    
    # Filter for top 3 ranks
    result = merged[merged['rank'] <= 3][['name_dept', 'name_emp', 'salary']].copy()
    
    # Rename columns to match expected output
    result.columns = ['Department', 'Employee', 'Salary']
    
    return result


# Test case
if __name__ == "__main__":
    # Sample data
    employee_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Joe', 'Jim', 'Henry', 'Sam', 'Max'],
        'salary': [85000, 90000, 80000, 60000, 90000],
        'departmentId': [1, 1, 2, 2, 1]
    }
    department_data = {
        'id': [1, 2],
        'name': ['IT', 'Sales']
    }
    employee_df = pd.DataFrame(employee_data)
    department_df = pd.DataFrame(department_data)
    print(top_three_salaries(employee_df, department_df))
