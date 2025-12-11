"""
Problem 3: Combine Two Tables
LeetCode #175

Write a solution to report the first name, last name, city, and state of each 
person in the Person table. If the address of a personId is not present in the 
Address table, report null instead.

Table: Person
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| personId    | int     |
| lastName    | varchar |
| firstName   | varchar |
+-------------+---------+
personId is the primary key.

Table: Address
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| addressId   | int     |
| personId    | int     |
| city        | varchar |
| state       | varchar |
+-------------+---------+
addressId is the primary key.
"""

import pandas as pd

def combine_two_tables(person: pd.DataFrame, address: pd.DataFrame) -> pd.DataFrame:
    """
    Solution:
    1. Perform a LEFT JOIN between Person and Address on personId
    2. This ensures all persons are included even without addresses
    3. Return firstName, lastName, city, and state columns
    """
    # Left join to keep all persons
    result = person.merge(address, on='personId', how='left')
    
    # Select required columns
    result = result[['firstName', 'lastName', 'city', 'state']]
    
    return result


# Test case
if __name__ == "__main__":
    # Sample data
    person_data = {
        'personId': [1, 2],
        'lastName': ['Wang', 'Alice'],
        'firstName': ['Allen', 'Bob']
    }
    address_data = {
        'addressId': [1, 2],
        'personId': [2, 3],
        'city': ['New York City', 'Leetcode'],
        'state': ['New York', 'California']
    }
    person_df = pd.DataFrame(person_data)
    address_df = pd.DataFrame(address_data)
    print(combine_two_tables(person_df, address_df))
