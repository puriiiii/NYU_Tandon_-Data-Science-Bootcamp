"""
Problem 1: Actors and Directors Who Cooperated At Least Three Times
LeetCode #1050

Write a solution to find all the pairs (actor_id, director_id) where the actor 
has cooperated with the director at least three times.

Table: ActorDirector
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| actor_id    | int     |
| director_id | int     |
| timestamp   | int     |
+-------------+---------+
timestamp is the primary key (column with unique values).
"""

import pandas as pd

def actors_and_directors(actor_director: pd.DataFrame) -> pd.DataFrame:
    """
    Solution:
    1. Group by both actor_id and director_id
    2. Count the number of cooperations
    3. Filter pairs with count >= 3
    4. Return actor_id and director_id columns
    """
    # Group by actor_id and director_id, count rows
    cooperation_counts = actor_director.groupby(['actor_id', 'director_id']).size()
    
    # Filter for pairs with at least 3 cooperations
    result = cooperation_counts[cooperation_counts >= 3].reset_index()[['actor_id', 'director_id']]
    
    return result


# Test case
if __name__ == "__main__":
    # Sample data
    data = {
        'actor_id': [1, 1, 1, 1, 1, 2, 2],
        'director_id': [1, 1, 1, 2, 2, 1, 1],
        'timestamp': [0, 1, 2, 3, 4, 5, 6]
    }
    df = pd.DataFrame(data)
    print(actors_and_directors(df))
