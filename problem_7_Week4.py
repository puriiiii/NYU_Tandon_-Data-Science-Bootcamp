"""
Problem 7: Game Play Analysis IV
LeetCode #550

Write a solution to report the fraction of players that logged in again on the 
day after the day they first logged in, rounded to 2 decimal places.

Table: Activity
+--------------+---------+
| Column Name  | Type    |
+--------------+---------+
| player_id    | int     |
| device_id    | int     |
| event_date   | date    |
| games_played | int     |
+--------------+---------+
(player_id, event_date) is the primary key.
"""

import pandas as pd

def gameplay_analysis(activity: pd.DataFrame) -> pd.DataFrame:
    """
    Solution:
    1. Find the first login date for each player
    2. Check if the player logged in the day after their first login
    3. Calculate the fraction: (players who returned next day) / (total players)
    4. Round to 2 decimal places
    """
    # Convert event_date to datetime
    activity['event_date'] = pd.to_datetime(activity['event_date'])
    
    # Get the first login date for each player
    first_login = activity.groupby('player_id')['event_date'].min().reset_index()
    first_login.columns = ['player_id', 'first_login']
    
    # Join with original activity to add first_login column
    merged = activity.merge(first_login, on='player_id')
    
    # Check if player logged in the day after their first login
    merged['next_day_login'] = (merged['event_date'] == merged['first_login'] + pd.Timedelta(days=1))
    
    # Calculate fraction
    total_players = activity['player_id'].nunique()
    players_returned_next_day = merged[merged['next_day_login']]['player_id'].nunique()
    
    fraction = round(players_returned_next_day / total_players, 2)
    
    return pd.DataFrame({'fraction': [fraction]})


# Test case
if __name__ == "__main__":
    # Sample data
    data = {
        'player_id': [1, 1, 2, 3, 3],
        'device_id': [2, 2, 3, 1, 4],
        'event_date': ['2016-03-01', '2016-03-02', '2017-06-25', '2016-03-02', '2018-07-03'],
        'games_played': [5, 6, 1, 0, 5]
    }
    df = pd.DataFrame(data)
    print(gameplay_analysis(df))
