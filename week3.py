import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. Load and Preprocess Data
# ==========================================
# Replace 'brooklyn_bridge_pedestrians.csv' with your actual file path or URL
# Common URL: 'https://data.cityofnewyork.us/api/views/6fi9-q3ta/rows.csv?accessType=DOWNLOAD'
file_path = 'brooklyn_bridge_pedestrians.csv'

# Load data
# Ensure 'hour_beginning' is parsed as a datetime object
try:
    df = pd.read_csv(file_path, parse_dates=['hour_beginning'])
except FileNotFoundError:
    print("File not found. Please update the 'file_path' variable.")
    # Creating dummy data for demonstration if file is missing
    dates = pd.date_range(start='2019-01-01', end='2019-12-31', freq='H')
    import numpy as np
    df = pd.DataFrame({
        'hour_beginning': dates,
        'pedestrians': np.random.randint(0, 1000, size=len(dates)),
        'weather_summary': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Snow'], size=len(dates)),
        'temperature': np.random.uniform(20, 90, size=len(dates)),
        'precipitation': np.random.uniform(0, 1, size=len(dates))
    })

# Extract useful time features
df['day_name'] = df['hour_beginning'].dt.day_name()
df['hour'] = df['hour_beginning'].dt.hour
df['year'] = df['hour_beginning'].dt.year

print("Data Loaded. Columns:", df.columns)


# ==========================================
# Question 1: Filter Weekdays & Plot Line Graph
# ==========================================
print("\n--- Question 1: Weekday Pedestrian Counts ---")

# 1. Filter for Weekdays (Monday to Friday)
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
weekday_df = df[df['day_name'].isin(weekdays)]

# 2. Aggregate counts by Day of Week
# We must ensure the days are sorted correctly, not alphabetically
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
daily_counts = weekday_df.groupby('day_name')['pedestrians'].sum().reindex(weekday_order)

# 3. Plot Line Graph
plt.figure(figsize=(10, 6))
plt.plot(daily_counts.index, daily_counts.values, marker='o', linestyle='-', color='b')
plt.title('Total Pedestrian Counts by Weekday')
plt.xlabel('Day of Week')
plt.ylabel('Total Pedestrians')
plt.grid(True)
plt.show()


# ==========================================
# Question 2: 2019 Analysis & Weather Correlation
# ==========================================
print("\n--- Question 2: 2019 Weather & Pedestrian Correlation ---")

# 1. Track counts for 2019
df_2019 = df[df['year'] == 2019].copy()

if not df_2019.empty:
    # 2. Analyze influence of weather summaries (e.g., average pedestrians per weather type)
    weather_influence = df_2019.groupby('weather_summary')['pedestrians'].mean().sort_values(ascending=False)
    print("Average Pedestrian Count by Weather Condition (2019):")
    print(weather_influence)

    # 3. Sort data by weather summary
    df_2019_sorted = df_2019.sort_values(by='weather_summary')

    # 4. Correlation Matrix
    # Select numerical columns relevant to weather and pedestrians
    corr_cols = ['pedestrians', 'temperature', 'precipitation']
    correlation_matrix = df_2019[corr_cols].corr()

    print("\nCorrelation Matrix (2019):")
    print(correlation_matrix)

    # Visualization of Correlation Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation: Weather vs Pedestrian Counts (2019)')
    plt.show()
else:
    print("No data found for the year 2019.")


# ==========================================
# Question 3: Time of Day Categorization & Analysis
# ==========================================
print("\n--- Question 3: Time of Day Analysis ---")

# 1. Implement custom function to categorize time of day
def categorize_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night' # Covers 21-24 and 0-6

# 2. Create new column
df['time_category'] = df['hour'].apply(categorize_time_of_day)

# 3. Analyze patterns (e.g., Average counts per time category)
# Define order for logical plotting
time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
time_analysis = df.groupby('time_category')['pedestrians'].mean().reindex(time_order)

print("Average Pedestrians by Time of Day:")
print(time_analysis)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=time_analysis.index, y=time_analysis.values, palette='viridis')
plt.title('Average Pedestrian Activity by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Average Count')
plt.show()
