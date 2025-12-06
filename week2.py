import numpy as np
import pandas as pd

# ==========================================
# PART 1: NUMPY QUESTIONS
# ==========================================
print("--- NumPy Solutions ---")

# 1. Define two custom numpy arrays (A and B)
A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
B = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

# Generate new arrays by stacking vertically and horizontally
vert_stack = np.vstack((A, B))
horz_stack = np.hstack((A, B))

print(f"1. Vertical Stack:\n{vert_stack}")
print(f"   Horizontal Stack:\n{horz_stack}\n")

# 2. Find common elements between A and B
common_elements = np.intersect1d(A, B)
print(f"2. Common elements (Intersection): {common_elements}\n")

# 3. Extract numbers from A which are within a specific range (e.g., between 5 and 10)
# Using boolean masking
mask = (A >= 5) & (A <= 10)
extracted_nums = A[mask]
print(f"3. Numbers in A between 5 and 10: {extracted_nums}\n")

# 4. Filter rows of iris_2d
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])

# Conditions: PetalLength (3rd col, index 2) > 1.5 AND SepalLength (1st col, index 0) < 5.0
condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
filtered_iris = iris_2d[condition]

print(f"4. Filtered Iris rows (showing first 5):\n{filtered_iris[:5]}\n")


# ==========================================
# PART 2: PANDAS QUESTIONS
# ==========================================
print("--- Pandas Solutions ---")

# 1. Filter specific columns for every 20th row
df_cars = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# slice[start:stop:step] -> ::20 gets every 20th row
filtered_cars = df_cars.iloc[::20][['Manufacturer', 'Model', 'Type']]
print("1. Every 20th row (Manufacturer, Model, Type):")
print(filtered_cars)
print("\n")

# 2. Replace missing values in Min.Price and Max.Price with mean
# Note: We reload the df to ensure we are working on raw data, though it's the same source
df_miss = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

min_price_mean = df_miss['Min.Price'].mean()
max_price_mean = df_miss['Max.Price'].mean()

df_miss['Min.Price'] = df_miss['Min.Price'].fillna(min_price_mean)
df_miss['Max.Price'] = df_miss['Max.Price'].fillna(max_price_mean)

print("2. Missing values imputed.")
print(f"   New Min.Price NaNs count: {df_miss['Min.Price'].isna().sum()}")
print(f"   New Max.Price NaNs count: {df_miss['Max.Price'].isna().sum()}\n")

# 3. Get rows of a dataframe with row sum > 100
# Setting a seed for reproducibility
np.random.seed(100)
df_random = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))

# axis=1 calculates the sum across columns (for each row)
rows_sum_gt_100 = df_random[df_random.sum(axis=1) > 100]

print("3. Rows with sum > 100:")
print(rows_sum_gt_100)
