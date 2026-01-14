import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# --- 1. Load CSV or Create Dummy Data ---
try:
    ho = pd.read_csv('housing.csv')
    print("Successfully loaded 'housing.csv'")
except :
    FileNotFoundError
    print("Error: 'housing.csv' not found. Creating a dummy DataFrame for demonstration.")
    
    num_samples = 200
    data = {
        'longitude': np.random.uniform(-125, -115, num_samples),
        'latitude': np.random.uniform(32, 42, num_samples),
        'housing_median_age': np.random.randint(5, 50, num_samples),
        'total_rooms': np.random.randint(100, 5000, num_samples),
        'total_bedrooms': np.random.randint(50, 1000, num_samples),
        'population': np.random.randint(100, 3000, num_samples),
        'households': np.random.randint(50, 1000, num_samples),
        'median_income': np.random.uniform(1, 15, num_samples),
        'median_house': np.random.randint(50000, 50050, num_samples),
        'ocean_proximity': np.random.choice(['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'], num_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'chas': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'zn': np.random.choice([0, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100], num_samples, p=[0.5] + [0.5/13]*13)
    }
    ho = pd.DataFrame(data)
    ho.loc[np.random.choice(ho.index, 10, replace=False), 'total_bedrooms'] = np.nan
    print("Dummy DataFrame created.")

# --- 2. Initial Data Exploration ---
print("\n--- Data Info ---")
print(ho.head())
ho.info()
print("\nMissing values per column:\n", ho.isnull().sum())
print("\n'chas' value counts:\n", ho['chas'].value_counts())
print("\n'zn' value counts:\n", ho['zn'].value_counts())
print("\n--- Descriptive Statistics ---")
print(ho.describe().round(3))

print("\nColumns available for plotting:", ho.columns.tolist())
plt.figure(figsize=(10, 6))
plt.hist(ho['median_house'],edgecolor='black',height='300')
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.legend()
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(ho['median_income'], bins=50, edgecolor='black')
plt.title('Distribution of Median Income')
plt.xlabel('Median Income')
plt.ylabel('Frequency')

plt.grid(axis='y', alpha=0.75)
plt.show()

print("\n--- Random Train-Test Split ---")
train_set_random, test_set_random = train_test_split(ho, test_size=0.3, random_state=42)
print(f'Train Set Size: {len(train_set_random)}')
print(f'Test Set Size: {len(test_set_random)}')
print("\n--- Stratified Split on 'chas' ---")
chas_counts = ho['chas'].value_counts()
if (chas_counts >= 2).all():
    split_chas = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=45)
    for train_idx, test_idx in split_chas.split(ho, ho['chas']):
        strat_train_chas = ho.iloc[train_idx]
        strat_test_chas = ho.iloc[test_idx]

    print("\nDistribution of 'chas':")
    print("Original:\n", ho['chas'].value_counts(normalize=True))
    print("Train:\n", strat_train_chas['chas'].value_counts(normalize=True))
    print("Test:\n", strat_test_chas['chas'].value_counts(normalize=True))
else:
    print("Not enough samples in 'chas' categories for stratification.")
print("\n--- Stratified Split on Binned 'median_income' ---")
ho['income_cat'] = pd.cut(
    ho['median_income'],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)
income_cat_counts = ho['income_cat'].value_counts()
print("\nIncome category counts:\n", income_cat_counts)
if (income_cat_counts >= 2).all():
    split_income = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=45)
    for train_idx, test_idx in split_income.split(ho, ho['income_cat']):
        strat_train_income = ho.iloc[train_idx].drop("income_cat", axis=1)
        strat_test_income = ho.iloc[test_idx].drop("income_cat", axis=1)
    ho.drop("income_cat", axis=1, inplace=True)
    print("\nDistribution of 'income_cat':")
    print("Original:\n", income_cat_counts.sort_index(normalize=True))
    print("Train:\n", strat_train_income['median_income'].describe())
    print("Test:\n", strat_test_income['median_income'].describe())
else:
    print("Not enough samples in 'income_cat' bins for stratification.")
    ho.drop("income_cat", axis=1, inplace=True)
print("\n--- Code Execution Complete ---")



