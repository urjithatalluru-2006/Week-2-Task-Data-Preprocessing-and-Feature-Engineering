import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("automobile.csv.csv")

print("Dataset Shape:")
print(df.shape)

# ---------------------
# Missing Values
# ---------------------

print("\nMissing Values:")
print(df.isnull().sum())

plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.savefig("outputs_missing_values.png")
plt.show()

# ---------------------
# Remove Duplicates
# ---------------------

print("\nBefore:", df.shape)

df.drop_duplicates(inplace=True)

print("After:", df.shape)

# ---------------------
# Outlier Detection
# ---------------------

plt.figure(figsize=(8,5))
sns.boxplot(x=df["price"])
plt.title("Price Before Outlier Removal")
plt.savefig("boxplot_before.png")
plt.show()

Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["price"] >= lower)
    &
    (df["price"] <= upper)
]

plt.figure(figsize=(8,5))
sns.boxplot(x=df["price"])
plt.title("Price After Outlier Removal")
plt.savefig("boxplot_after.png")
plt.show()

# ---------------------
# Feature Engineering
# ---------------------

df["power_engine_ratio"] = (
    df["horsepower"] /
    df["enginesize"]
)

df["fuel_economy_score"] = (
    df["citympg"] +
    df["highwaympg"]
) / 2

df["price_per_hp"] = (
    df["price"] /
    df["horsepower"]
)

print("\nNew Features Created")

# ---------------------
# Distribution Plot
# ---------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df["fuel_economy_score"],
    kde=True
)

plt.title("Fuel Economy Score")
plt.savefig("feature_distribution.png")
plt.show()

# ---------------------
# Encode Categorical Data
# ---------------------

df = pd.get_dummies(
    df,
    drop_first=True
)

# ---------------------
# Scaling
# ---------------------

scaler = StandardScaler()

num_cols = df.select_dtypes(
    include=np.number
).columns

df[num_cols] = scaler.fit_transform(
    df[num_cols]
)

# ---------------------
# Correlation Heatmap
# ---------------------

plt.figure(figsize=(16,10))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "correlation_heatmap.png"
)

plt.show()

# ---------------------
# Save Dataset
# ---------------------

df.to_csv(
    "processed_automobile.csv",
    index=False
)

print("\nProject Completed Successfully!")
