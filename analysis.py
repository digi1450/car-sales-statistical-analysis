import pandas as pd

df = pd.read_csv("Carsale.csv")

print(df.shape)
print(df.head())
# Data cleaning
# check
print("\nData Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# DMY
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# negativeprice check
print("\nCheck negative price:")
print((df["Price ($)"] <= 0).sum())

print("\nCheck negative income:")
print((df["Annual Income"] <= 0).sum())

# outlier check
Q1 = df["Price ($)"].quantile(0.25)
Q3 = df["Price ($)"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["Price ($)"] < lower) | (df["Price ($)"] > upper)]
print("Number of price outliers:", len(outliers))

# check carduplicate_id
duplicate_count= df["Car_id"].duplicated().sum()
print("Number of duplicated Car_id:", duplicate_count)

# check category
print(df["Gender"].unique())
print(df["Transmission"].unique())
print(df["Body Style"].unique())

# EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Histrogram price and frequency
sns.set_style("whitegrid")
plt.figure(figsize=(10,6))
sns.histplot(df["Price ($)"], bins=40, kde=True)
plt.title("Distribution of Car Prices", fontsize=16)
plt.xlabel("Price ($)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

# Histrogram  Number of Cars Sold by Brand
sns.set_style("whitegrid")
plt.figure(figsize=(12,6))
brand_counts = df["Company"].value_counts()
sns.barplot(
    x=brand_counts.index,
    y=brand_counts.values
)
plt.xticks(rotation=45)
plt.title("Number of Cars Sold by Brand", fontsize=16)
plt.xlabel("Brand", fontsize=12)
plt.ylabel("Number of Cars Sold", fontsize=12)
plt.tight_layout()
plt.show()

# Boxplot to check outlier
plt.boxplot(df["Price ($)"])
plt.title("Boxplot of Car Prices")
plt.show()

# Boxplot Auto vs manual
plt.figure(figsize=(10,6))
sns.boxplot(x="Transmission", y="Price ($)", data=df)
plt.title("Price Distribution by Transmission Type")
plt.show()

plt.figure(figsize=(10,6))

# Boxplot Body vs Price
sns.boxplot(
    x="Body Style",
    y="Price ($)",
    data=df
)
plt.xticks(rotation=45)
plt.title("Car Price by Body Style")
plt.show()

# Income vs Price
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))

sns.scatterplot(
    x="Annual Income",
    y="Price ($)",
    data=df,
    alpha=0.5
)
plt.title("Annual Income vs Car Price")
plt.xlabel("Annual Income")
plt.ylabel("Car Price ($)")
plt.tight_layout()
plt.show()