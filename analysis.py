import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

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

def iqr_bounds(series, k=1.5):
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return q1, q3, iqr, lower, upper

# outlier check
Q1, Q3, IQR, lower, upper = iqr_bounds(df["Price ($)"])
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
# Histrogram: price and frequency
sns.set_style("whitegrid")
plt.figure(figsize=(10,6))
sns.histplot(df["Price ($)"], bins=40, kde=True)
plt.title("Distribution of Car Prices", fontsize=16)
plt.xlabel("Price ($)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

# Histogram: Number of Cars Sold by Brand
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

#Descriptive Statistics
# Helper: Summary for a numeric column
def numeric_summary(series, name=""):
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    summary = {
        "variable": name,
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "Q1": float(q1),
        "Q3": float(q3),
        "IQR": float(iqr),
        "max": float(s.max()),
    }
    return summary

# 1) Overall numeric summaries
price_sum = numeric_summary(df["Price ($)"], "Price ($)")
income_sum = numeric_summary(df["Annual Income"], "Annual Income")

desc_df = pd.DataFrame([price_sum, income_sum])

# Round to 2 decimals for reporting
desc_df_rounded = desc_df.copy()
for c in ["mean", "median", "std", "min", "Q1", "Q3", "IQR", "max"]:
    desc_df_rounded[c] = desc_df_rounded[c].round(2)

print("\nOverall summaries (rounded to 2 decimals):")
print(desc_df_rounded.to_string(index=False))

# 2) Categorical counts (basic descriptive)
print("\nCategory counts:")
for col in ["Gender", "Transmission", "Body Style", "Company"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))  # show top 10 only (Company can be long)

# 3) Grouped descriptive stats for Price by key categories
#    (These are very useful for report tables + interpretation)
group_cols = ["Transmission", "Gender", "Body Style"]
for g in group_cols:
    if g in df.columns:
        grp = df.groupby(g)["Price ($)"].agg(["count", "mean", "median", "std"]).round(2)
        print(f"\nPrice ($) grouped by {g}:")
        print(grp.to_string())

# 4) Flag price outliers using  (for transparency)
Q1, Q3, IQR, lower, upper = iqr_bounds(df["Price ($)"])

df["Price_Outlier_IQR"] = (df["Price ($)"] < lower) | (df["Price ($)"] > upper)

outlier_count = df["Price_Outlier_IQR"].sum()
outlier_pct = (outlier_count / len(df)) * 100

print("\nOutlier flag (IQR rule) for Price ($):")
print(f"- Lower bound = {lower:.2f}, Upper bound = {upper:.2f}")
print(f"- Number of outliers = {outlier_count} ({outlier_pct:.2f}%)")

# Inferential Statistics
#1) CI for mean Price
price = df["Price ($)"].dropna()

mean_price = price.mean()
std_price = price.std(ddof=1)
n = len(price)

# 95% CI using t-distribution
confidence = 0.95
t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)

margin_error = t_critical * std_price / np.sqrt(n)

ci_lower = mean_price - margin_error
ci_upper = mean_price + margin_error

print(f"Mean price: {mean_price:.2f}")
print(f"95% CI for mean price: ({ci_lower:.2f}, {ci_upper:.2f})")
#2) T-test: Price by Transmission
auto_price = df[df["Transmission"] == "Auto"]["Price ($)"].dropna()
manual_price = df[df["Transmission"] == "Manual"]["Price ($)"].dropna()

t_stat, p_value = stats.ttest_ind(auto_price, manual_price, equal_var=False)

print(f"Auto mean price: {auto_price.mean():.2f}")
print(f"Manual mean price: {manual_price.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

print("H0: mean price (Auto) = mean price (Manual)")
print("H1: mean price (Auto) != mean price (Manual)")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: Mean prices are significantly different.")
else:
    print("Fail to reject H0: No significant difference in mean prices.")