import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------
# 1. LOAD DATASET
# -----------------------------------------

df = pd.read_csv("canteen_data.csv")

print("=" * 60)
print("SMARTCANTEEN AI - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# -----------------------------------------
# 2. BASIC INFORMATION
# -----------------------------------------

print("\n1. DATASET SHAPE")
print("-" * 30)
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\n2. COLUMN NAMES")
print("-" * 30)
print(df.columns.tolist())

print("\n3. FIRST 10 RECORDS")
print("-" * 30)
print(df.head(10).to_string(index=False))

# -----------------------------------------
# 3. DATA TYPES
# -----------------------------------------

print("\n4. DATA TYPES")
print("-" * 30)
print(df.dtypes)

# -----------------------------------------
# 4. MISSING VALUES
# -----------------------------------------

print("\n5. MISSING VALUES")
print("-" * 30)
print(df.isnull().sum())

# -----------------------------------------
# 5. DUPLICATE RECORDS
# -----------------------------------------

print("\n6. DUPLICATE RECORDS")
print("-" * 30)
print("Duplicates:", df.duplicated().sum())

# -----------------------------------------
# 6. STATISTICAL SUMMARY
# -----------------------------------------

print("\n7. STATISTICAL SUMMARY")
print("-" * 30)
print(df.describe().round(2).to_string())

# -----------------------------------------
# 7. FOOD ITEM ANALYSIS
# -----------------------------------------

print("\n8. FOOD ITEM SALES")
print("-" * 30)

item_sales = (
    df.groupby("Food_Item")["Quantity_Sold"]
    .sum()
    .sort_values(ascending=False)
)

print(item_sales)

# -----------------------------------------
# 8. AVERAGE DEMAND BY FOOD ITEM
# -----------------------------------------

print("\n9. AVERAGE DEMAND BY FOOD ITEM")
print("-" * 30)

avg_demand = (
    df.groupby("Food_Item")["Quantity_Sold"]
    .mean()
    .sort_values(ascending=False)
)

print(avg_demand.round(2))

# -----------------------------------------
# 9. REVENUE BY FOOD ITEM
# -----------------------------------------

print("\n10. REVENUE BY FOOD ITEM")
print("-" * 30)

revenue_by_item = (
    df.groupby("Food_Item")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print(revenue_by_item.round(2))

# -----------------------------------------
# 10. WEATHER ANALYSIS
# -----------------------------------------

print("\n11. AVERAGE DEMAND BY WEATHER")
print("-" * 30)

weather_demand = (
    df.groupby("Weather")["Quantity_Sold"]
    .mean()
    .sort_values(ascending=False)
)

print(weather_demand.round(2))

# -----------------------------------------
# 11. EXAM DAY ANALYSIS
# -----------------------------------------

print("\n12. EXAM DAY VS NORMAL DAY")
print("-" * 30)

exam_demand = (
    df.groupby("Exam_Day")["Quantity_Sold"]
    .mean()
)

print(exam_demand.round(2))

# -----------------------------------------
# 12. HOLIDAY ANALYSIS
# -----------------------------------------

print("\n13. HOLIDAY VS NORMAL DAY")
print("-" * 30)

holiday_demand = (
    df.groupby("Holiday")["Quantity_Sold"]
    .mean()
)

print(holiday_demand.round(2))

# -----------------------------------------
# 13. TOTAL WASTE
# -----------------------------------------

print("\n14. WASTE ANALYSIS")
print("-" * 30)

total_prepared = df["Quantity_Prepared"].sum()
total_sold = df["Quantity_Sold"].sum()
total_waste = df["Waste_Quantity"].sum()

print("Total Prepared :", total_prepared)
print("Total Sold     :", total_sold)
print("Total Waste    :", total_waste)

waste_percentage = (
    total_waste / total_prepared
) * 100

print("Waste %        :", round(waste_percentage, 2), "%")

# -----------------------------------------
# 14. TOTAL REVENUE
# -----------------------------------------

print("\n15. TOTAL REVENUE")
print("-" * 30)

print("Total Revenue: ₹", round(df["Revenue"].sum(), 2))

# -----------------------------------------
# 15. VISUALIZATION 1
# Food Item Demand
# -----------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    x=item_sales.index,
    y=item_sales.values
)

plt.title("Total Food Item Demand")
plt.xlabel("Food Item")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

# -----------------------------------------
# 16. VISUALIZATION 2
# Average Demand by Weather
# -----------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    x=weather_demand.index,
    y=weather_demand.values
)

plt.title("Average Food Demand by Weather")
plt.xlabel("Weather")
plt.ylabel("Average Quantity Sold")

plt.tight_layout()
plt.show()

# -----------------------------------------
# 17. VISUALIZATION 3
# Exam Day Analysis
# -----------------------------------------

exam_labels = ["Normal Day", "Exam Day"]

exam_values = [
    exam_demand.get(0, 0),
    exam_demand.get(1, 0)
]

plt.figure(figsize=(8, 5))

sns.barplot(
    x=exam_labels,
    y=exam_values
)

plt.title("Average Demand: Normal Day vs Exam Day")
plt.xlabel("Day Type")
plt.ylabel("Average Quantity Sold")

plt.tight_layout()
plt.show()

# -----------------------------------------
# 18. VISUALIZATION 4
# Students vs Quantity Sold
# -----------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Students",
    y="Quantity_Sold",
    hue="Food_Item"
)

plt.title("Students vs Food Demand")
plt.xlabel("Number of Students")
plt.ylabel("Quantity Sold")

plt.tight_layout()
plt.show()

# -----------------------------------------
# 19. VISUALIZATION 5
# Revenue by Food Item
# -----------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    x=revenue_by_item.index,
    y=revenue_by_item.values
)

plt.title("Total Revenue by Food Item")
plt.xlabel("Food Item")
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

# -----------------------------------------
# 20. VISUALIZATION 6
# Prepared vs Sold vs Waste
# -----------------------------------------

comparison = df[
    ["Quantity_Prepared", "Quantity_Sold", "Waste_Quantity"]
].sum()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=comparison.index,
    y=comparison.values
)

plt.title("Prepared vs Sold vs Waste")
plt.xlabel("Category")
plt.ylabel("Quantity")

plt.tight_layout()
plt.show()

# -----------------------------------------
# COMPLETE
# -----------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)