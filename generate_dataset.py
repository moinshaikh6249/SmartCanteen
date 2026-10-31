import pandas as pd
import numpy as np

# Reproducible results
np.random.seed(42)

# -----------------------------
# SETTINGS
# -----------------------------
start_date = "2025-07-01"
end_date = "2026-06-30"

food_items = {
    "Samosa": 15,
    "Vada Pav": 20,
    "Sandwich": 40,
    "Tea": 10,
    "Cold Drink": 25,
    "Maggi": 35
}

weather_options = ["Sunny", "Cloudy", "Rainy"]

records = []

dates = pd.date_range(start=start_date, end=end_date, freq="D")

# -----------------------------
# DATA GENERATION
# -----------------------------
for date in dates:

    day = date.strftime("%A")
    month = date.month

    # Weekend effect
    is_weekend = day in ["Saturday", "Sunday"]

    # Simulated college holiday
    is_holiday = np.random.choice(
        [0, 1],
        p=[0.90, 0.10]
    )

    # Simulated exam day
    is_exam_day = np.random.choice(
        [0, 1],
        p=[0.78, 0.22]
    )

    # Weather
    weather = np.random.choice(
        weather_options,
        p=[0.45, 0.30, 0.25]
    )

    # Temperature based on weather
    if weather == "Sunny":
        temperature = np.random.randint(28, 36)
    elif weather == "Cloudy":
        temperature = np.random.randint(25, 32)
    else:
        temperature = np.random.randint(22, 29)

    # Rainfall
    if weather == "Rainy":
        rainfall = round(np.random.uniform(5, 60), 1)
    elif weather == "Cloudy":
        rainfall = round(np.random.uniform(0, 8), 1)
    else:
        rainfall = 0.0

    # Expected students
    if is_holiday:
        students = np.random.randint(50, 180)
    elif is_weekend:
        students = np.random.randint(250, 500)
    else:
        students = np.random.randint(650, 950)

    # Exam days may increase attendance
    if is_exam_day and not is_holiday:
        students += np.random.randint(30, 100)

    students = min(students, 1000)

    # -----------------------------
    # FOOD ITEM DATA
    # -----------------------------
    for item, price in food_items.items():

        # Base demand for each item
        base_demand = {
            "Samosa": 110,
            "Vada Pav": 100,
            "Sandwich": 65,
            "Tea": 230,
            "Cold Drink": 85,
            "Maggi": 70
        }[item]

        demand = base_demand

        # Student footfall effect
        demand *= students / 800

        # Weekend effect
        if is_weekend:
            demand *= 0.65

        # Holiday effect
        if is_holiday:
            demand *= 0.30

        # Exam effect
        if is_exam_day:
            if item in ["Tea", "Samosa", "Vada Pav", "Maggi"]:
                demand *= 1.15
            else:
                demand *= 1.05

        # Weather effects
        if weather == "Rainy":
            if item in ["Tea", "Maggi", "Samosa"]:
                demand *= 1.20
            elif item == "Cold Drink":
                demand *= 0.75

        elif weather == "Sunny":
            if item == "Cold Drink":
                demand *= 1.25
            elif item == "Tea":
                demand *= 0.90

        # Small monthly variation
        seasonal_factor = 1 + (np.sin(month / 12 * 2 * np.pi) * 0.05)
        demand *= seasonal_factor

        # Random variation
        demand += np.random.normal(0, 8)

        # Ensure positive integer
        quantity_sold = max(0, int(round(demand)))

        # Preparation quantity
        # Canteen normally prepares slightly more than expected demand
        extra_stock = np.random.uniform(0.05, 0.15)

        quantity_prepared = int(
            round(quantity_sold * (1 + extra_stock))
        )

        # Waste
        waste_quantity = max(
            0,
            quantity_prepared - quantity_sold
        )

        # Revenue
        revenue = quantity_sold * price

        records.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Day": day,
            "Weather": weather,
            "Temperature": temperature,
            "Rainfall_mm": rainfall,
            "Exam_Day": is_exam_day,
            "Holiday": is_holiday,
            "Students": students,
            "Food_Item": item,
            "Price": price,
            "Quantity_Prepared": quantity_prepared,
            "Quantity_Sold": quantity_sold,
            "Waste_Quantity": waste_quantity,
            "Revenue": revenue
        })

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame(records)

# Shuffle rows
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save CSV
df.to_csv(
    "canteen_data.csv",
    index=False
)

# -----------------------------
# INFORMATION
# -----------------------------
print("=" * 50)
print("SMARTCANTEEN DATASET CREATED")
print("=" * 50)

print(f"Total Records : {len(df)}")
print(f"Total Columns : {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 Records:")
print(df.head(10).to_string(index=False))

print("\nDataset saved as:")
print("canteen_data.csv")

print("=" * 50)