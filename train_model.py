import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("canteen_data.csv")

print("=" * 70)
print("SMARTCANTEEN AI - MACHINE LEARNING MODEL TRAINING")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "Day",
    "Weather",
    "Temperature",
    "Rainfall_mm",
    "Exam_Day",
    "Holiday",
    "Students",
    "Food_Item",
    "Price"
]

target = "Quantity_Sold"

X = df[features]
y = df[target]


# ============================================================
# 3. DEFINE COLUMNS
# ============================================================

categorical_features = [
    "Day",
    "Weather",
    "Food_Item"
]

numeric_features = [
    "Temperature",
    "Rainfall_mm",
    "Exam_Day",
    "Holiday",
    "Students",
    "Price"
]


# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records :", len(X_train))
print("Testing Records  :", len(X_test))


# ============================================================
# 6. DEFINE MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=12,
        min_samples_leaf=2
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=15,
        min_samples_leaf=2,
        n_jobs=-1
    )
}


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 70)
    print("Training:", model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2
    })

    trained_models[model_name] = pipeline

    print("MAE      :", round(mae, 2))
    print("RMSE     :", round(rmse, 2))
    print("R² Score :", round(r2, 4))


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Selected Model:", best_model_name)

print(
    "Reason: Lowest Mean Absolute Error (MAE)"
)


# ============================================================
# 10. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "demand_model.pkl"
)

print("\nModel saved as:")
print("demand_model.pkl")


# ============================================================
# 11. ACTUAL VS PREDICTED
# ============================================================

best_predictions = best_model.predict(
    X_test
)

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.5
)

plt.xlabel("Actual Quantity Sold")
plt.ylabel("Predicted Quantity Sold")

plt.title(
    f"Actual vs Predicted Demand - {best_model_name}"
)

# Perfect prediction reference line
min_value = min(
    y_test.min(),
    best_predictions.min()
)

max_value = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.tight_layout()
plt.show()


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNext Step:")
print("Build the Streamlit SmartCanteen AI dashboard.")