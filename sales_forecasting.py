
import pandas as pd
from pathlib import Path

# Load the sales dataset
file_path = Path(__file__).parent / "sales_sample_2024.csv"
df = pd.read_csv(file_path)

# Display the first five rows
print("FIRST FIVE ROWS:")
print(df.head())

# Display dataset information
print("\nDATASET INFORMATION:")
print(df.info())

# Check for missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Display the number of rows and columns
print("\nDATASET SHAPE:")
print(df.shape)


import pandas as pd
from pathlib import Path

# Step 1: Load the dataset
file_path = Path(__file__).parent / "sales_sample_2024.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Step 2: Convert sales_date into a proper date format
df["sales_date"] = pd.to_datetime(df["sales_date"], errors="coerce")

# Step 3: Remove rows with missing or invalid dates and sales amounts
df = df.dropna(subset=["sales_date", "sales_amount"])

# Step 4: Remove duplicate records
df = df.drop_duplicates()

# Step 5: Sort data by date
df = df.sort_values("sales_date").reset_index(drop=True)

# Step 6: Create numerical date features for machine learning
df["day_number"] = (
    df["sales_date"] - df["sales_date"].min()
).dt.days

df["month"] = df["sales_date"].dt.month
df["day_of_week"] = df["sales_date"].dt.day_of_week

# Step 7: Display the cleaned dataset
print("\nCLEANED DATASET:")
print(df.head())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFinal dataset shape:", df.shape)
print("\nData preprocessing completed successfully!")


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Step 1: Load the dataset
file_path = Path(__file__).parent / "sales_sample_2024.csv"
df = pd.read_csv(file_path)

# Step 2: Clean and prepare the data
df["sales_date"] = pd.to_datetime(df["sales_date"], errors="coerce")
df["sales_amount"] = pd.to_numeric(df["sales_amount"], errors="coerce")
df = df.dropna(subset=["sales_date", "sales_amount"])
df = df.drop_duplicates()
df = df.sort_values("sales_date").reset_index(drop=True)

# Step 3: Create date features
df["day_number"] = (
    df["sales_date"] - df["sales_date"].min()
).dt.days
df["month"] = df["sales_date"].dt.month
df["day_of_week"] = df["sales_date"].dt.day_of_week

print("Dataset prepared successfully!")
print("Number of records:", len(df))

# Step 4: Select features and target
features = ["day_number", "month", "day_of_week"]
X = df[features]
y = df["sales_amount"]

# Step 5: Split chronologically: first 80% for training
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Step 6: Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict sales for the test period
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nMODEL EVALUATION")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

# Step 9: Compare actual and predicted sales
results = pd.DataFrame({
    "Date": df["sales_date"].iloc[split_index:].values,
    "Actual_Sales": y_test.values,
    "Predicted_Sales": y_pred.round(2)
})

print("\nACTUAL VS PREDICTED SALES")
print(results.head(10))

# Step 10: Save predictions
output_path = Path(__file__).parent / "sales_predictions.csv"
results.to_csv(output_path, index=False)
print("\nPredictions saved to:", output_path)

# Step 11: Plot actual vs predicted sales
plt.figure(figsize=(12, 6))
plt.plot(results["Date"], results["Actual_Sales"],
         label="Actual Sales")
plt.plot(results["Date"], results["Predicted_Sales"],
         label="Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Step 12: Forecast sales for the next 30 days

last_date = df["sales_date"].max()
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future_df = pd.DataFrame({
    "sales_date": future_dates
})

# Create the same features used to train the model
future_df["day_number"] = (
    future_df["sales_date"] - df["sales_date"].min()
).dt.days

future_df["month"] = future_df["sales_date"].dt.month
future_df["day_of_week"] = (
    future_df["sales_date"].dt.day_of_week
)

# Predict future sales
future_df["Predicted_Sales"] = model.predict(
    future_df[["day_number", "month", "day_of_week"]]
).round(2)

print("\nNEXT 30 DAYS SALES FORECAST")
print(future_df[["sales_date", "Predicted_Sales"]])

# Save future predictions
future_output = Path(__file__).parent / "future_sales_forecast.csv"
future_df.to_csv(future_output, index=False)

print("\nFuture forecast saved to:", future_output)

# Plot historical sales and future forecast
plt.figure(figsize=(12, 6))
plt.plot(
    df["sales_date"],
    df["sales_amount"],
    label="Historical Sales"
)
plt.plot(
    future_df["sales_date"],
    future_df["Predicted_Sales"],
    label="Next 30 Days Forecast",
    linestyle="--"
)
plt.title("Historical Sales and Future Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
