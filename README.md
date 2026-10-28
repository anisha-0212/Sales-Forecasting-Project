# Sales Forecasting Using Machine Learning

## 1. Project Overview
This project uses historical sales data to predict future sales. It applies machine learning to identify trends in sales and generate forecasts for upcoming dates.

## 2. Objectives
- Analyze historical sales data.
- Clean and preprocess the dataset.
- Train a machine learning regression model.
- Evaluate model performance using MAE and RMSE.
- Visualize actual and predicted sales.
- Forecast sales for the next 30 days.

## 3. Dataset
The project uses `sales_sample_2024.csv`.

The dataset contains:
- `id` – record identifier
- `sales_date` – date of the sales record
- `sales_amount` – recorded sales amount

The dataset contains 366 daily records before preprocessing.

## 4. Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- NumPy
- PyCharm

## 5. Algorithm
Linear Regression is used to estimate sales based on date-related features:
- Day number
- Month
- Day of the week

The first 80% of the records are used for training, and the final 20% are used for testing in chronological order.

## 6. Evaluation Metrics
- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted sales.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving larger errors more weight.

Lower values generally indicate smaller prediction errors on the test data.

## 7. Project Workflow
1. Load the historical sales dataset.
2. Clean invalid or missing records.
3. Convert dates into a suitable format.
4. Create date-related features.
5. Train the Linear Regression model.
6. Evaluate predictions on historical test data.
7. Plot actual versus predicted sales.
8. Forecast sales for the next 30 days.
9. Save prediction results as CSV files.

## 8. Output Files
- `sales_predictions.csv` – actual and predicted values for the test period.
- `future_sales_forecast.csv` – forecasted sales for the next 30 days.

## 9. How to Run
1. Install the required Python libraries.
2. Place `sales_sample_2024.csv` in the project folder.
3. Run `sales_forecasting.py`.
4. View the evaluation metrics and graphs.
5. Open the generated CSV files to inspect predictions.

## 10. Limitations
- The model uses a simple regression approach.
- It uses date-related features but does not account for holidays, promotions, or external factors.
- Forecast accuracy depends on the quality and patterns of the sample dataset.
- Future predictions are estimates and may not reflect actual sales.

## 11. Future Enhancements
- Compare Linear Regression with other forecasting models.
- Add holiday and promotional information.
- Develop an interactive dashboard.
- Evaluate more advanced time-series forecasting methods.

## 12. Conclusion
This project demonstrates how machine learning can be applied to historical sales data to predict future sales. It includes data preprocessing, model training, performance evaluation, visualizations, and a 30-day forecast. The project provides a foundation for data-driven forecasting and sales planning.
