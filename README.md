# Ex.No: 6 HOLT WINTERS METHOD
### Date: 18/05/2026



### AIM:
To implement and evaluate the Holt-Winters Exponential Smoothing method for time series forecasting using the Air Passengers dataset.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
    
### PROGRAM:
```py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv('/content/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
display(data.head())

data_monthly = data.resample('MS').sum()
display(data_monthly.head())

data_monthly.plot(figsize=(12, 6))
plt.title('Monthly Air Passengers Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.grid(True)
plt.show()

scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten())

scaled_data.plot(figsize=(12, 6))
plt.title('Scaled Monthly Air Passengers Data')
plt.xlabel('Time')
plt.ylabel('Scaled Passengers')
plt.grid(True)
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data_monthly, model="additive", period=12)

decomposition.plot()
plt.show()

scaled_data = scaled_data + 1

train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

plt.figure(figsize=(12, 6))
ax = train_data.plot(label='Train Data')
test_data.plot(ax=ax, label='Test Data')
test_predictions_add.plot(ax=ax, label='Test Predictions (Additive Trend, Multiplicative Seasonal)')
ax.legend()
ax.set_title('Visual Evaluation: Holt-Winters Predictions vs Actuals')
plt.xlabel('Time')
plt.ylabel('Scaled Passengers')
plt.grid(True)
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
mae = mean_absolute_error(test_data, test_predictions_add)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse:.4f}")
print(f"Mean Absolute Error (MAE) on test data: {mae:.4f}")

print(f"Scaled data variance: {np.sqrt(scaled_data.var()):.4f}, Scaled data mean: {scaled_data.mean():.4f}")

final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()

final_predictions = final_model.forecast(steps=12)

plt.figure(figsize=(12, 6))
ax = data_monthly.plot(label='Historical Data')
final_predictions.plot(ax=ax, label='Final Predictions')
ax.legend()
ax.set_xlabel('Months')
ax.set_ylabel('Number of monthly passengers')
ax.set_title('Prediction: Monthly Air Passengers')
plt.grid(True)
plt.show()
```

### OUTPUT:
<img width="600" height="450" alt="Screenshot 2026-05-18 092714" src="https://github.com/user-attachments/assets/a2e657e3-e1f1-46b2-83e0-fb0ab8a9662c" />

<img width="1000" height="500" alt="Screenshot 2026-05-18 092707" src="https://github.com/user-attachments/assets/87ed24cf-5dec-4ade-aa35-7b9c84daf1fd" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
