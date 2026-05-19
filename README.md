# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date:19-05-26 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
data = pd.read_csv('/content/Chocolate Sales (2).csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
plt.plot(data['Date'], data['Boxes Shipped'])
plt.xlabel('Date')
plt.ylabel('Boxes Shipped')
plt.title('Chocolate sales Time Series')
plt.show()
def check_stationarity(timeseries):
  result = adfuller(timeseries)
  print('ADF Statistic:', result[0])
  print('p-value:', result[1])
  print('Critical Values:')
  for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
check_stationarity(data['Boxes Shipped'])
plot_acf(data['Boxes Shipped'])
plt.show()
plot_pacf(data['Boxes Shipped'])
plt.show()
train_size = int(len(data) * 0.8)
train, test = data['Boxes Shipped'][:train_size], data['Boxes Shipped'][train_size:]
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
predictions = sarima_result.predict(start=len(train), end=len(data) - 1)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Boxes Shipped')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```

### OUTPUT:
<img width="578" height="455" alt="download" src="https://github.com/user-attachments/assets/571c11ee-b785-4233-ab7c-bf681819f775" />

<img width="568" height="435" alt="download" src="https://github.com/user-attachments/assets/1b8aa2f0-faa3-44c5-8315-9d487d2c1ea8" />

<img width="568" height="435" alt="download" src="https://github.com/user-attachments/assets/5e523c0b-2461-4eb7-bbb8-f114a476ed5c" />

<img width="579" height="455" alt="download" src="https://github.com/user-attachments/assets/8df9a420-ad64-4b80-b0a4-ff70ffc70298" />



### RESULT:
Thus the program run successfully based on the SARIMA model.
