# Required imports
import pandas as pd
import numpy as np
from prophet import Prophet

df = pd.read_csv('your_data.csv')  

# Convert to correct types
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['y'].astype(float)

train_df = df.iloc[:-30]
test_df = df.iloc[-30:]

model = Prophet(
    daily_seasonality=False,      # turn off if daily data is noisy
    weekly_seasonality=True,      # captures weekly trends
    yearly_seasonality=True       # captures yearly seasonality
)

# Add US holidays to improve accuracy
model.add_country_holidays(country_name='US')

# Fit the model
model.fit(train_df)

future = model.make_future_dataframe(periods=len(test_df), freq='D')
forecast = model.predict(future)

y_pred = forecast['yhat'][-len(test_df):].values
y_true = test_df['y'].values

rmse = np.sqrt(np.mean((y_true - y_pred)**2))
print(f'RMSE: {rmse}')

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(df['ds'], df['y'], label='Actual')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
plt.axvline(x=train_df['ds'].iloc[-1], color='r', linestyle='--', label='Train/Test split')
plt.legend()
plt.show()
