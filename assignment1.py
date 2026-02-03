import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('data.csv')  # replace with your actual filename
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['y'].astype(float)

# Linear trend
df['t'] = np.arange(len(df))

# Weekly seasonality
df['dow'] = df['ds'].dt.dayofweek  # 0=Mon, 6=Sun
dow_dummies = pd.get_dummies(df['dow'], prefix='dow', drop_first=True)

# Combine features
X = pd.concat([df[['t']], dow_dummies], axis=1)
X = sm.add_constant(X)  # add intercept

train_X = X.iloc[:-30]
train_y = df['y'].iloc[:-30]
test_X = X.iloc[-30:]
test_y = df['y'].iloc[-30:]

model = sm.OLS(train_y, train_X).fit()

y_pred = model.predict(test_X)

rmse = np.sqrt(np.mean((test_y - y_pred)**2))
print(f'RMSE: {rmse}')

forecast = y_pred.values
