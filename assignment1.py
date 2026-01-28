# assignment1.py let's go!

import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

# getting data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url  = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

train = pd.read_csv(train_url)
test  = pd.read_csv(test_url)

# time series
train['Timestamp'] = pd.to_datetime(train['Timestamp'])
train.set_index('Timestamp', inplace=True)

test['Timestamp'] = pd.to_datetime(test['Timestamp'])
test.set_index('Timestamp', inplace=True)

# Target variable
y_train = train['trips'].asfreq('h')

# Defining model
model = ExponentialSmoothing(
    y_train,
    trend='add',
    damped_trend=True,
    seasonal='mul',
    seasonal_periods=168   # weekly seasonality
)

modelFit = model.fit(optimized=True)

# Forecast
pred = modelFit.forecast(steps=len(test))
pred = pd.Series(pred, index=test.index)

pred = pred.clip(lower=0)

print(pred.min())
print(pred.mean())
print(len(pred))
