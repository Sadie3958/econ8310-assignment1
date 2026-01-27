# assignment1.py

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

# Target variable (hourly frequency)
y_train = train['trips'].asfreq('h')

# model
model = ExponentialSmoothing(
    y_train,
    trend='add',
    seasonal='add',
    seasonal_periods=24
)

modelFit = model.fit()

# forecast
pred = modelFit.forecast(steps=len(test))
pred = pd.Series(pred, index=test.index)

# Prevent negative trip counts in case this matters
pred = pred.clip(lower=0)
