# assignment1.py attempt #10

import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing

# data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url  = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

train = pd.read_csv(train_url)
test  = pd.read_csv(test_url)

# timestamps
train["Timestamp"] = pd.to_datetime(train["Timestamp"])
test["Timestamp"]  = pd.to_datetime(test["Timestamp"])

train = train.set_index("Timestamp")
test  = test.set_index("Timestamp")

# series
y = train["trips"].astype(float)
y = y.asfreq("h")

# time
y = y.interpolate(method="time")

# Model HOLT
model = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="mul",
    seasonal_periods=24,
    damped_trend=True
)

fit = model.fit(
    optimized=True,
    use_brute=True
)

# Forecast
pred = fit.forecast(len(test))

# Match test index exactly
pred = pd.Series(pred.values, index=test.index)

# Guardrails
pred = pred.clip(lower=0)
pred = pred.fillna(method="ffill").fillna(method="bfill")
