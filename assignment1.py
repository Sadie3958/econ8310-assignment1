# assignment1.py attemmp #10

import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing

# Load data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url  = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

train = pd.read_csv(train_url)
test  = pd.read_csv(test_url)

# Datetime handling
train["Timestamp"] = pd.to_datetime(train["Timestamp"])
test["Timestamp"]  = pd.to_datetime(test["Timestamp"])

train = train.set_index("Timestamp")
test  = test.set_index("Timestamp")

# Target series (hourly)
y = train["trips"].astype(float)
y = y.asfreq("h")
y = y.interpolate(method="time")

# LOG TRANSFORM (critical)
y_log = np.log1p(y)

# Model: Holt-Winters (tuned)
model = ExponentialSmoothing(
    y_log,
    trend=None,
    seasonal="mul",
    seasonal_periods=24
)

modelFit = model.fit(
    optimized=True,
    use_brute=True
)

# Forecast
pred_log = modelFit.forecast(len(test))

# Back-transform
pred = np.expm1(pred_log)

# Match test index exactly
pred = pd.Series(pred.values, index=test.index)

# Safety
pred = pred.clip(lower=0)
pred = pred.fillna(method="ffill").fillna(method="bfill")
