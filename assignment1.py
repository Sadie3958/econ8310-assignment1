import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing

# data
train = pd.read_csv("assignment_data_train.csv")

# Ensure proper datetime ordering
train["Timestamp"] = pd.to_datetime(train["Timestamp"])
train = train.sort_values("Timestamp")

# Extract target series
y = train["trips"].astype(float)

# Holt model
model = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="add",
    seasonal_periods=24
)

modelFit = model.fit(optimized=True)

# test data load here
test = pd.read_csv("assignment_data_test.csv")

test["Timestamp"] = pd.to_datetime(test["Timestamp"])
test = test.sort_values("Timestamp")

n_forecast = len(test)

# predict
pred = modelFit.forecast(n_forecast)

# Ensure numpy array output
pred = np.array(pred)
