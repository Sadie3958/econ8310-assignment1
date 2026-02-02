# assignment1.py

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load training and test data
train = pd.read_csv("assignment_data_train.csv")
test = pd.read_csv("assignment_data_test.csv")

# Prepare time series
train["Timestamp"] = pd.to_datetime(train["Timestamp"])
train = train.sort_values("Timestamp")
train.set_index("Timestamp", inplace=True)

# Hourly frequency
y = train["trips"].asfreq("H")

# Define model 
model = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="add",
    seasonal_periods=24
)

# Fit model
modelFit = model.fit(optimized=True)

# Forecast January
pred = modelFit.forecast(steps=len(test))

# Ensure numeric & non-negative
pred = np.maximum(np.array(pred, dtype=float), 0)
