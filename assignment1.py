import pandas as pd
import numpy as np
from pygam import LinearGAM, s, f

train = pd.read_csv("assignment_data_train.csv")

# Target
y = train["trips"].values

# Features: hour of day, day of week, trend
X = pd.DataFrame({
    "hour": train.index % 24,                     # hourly seasonality
    "day_of_week": (train.index // 24) % 7,      # weekly seasonality
    "trend": np.arange(len(train))               # overall trend
})

model = LinearGAM(
    s(0, n_splines=24, spline_order=3) +   # hour
    f(1) +                                 # day of week as factor
    s(2, n_splines=50)                     # trend
).fit(X.values, y)

modelFit = model  # autograder expects modelFit

X_pred = pd.DataFrame({
    "hour": np.arange(len(train), len(train)+744) % 24,
    "day_of_week": (np.arange(len(train), len(train)+744)//24) % 7,
    "trend": np.arange(len(train), len(train)+744)
})

pred = modelFit.predict(X_pred.values)
