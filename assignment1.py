import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

train = pd.read_csv("assignment_data_train.csv")

# Target variable
y = train["trips"]

model = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="add",
    seasonal_periods=24
)

modelFit = model.fit()

pred = modelFit.forecast(744)
