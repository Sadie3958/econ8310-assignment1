# assignment1.py attempt #7

import pandas as pd
import numpy as np
from pygam import LinearGAM, s

# Get data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url  = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

train = pd.read_csv(train_url)
test  = pd.read_csv(test_url)

# Preparing timestamps
train['Timestamp'] = pd.to_datetime(train['Timestamp'])
test['Timestamp'] = pd.to_datetime(test['Timestamp'])

# Feature engineering (time-based)
for df in [train, test]:
    df['hour'] = df['hour']
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    df['time_idx'] = (df['Timestamp'] - train['Timestamp'].min()).dt.total_seconds() / 3600

# Defining X and y variables
X_train = train[['hour', 'dayofweek', 'time_idx']].values
y_train = train['trips'].values

X_test = test[['hour', 'dayofweek', 'time_idx']].values

# GAM model
model = LinearGAM(
    s(0, n_splines=24) +        # hour of day
    s(1, n_splines=7) +         # day of week
    s(2, n_splines=40)          # long-run trend
)

modelFit = model.gridsearch(X_train, y_train)

# Forecast
pred = modelFit.predict(X_test)

# no negative trips
pred = np.clip(pred, 0, None)
