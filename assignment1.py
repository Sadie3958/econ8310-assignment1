# assignment1.py

import pandas as pd
import numpy as np
from pygam import LinearGAM, s

# -------------------------
# Load data
# -------------------------
train = pd.read_csv("assignment_data_train.csv")
test = pd.read_csv("assignment_data_test.csv")

# -------------------------
# Select features and target
# ONLY use hour, day, month
# -------------------------
X_train = train[["hour", "day", "month"]]
y_train = train["trips"]

X_test = test[["hour", "day", "month"]]

# -------------------------
# Fit GAM
# -------------------------
# Smooth terms for each calendar feature
gam = LinearGAM(
    s(0, n_splines=10) +  # hour
    s(1, n_splines=10) +  # day
    s(2, n_splines=10)    # month
)

gam.fit(X_train, y_train)

# -------------------------
# Predict
# -------------------------
predictions = gam.predict(X_test)

# Ensure no negative trip predictions
predictions = np.maximum(predictions, 0)

# -------------------------
# Output
# -------------------------
# The autograder expects ONLY predictions, one per line
for p in predictions:
    print(p)
