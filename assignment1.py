import pandas as pd
import numpy as np
from pygam import LinearGAM, s, f

# 1. Load the training data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
df_train = pd.read_csv(train_url)

# 2. Preprocess data
df_train['Timestamp'] = pd.to_datetime(df_train['Timestamp'])
df_train['day_of_week'] = df_train['Timestamp'].dt.dayofweek

X_train = df_train[['hour', 'day_of_week']]
y_train = df_train['trips']

# 3. Build and Fit the Model
model = LinearGAM(s(0, n_splines=24) + f(1))
modelFit = model.gridsearch(X_train.values, y_train.values)

# 4. Load the test data and generate predictions
test_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
df_test = pd.read_csv(test_url)

# Preprocess test data the same way as training data
df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'])
df_test['day_of_week'] = df_test['Timestamp'].dt.dayofweek
X_test = df_test[['hour', 'day_of_week']]

# Generate the 'pred' vector (744 hours for January)
pred = modelFit.predict(X_test.values)

# Ensure pred is a numpy array or list as expected by the autograder
pred = np.array(pred)
