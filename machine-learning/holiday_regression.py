import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
x_train = pd.read_csv('x_train_sncf.csv')
x_test_submit = pd.read_csv('x_test_sncf.csv')
y_train = pd.read_csv('y_train_sncf.csv')

############################################

# Preprocess the data
# For example, parse the dates and convert them to a numerical value, such as the number of days since a certain date
x_train['date'] = pd.to_datetime(x_train['date'])
x_train['days_since'] = (x_train['date'] - x_train['date'].min()).dt.days
X_station = x_train['station']
station_mapping = {station: i for i, station in enumerate(X_station.unique())}
x_train['station_id'] = x_train['station'].map(station_mapping)
X_train_IDS= x_train[['days_since','station_id', 'job', 'ferie', 'vacances']]
X_train_IDS_ARRAY = X_train_IDS.to_numpy()

Y_IDS = y_train['y']
Y_IDS_ARRAY = Y_IDS.to_numpy()

#############################################

x_test_submit['date'] = pd.to_datetime(x_test_submit['date'])
x_test_submit['days_since'] = (x_test_submit['date'] - x_test_submit['date'].min()).dt.days
X_station = x_test_submit['station']
station_mapping = {station: i for i, station in enumerate(X_station.unique())}
x_test_submit['station_id'] = x_test_submit['station'].map(station_mapping)
X_test_submit_IDS= x_test_submit[['days_since','station_id', 'job', 'ferie', 'vacances']]
X_test_submit_IDS_ARRAY = X_test_submit_IDS.to_numpy()

# Create the index to use for y_test
# Extract from x_test_submit the date and the name of the station 
data_station = x_test_submit['date'].astype(str) + '_' + x_test_submit['station']
data_station_array = np.array(data_station)

#########################################################"
X_train_IDS_ARRAY_job_1 = X_train_IDS_ARRAY[X_train_IDS_ARRAY[:, 4] == 1]
X_train_IDS_ARRAY_job_0 = X_train_IDS_ARRAY[X_train_IDS_ARRAY[:, 4] == 0]

# Associate the y to the new X_train
Y_train_IDS_ARRAY_job_1 = Y_IDS_ARRAY[X_train_IDS_ARRAY[:, 4] == 1]
Y_train_IDS_ARRAY_job_0 = Y_IDS_ARRAY[X_train_IDS_ARRAY[:, 4] == 0]

######################
## LINEAR REGRESSION ##
######################

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_IDS_ARRAY, Y_IDS_ARRAY, test_size=0.3, random_state=42)

# Train the linear regression model with the new data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the training data
y_pred = model.predict(X_test)

# Evaluate the model
epsilon = 1  # Small epsilon value to avoid division by zero
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

# Calculate MAPE excluding zero values in y_test
non_zero_indices = np.nonzero(y_test)
mape_non_zero = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / (y_test[non_zero_indices] + epsilon))) * 100

print(f'MAPE (excluding zero values): {mape_non_zero}%')

print(f'MAPE: {mape}%')

######################

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_IDS_ARRAY_job_1, Y_train_IDS_ARRAY_job_1, test_size=0.3, random_state=42)

# Train the linear regression model with the new data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the training data
y_pred = model.predict(X_test)

# Evaluate the model
epsilon = 1  # Small epsilon value to avoid division by zero
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

# Calculate MAPE excluding zero values in y_test
non_zero_indices = np.nonzero(y_test)
mape_non_zero = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / (y_test[non_zero_indices] + epsilon))) * 100

print(f'MAPE (excluding zero values): {mape_non_zero}%')

print(f'MAPE: {mape}%')

######################

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_IDS_ARRAY_job_0, Y_train_IDS_ARRAY_job_0, test_size=0.3, random_state=42)

# Train the linear regression model with the new data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the training data
y_pred = model.predict(X_test)

# Evaluate the model
epsilon = 1  # Small epsilon value to avoid division by zero
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

# Calculate MAPE excluding zero values in y_test
non_zero_indices = np.nonzero(y_test)
mape_non_zero = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / (y_test[non_zero_indices] + epsilon))) * 100

print(f'MAPE (excluding zero values): {mape_non_zero}%')

print(f'MAPE: {mape}%')

# ############################################

from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X_train_IDS_ARRAY, Y_IDS_ARRAY, test_size=0.3, random_state=42)
model = RandomForestRegressor(n_estimators=5)
model.fit(X_train, y_train)

# Predict on the training data
y_pred = model.predict(X_test)
print(y_pred.shape)
print(y_test.shape)

# Evaluate the model
epsilon = 1  # Small epsilon value to avoid division by zero
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

# Calculate MAPE excluding zero values in y_test
non_zero_indices = np.nonzero(y_test)
mape_non_zero = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / (y_test[non_zero_indices] + epsilon))) * 100

print(f'MAPE (excluding zero values): {mape_non_zero}%')

print(f'MAPE: {mape}%')

# Train the random forest regression model with the new data
X_train, X_test, y_train, y_test = train_test_split(X_train_IDS_ARRAY_job_1, Y_train_IDS_ARRAY_job_1, test_size=0.3, random_state=42)
model = RandomForestRegressor(n_estimators=5)
model.fit(X_train, y_train)

# Predict on the training data
y_pred = model.predict(X_test)
print(y_pred.shape)
print(y_test.shape)

# Evaluate the model
epsilon = 1  # Small epsilon value to avoid division by zero
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

# Calculate MAPE excluding zero values in y_test
non_zero_indices = np.nonzero(y_test)
mape_non_zero = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / (y_test[non_zero_indices] + epsilon))) * 100

print(f'MAPE (excluding zero values): {mape_non_zero}%')

print(f'MAPE: {mape}%')

# Train the random forest regression model with the new data
X_train, X_test, y_train, y_test = train_test_split(X_train_IDS_ARRAY_job_0, Y_train_IDS_ARRAY_job_0, test_size=0.3, random_state=42)
model = RandomForestRegressor(n_estimators=5)
model.fit(X_train, y_train)

# Predict on the training data
y_pred = model.predict(X_test)
print(y_pred.shape)
print(y_test.shape)

# Evaluate the model
epsilon = 1  # Small epsilon value to avoid division by zero
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

# Calculate MAPE excluding zero values in y_test
non_zero_indices = np.nonzero(y_test)
mape_non_zero = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / (y_test[non_zero_indices] + epsilon))) * 100

print(f'MAPE (excluding zero values): {mape_non_zero}%')

print(f'MAPE: {mape}%')


