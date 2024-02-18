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
# x_train = x_train.sort_values('date')
# Convert the 'date' column to datetime
x_train['date'] = pd.to_datetime(x_train['date'])

# Create a new column with the numerical day of the year
x_train['day_of_year'] = x_train['date'].dt.dayofyear

# x_train['date'] = pd.to_datetime(x_train['date'])
x_train['days_since'] = (x_train['date'] - x_train['date'].min()).dt.days
# x_train['day_in_year'] = x_train['date'].dt.dayofyear
X_station = x_train['station']
station_mapping = {station: i for i, station in enumerate(X_station.unique())}
x_train['station_id'] = x_train['station'].map(station_mapping)
X_train_IDS= x_train[['day_of_year','station_id', 'job', 'ferie', 'vacances']]
X_train_IDS_ARRAY = X_train_IDS.to_numpy()
print(X_train_IDS.iloc[1500:1505])


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


# ############################################

from sklearn.ensemble import RandomForestRegressor

# Train the random forest regression model with the new data
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


# # create a y_test_csv file
# y_test_submit = model.predict(X_test_submit_IDS_ARRAY)
# print(y_test_submit[:5])
# result_df = pd.DataFrame({'data_station': data_station_array, 'y_test_submit': y_test_submit})
# print(result_df.head())
# result_df.rename(columns={'data_station': 'index', 'y_test_submit': 'y'}, inplace=True)
# result_df.to_csv('y_test_sncf.csv', index=False)


# ############################################
# from sklearn.svm import SVR

# # Train the SVM regression model with the new data
# svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
# svm_model.fit(X_train, y_train)

# # Predict on the training data
# y_pred_svm = svm_model.predict(X_test)
# print(y_pred_svm.shape)
# print(y_test.shape)

# # Evaluate the model
# epsilon = 1  # Small epsilon value to avoid division by zero
# mape_svm = np.mean(np.abs((y_test - y_pred_svm) / (y_test + epsilon))) * 100

# # Calculate MAPE excluding zero values in y_test
# non_zero_indices_svm = np.nonzero(y_test)
# mape_non_zero_svm = np.mean(np.abs((y_test[non_zero_indices_svm] - y_pred_svm[non_zero_indices_svm]) / (y_test[non_zero_indices_svm] + epsilon))) * 100

# print(f'MAPE (excluding zero values) for SVM: {mape_non_zero_svm}%')
# print(f'MAPE for SVM: {mape_svm}%')


