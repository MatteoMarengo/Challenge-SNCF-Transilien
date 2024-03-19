#############################################
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

############################################
# Load the data
x_train = pd.read_csv('x_train_sncf.csv')
x_train = pd.read_csv('train_f_x.csv')
x_test_submit = pd.read_csv('x_test_sncf.csv')
y_train = pd.read_csv('y_train_sncf.csv')

############################################

# Preprocess the data
# For example, parse the dates and convert them to a numerical value, such as the number of days since a certain date
# x_train = x_train.sort_values('date')
# Convert the 'date' column to datetime
# x_train = x_train.sort_values('date')
print(x_train['station'].nunique())
print(x_train['date'].nunique())
print(f"Total number of lines: {len(x_train)}")

station_counts = x_train['station'].value_counts()
top_10_stations = station_counts.nlargest(10)
print(top_10_stations)
# print the top 10 smallest
bottom_10_stations = station_counts.nsmallest(10)
print(bottom_10_stations)

print(x_train.head())

x_train['date'] = pd.to_datetime(x_train['date'])
print(x_train['date'].head())
x_train['day_of_week'] = x_train['date'].dt.dayofweek + 1
print(x_train['day_of_week'].head())
print(x_train['day_of_week'].value_counts())
x_train['day_of_week_cos'] = np.cos(2 * np.pi * (x_train['day_of_week'] - 1) / 7)
print(x_train['day_of_week_cos'].head())
x_train['day_of_week_sin'] = np.sin(2 * np.pi * (x_train['day_of_week'] - 1) / 7)

# print me the value for each day of the week
print(x_train['day_of_week_cos'].value_counts())
# Create a new column with the numerical day of the year
x_train['day_of_year'] = x_train['date'].dt.dayofyear

# x_train['date'] = pd.to_datetime(x_train['date'])
x_train['days_since'] = (x_train['date'] - x_train['date'].min()).dt.days
# x_train['day_in_year'] = x_train['date'].dt.dayofyear
X_station = x_train['station']
station_mapping = {station: i for i, station in enumerate(X_station.unique())}
x_train['station_id'] = x_train['station'].map(station_mapping)
X_train_IDS= x_train[['day_of_week_cos','day_of_week_sin','station_id', 'job', 'ferie', 'vacances']]
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


# ######################
# ## LINEAR REGRESSION ##
# ######################

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_IDS_ARRAY, Y_IDS_ARRAY, test_size=0.3, random_state=42)

import torch
import torch.nn as nn
import torch.optim as optim

# Define the deep learning model
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = RegressionModel(X_train.shape[1])

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Train the model
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.unsqueeze(1))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Predict on the training data
batch_size = 1000  # Set the batch size
num_samples = X_test_tensor.shape[0]  # Get the number of samples
num_batches = num_samples // batch_size  # Calculate the number of batches

mape_sum = 0  # Initialize the sum of MAPE values

for i in range(num_batches):
    start_idx = i * batch_size  # Calculate the start index of the batch
    end_idx = (i + 1) * batch_size  # Calculate the end index of the batch

    # Get the batch of data
    X_batch = X_test_tensor[start_idx:end_idx]
    y_batch = y_test_tensor[start_idx:end_idx]

    # Predict on the batch
    y_pred_batch_tensor = model(X_batch)
    y_pred_batch = y_pred_batch_tensor.detach().numpy()

    # Remove negative predictions
    y_pred_batch = np.maximum(y_pred_batch, 0)

    # Round and convert to integer
    y_pred_batch = np.round(y_pred_batch).astype(int)

    # Calculate MAPE for the batch
    mape_batch = np.mean(np.abs((y_batch - y_pred_batch) / y_batch)) * 100

    # Add the batch's MAPE to the sum
    mape_sum += mape_batch

# Calculate the average MAPE
mape_fake = mape_sum / num_batches

print(f'MAPE_fake: {mape_fake}%')


# # Train the linear regression model with the new data
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict on the training data
# y_pred = model.predict(X_test)

# # Evaluate the model
# epsilon = 1  # Small epsilon value to avoid division by zero
# mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

# # Calculate MAPE excluding zero values in y_test
# non_zero_indices = np.nonzero(y_test)
# mape_non_zero = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / (y_test[non_zero_indices] + epsilon))) * 100

# print(f'MAPE (excluding zero values): {mape_non_zero}%')

# print(f'MAPE: {mape}%')


# ############################################

# from sklearn.ensemble import RandomForestRegressor

# # Train the random forest regression model with the new data
# model = RandomForestRegressor(n_estimators=5,max_depth=None)
# model.fit(X_train, y_train)

# # Predict on the training data
# y_pred = model.predict(X_test)
# y_pred = np.maximum(y_pred, 0)  # Remove negative predictions
# print(y_pred.shape)
# print(y_test.shape)

# # add the following line to avoid negative values
# y_pred = np.maximum(y_pred, 0)  # Remove negative predictions
# y_pred = np.round(y_pred).astype(int)  # Round and convert to integer
# # add 1 to avoid division by zero
# y_test = y_test + 1
# y_pred = y_pred + 1
# mape_fake = np.mean(np.abs((y_test - y_pred) / (y_test))) * 100
# print(f'MAPE_fake: {mape_fake}%')

# # Evaluate the model
# epsilon = 10e-7  # Small epsilon value to avoid division by zero
# mape = np.mean(np.abs((y_test + epsilon - y_pred) / (y_test + epsilon))) * 100

# # Calculate MAPE excluding zero values in y_test
# non_zero_indices = np.nonzero(y_test)
# mape_non_zero = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / (y_test[non_zero_indices]))) * 100

# print(f'MAPE (excluding zero values): {mape_non_zero}%')

# print(f'MAPE: {mape}%')

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# # Define the parameter grid for grid search
# param_grid = {
#     'n_estimators': [5, 10, 20],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10]
# }

# # Create the random forest regression model
# model = RandomForestRegressor()

# # Perform grid search to find the best hyperparameters
# grid_search = GridSearchCV(model, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # Get the best model with the optimal hyperparameters
# best_model = grid_search.best_estimator_

# # Train the best model with the new data
# best_model.fit(X_train, y_train)

# # Predict on the training data
# y_pred = best_model.predict(X_test)
# y_pred = np.maximum(y_pred, 0)  # Remove negative predictions

# y_test = y_test + 1
# y_pred = y_pred + 1
# mape_fake = np.mean(np.abs((y_test - y_pred) / (y_test))) * 100
# print(f'MAPE_fake: {mape_fake}%')

# # Evaluate the model
# epsilon = 10e-7  # Small epsilon value to avoid division by zero
# mape = np.mean(np.abs((y_test + epsilon - y_pred) / (y_test + epsilon))) * 100

# # Calculate MAPE excluding zero values in y_test
# non_zero_indices = np.nonzero(y_test)
# mape_non_zero = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / (y_test[non_zero_indices]))) * 100

# print(f'MAPE (excluding zero values): {mape_non_zero}%')
# print(f'MAPE: {mape}%')

# ############################################

# import xgboost as xgb

# # Train the XGBoost regression model with the new data
# model = xgb.XGBRegressor(n_estimators=5)
# model.fit(X_train, y_train)

# # Predict on the training data
# y_pred = model.predict(X_test)
# y_pred = np.maximum(y_pred, 0)  # Remove negative predictions
# print(y_pred.shape)
# print(y_test.shape)

# # add the following line to avoid negative values
# y_pred = np.maximum(y_pred, 0)  # Remove negative predictions
# y_pred = np.round(y_pred).astype(int)  # Round and convert to integer
# # add 1 to avoid division by zero
# y_test = y_test + 1
# y_pred = y_pred + 1
# mape_fake = np.mean(np.abs((y_test - y_pred) / (y_test))) * 100
# print(f'MAPE_fake: {mape_fake}%')

# # Evaluate the model
# epsilon = 10e-7  # Small epsilon value to avoid division by zero
# mape = np.mean(np.abs((y_test + epsilon - y_pred) / (y_test + epsilon))) * 100

# # Calculate MAPE excluding zero values in y_test
# non_zero_indices = np.nonzero(y_test)
# mape_non_zero = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / (y_test[non_zero_indices]))) * 100

# print(f'MAPE (excluding zero values): {mape_non_zero}%')

# print(f'MAPE: {mape}%')


# # # create a y_test_csv file
# # y_test_submit = model.predict(X_test_submit_IDS_ARRAY)
# # print(y_test_submit[:5])
# # result_df = pd.DataFrame({'data_station': data_station_array, 'y_test_submit': y_test_submit})
# # print(result_df.head())
# # result_df.rename(columns={'data_station': 'index', 'y_test_submit': 'y'}, inplace=True)
# # result_df.to_csv('y_test_sncf.csv', index=False)


# # ############################################
# # from sklearn.svm import SVR

# # # Train the SVM regression model with the new data
# # svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
# # svm_model.fit(X_train, y_train)

# # # Predict on the training data
# # y_pred_svm = svm_model.predict(X_test)
# # print(y_pred_svm.shape)
# # print(y_test.shape)

# # # Evaluate the model
# # epsilon = 1  # Small epsilon value to avoid division by zero
# # mape_svm = np.mean(np.abs((y_test - y_pred_svm) / (y_test + epsilon))) * 100

# # # Calculate MAPE excluding zero values in y_test
# # non_zero_indices_svm = np.nonzero(y_test)
# # mape_non_zero_svm = np.mean(np.abs((y_test[non_zero_indices_svm] - y_pred_svm[non_zero_indices_svm]) / (y_test[non_zero_indices_svm] + epsilon))) * 100

# # print(f'MAPE (excluding zero values) for SVM: {mape_non_zero_svm}%')
# # print(f'MAPE for SVM: {mape_svm}%')


