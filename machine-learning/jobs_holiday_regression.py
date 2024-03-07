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
X_train_IDS_ARRAY_job_1_vac_1_fer_1 = X_train_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 1) & (X_train_IDS_ARRAY[:, 4] == 1) & (X_train_IDS_ARRAY[:, 3] == 1)]
X_train_IDS_ARRAY_job_1_vac_1_fer_0 = X_train_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 1) & (X_train_IDS_ARRAY[:, 4] == 1) & (X_train_IDS_ARRAY[:, 3] == 0)]
X_train_IDS_ARRAY_job_1_vac_0_fer_1 = X_train_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 1) & (X_train_IDS_ARRAY[:, 4] == 0) & (X_train_IDS_ARRAY[:, 3] == 1)]
X_train_IDS_ARRAY_job_1_vac_0_fer_0 = X_train_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 1) & (X_train_IDS_ARRAY[:, 4] == 0) & (X_train_IDS_ARRAY[:, 3] == 0)]
X_train_IDS_ARRAY_job_0_vac_1_fer_1 = X_train_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 0) & (X_train_IDS_ARRAY[:, 4] == 1) & (X_train_IDS_ARRAY[:, 3] == 1)]
X_train_IDS_ARRAY_job_0_vac_1_fer_0 = X_train_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 0) & (X_train_IDS_ARRAY[:, 4] == 1) & (X_train_IDS_ARRAY[:, 3] == 0)]
X_train_IDS_ARRAY_job_0_vac_0_fer_1 = X_train_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 0) & (X_train_IDS_ARRAY[:, 4] == 0) & (X_train_IDS_ARRAY[:, 3] == 1)]
X_train_IDS_ARRAY_job_0_vac_0_fer_0 = X_train_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 0) & (X_train_IDS_ARRAY[:, 4] == 0) & (X_train_IDS_ARRAY[:, 3] == 0)]

# Associate the y to the new X_train
Y_train_IDS_ARRAY_job_1_vac_1_fer_1 = Y_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 1) & (X_train_IDS_ARRAY[:, 4] == 1) & (X_train_IDS_ARRAY[:, 3] == 1)]
Y_train_IDS_ARRAY_job_1_vac_1_fer_0 = Y_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 1) & (X_train_IDS_ARRAY[:, 4] == 1) & (X_train_IDS_ARRAY[:, 3] == 0)]
Y_train_IDS_ARRAY_job_1_vac_0_fer_1 = Y_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 1) & (X_train_IDS_ARRAY[:, 4] == 0) & (X_train_IDS_ARRAY[:, 3] == 1)]
Y_train_IDS_ARRAY_job_1_vac_0_fer_0 = Y_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 1) & (X_train_IDS_ARRAY[:, 4] == 0) & (X_train_IDS_ARRAY[:, 3] == 0)]
Y_train_IDS_ARRAY_job_0_vac_1_fer_1 = Y_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 0) & (X_train_IDS_ARRAY[:, 4] == 1) & (X_train_IDS_ARRAY[:, 3] == 1)]
Y_train_IDS_ARRAY_job_0_vac_1_fer_0 = Y_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 0) & (X_train_IDS_ARRAY[:, 4] == 1) & (X_train_IDS_ARRAY[:, 3] == 0)]
Y_train_IDS_ARRAY_job_0_vac_0_fer_1 = Y_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 0) & (X_train_IDS_ARRAY[:, 4] == 0) & (X_train_IDS_ARRAY[:, 3] == 1)]
Y_train_IDS_ARRAY_job_0_vac_0_fer_0 = Y_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == 0) & (X_train_IDS_ARRAY[:, 4] == 0) & (X_train_IDS_ARRAY[:, 3] == 0)]


# ######################
# ## RANDOM FOREST ##
# ######################

from sklearn.ensemble import RandomForestRegressor

def Random_FOREST_train(array1,array2,n_estimators=5):

    X_train, X_test, y_train, y_test = train_test_split(array1, array2, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators)
    model.fit(X_train, y_train)

    # Predict on the training data
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Remove negative predictions
    y_pred = np.round(y_pred).astype(int)  # Round and convert to integer
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

    return model

model1 = Random_FOREST_train(X_train_IDS_ARRAY,Y_IDS_ARRAY)
y_test_submit = model1.predict(X_test_submit_IDS_ARRAY)
y_test_submit = np.maximum(y_test_submit, 0)  # Remove negative predictions
y_test_submit = np.round(y_test_submit).astype(int)  # Round and convert to integer

models = []

for job in [0, 1]:
    for vacation in [0, 1]:
        for holiday in [0, 1]:
            print(job, vacation, holiday)
            X_train_subset = X_train_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == job) & (X_train_IDS_ARRAY[:, 4] == vacation) & (X_train_IDS_ARRAY[:, 3] == holiday)]
            Y_train_subset = Y_IDS_ARRAY[(X_train_IDS_ARRAY[:, 2] == job) & (X_train_IDS_ARRAY[:, 4] == vacation) & (X_train_IDS_ARRAY[:, 3] == holiday)]
            model = Random_FOREST_train(X_train_subset, Y_train_subset)
            models.append(model)

# job = 1, vacation = 0, holiday = 0
X_test_submit_IDS_ARRAY_job_1_vac_0_fer_0 = X_test_submit_IDS_ARRAY[(X_test_submit_IDS_ARRAY[:, 2] == 1) & (X_test_submit_IDS_ARRAY[:, 4] == 0) & (X_test_submit_IDS_ARRAY[:, 3] == 0)]
y_test_new_job_1_vac_0_fer_0 = models[4].predict(X_test_submit_IDS_ARRAY_job_1_vac_0_fer_0)
y_test_submit[(X_test_submit_IDS_ARRAY[:, 2] == 1) & (X_test_submit_IDS_ARRAY[:, 4] == 0) & (X_test_submit_IDS_ARRAY[:, 3] == 0)] = y_test_new_job_1_vac_0_fer_0
y_test_submit = np.maximum(y_test_submit, 0)  # Remove negative predictions
y_test_submit = np.round(y_test_submit).astype(int)  # Round and convert to integer


# job = 1, vacation = 1, holiday = 0
X_test_submit_IDS_ARRAY_job_1_vac_1_fer_0 = X_test_submit_IDS_ARRAY[(X_test_submit_IDS_ARRAY[:, 2] == 1) & (X_test_submit_IDS_ARRAY[:, 4] == 1) & (X_test_submit_IDS_ARRAY[:, 3] == 0)]
y_test_new_job_1_vac_1_fer_0 = models[6].predict(X_test_submit_IDS_ARRAY_job_1_vac_1_fer_0)
y_test_submit[(X_test_submit_IDS_ARRAY[:, 2] == 1) & (X_test_submit_IDS_ARRAY[:, 4] == 1) & (X_test_submit_IDS_ARRAY[:, 3] == 0)] = y_test_new_job_1_vac_1_fer_0
y_test_submit = np.maximum(y_test_submit, 0)  # Remove negative predictions
y_test_submit = np.round(y_test_submit).astype(int)  # Round and convert to integer

# create a y_test_csv file
result_df = pd.DataFrame({'data_station': data_station_array, 'y_test_submit': y_test_submit})
print(result_df.head())
result_df.rename(columns={'data_station': 'index', 'y_test_submit': 'y'}, inplace=True)
result_df.to_csv('y_test_sncf.csv', index=False)
