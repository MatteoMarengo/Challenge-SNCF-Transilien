import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
x_train = pd.read_csv('x_train_sncf.csv')
x_train = pd.read_csv('train_f_x.csv')
x_test_submit = pd.read_csv('x_test_sncf.csv')
y_train = pd.read_csv('y_train_sncf.csv')

# Create one dataframe for each station
station_dfs = []
for station in x_train['station'].unique():
    station_df = x_train[x_train['station'] == station].copy()
    station_dfs.append(station_df)
    print(f"The station name is {station}")

# Access the first station dataframe
station_df = station_dfs[0]
# Perform operations on the first station dataframe
# For example, you can preprocess the data specific to the first station
# station_df['column'] = station_df['column'].apply(some_function)
print(station_df.head())

def x_values_processing(dfx):
    dfx['date'] = pd.to_datetime(dfx['date'])
    dfx['day_of_week'] = dfx['date'].dt.dayofweek + 1
    dfx['day_of_week_cos'] = np.cos(2 * np.pi * (dfx['day_of_week'] - 1) / 7)
    dfx['day_of_week_sin'] = np.sin(2 * np.pi * (dfx['day_of_week'] - 1) / 7)

    X_station = dfx['station']
    station_mapping = {station: i for i, station in enumerate(X_station.unique())}
    dfx['station_id'] = dfx['station'].map(station_mapping)

    DFX_train_IDS = dfx[['day_of_week_cos', 'day_of_week_sin', 'station_id', 'job', 'ferie', 'vacances']]
    DFX_IDS_ARRAY = DFX_train_IDS.to_numpy()

    return DFX_IDS_ARRAY

# Perform x_values_processing on each station dataframe
processed_station_dfs = []
for station_df in station_dfs:
    processed_station_df = x_values_processing(station_df)
    processed_station_dfs.append(processed_station_df)

# Create a new dataframe with three columns: date, station, and y
new_y_train = pd.DataFrame()
new_y_train['date'] = y_train['index'].str.split('_').str[0]
new_y_train['station'] = y_train['index'].str.split('_').str[1]
new_y_train['y'] = y_train['y'].values


# Create one dataframe for each station
station_dfs_y = []
for station in new_y_train['station'].unique():
    station_df = new_y_train[new_y_train['station'] == station]['y'].copy()
    station_dfs_y.append(station_df)
    print(f"The station name is {station}")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Split each dataframe into train and test sets
train_dfs = []
test_dfs = []
for station_df,station_dfy in zip(processed_station_dfs,station_dfs_y):
    X_train, X_test, y_train, y_test = train_test_split(station_df, station_dfy, test_size=0.2, random_state=42)
    train_dfs.append((X_train, y_train))
    test_dfs.append((X_test, y_test))

# Perform Random Forest Regression and predict y_test for each station
y_preds = []
for i, (X_train, y_train) in enumerate(train_dfs):
    X_test, y_test = test_dfs[i]
    model = RandomForestRegressor(n_estimators=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_preds.append(y_pred)

# Merge all results of y_pred
merged_y_pred = np.concatenate(y_preds)

# Compute MAPE score
mape = mean_absolute_percentage_error(new_y_train['y'], merged_y_pred)
print(f"MAPE score: {mape}")


