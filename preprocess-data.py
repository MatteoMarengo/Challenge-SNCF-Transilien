gitimport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def x_values_processing(dfx):
    dfx['day_of_week'] = dfx['date'].dt.dayofweek + 1
    dfx['day_of_week_cos'] = np.cos(2 * np.pi * (dfx['day_of_week'] - 1) / 7)
    dfx['day_of_week_sin'] = np.sin(2 * np.pi * (dfx['day_of_week'] - 1) / 7)
    dfx['day_of_year'] = dfx['date'].dt.dayofyear

    X_station = dfx['station']
    station_mapping = {station: i for i, station in enumerate(X_station.unique())}
    dfx['station_id'] = dfx['station'].map(station_mapping)

    DFX_train_IDS = dfx[['day_of_week_cos', 'day_of_week_sin', 'station_id', 'job', 'ferie', 'vacances']]
    DFX_IDS_ARRAY = X_train_IDS.to_numpy()

    return DFX_IDS_ARRAY

def load_data_univariate(x_train_path, y_train_path, x_test_submit_path):
    # Load the data
    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    x_test_submit = pd.read_csv(x_test_submit_path)

    # XTRAIN
    X_train_IDS_ARRAY = x_values_processing(x_train)

    # XTEST
    X_test_submit_IDS_ARRAY = x_values_processing(x_test_submit)

    # YTRAIN
    Y_IDS = y_train['y']
    Y_IDS_ARRAY = Y_IDS.to_numpy()



