import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


y_train = pd.read_csv('y_train_sncf.csv')

# Create a new dataframe with three columns: date, station, and y
new_y_train = pd.DataFrame()
new_y_train['date'] = y_train['index'].str.split('_').str[0]
new_y_train['station'] = y_train['index'].str.split('_').str[1]
new_y_train['y'] = y_train['y'].values
print(y_train.shape)

# Print the new dataframe
print(new_y_train.head())