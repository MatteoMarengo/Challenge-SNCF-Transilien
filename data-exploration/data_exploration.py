# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Load the data
# x_train = pd.read_csv('x_train_sncf.csv')
# y_train = pd.read_csv('y_train_sncf.csv')
# print(x_train.head())

# # Preprocess the data
# # For example, parse the dates and convert them to a numerical value, such as the number of days since a certain date
# x_train['date'] = pd.to_datetime(x_train['date'])
# x_train['days_since'] = (x_train['date'] - x_train['date'].min()).dt.days

# # You would typically convert categorical data to numerical here, but for simplicity, let's assume 'station' is numerical
# # You could also drop the 'date' column if it's no longer needed after extracting relevant features from it

# # Prepare the features and target variables
# X = x_train[['date','station', 'job', 'ferie', 'vacances']]  # Assuming these are the features you want to use
# print(X.head())
# y = y_train['y']

# X_station = x_train['station']
# print(X_station.head())
# station_mapping = {station: i for i, station in enumerate(X_station.unique())}
# for station in station_mapping:
#     print(station, station_mapping[station])

# X['station_id'] = X['station'].map(station_mapping)
# print(X.head())
# x_train['station_id'] = x_train['station'].map(station_mapping)

# X_new_int = x_train[['days_since','station_id', 'job', 'ferie', 'vacances']]
# print(X_new_int.head())

# X_new_int_array = X_new_int.to_numpy()
# print(X_new_int_array)

# y_int_array = y.to_numpy()
# print(y_int_array)

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('y_train_sncf.csv')

# Define the station name
station = 'RDK'

# The first column is date_station, so we need to split it into date and station
df['date'] = df['index'].str.split('_').str[0]
#print(df['date'].head())

# now add the station name next to the date
df['station'] = df['index'].str.split('_').str[1]
#print(df['station'].head())

# Now make a dataframe with the date, the station name and the target variable
df = df[['date', 'station', 'y']]
#print(df.head())

# Sort the dataframe by date (ascending)
df = df.sort_values('date')
# print(df.head())

# Now we can filter the dataframe to only include the station we want
df = df[df['station'] == station]
# print(df.head())

# Now we can plot the target variable
plt.plot(range(len(df['y'])),df['y'])
plt.xlabel('Date')
plt.ylabel('Target variable')
plt.title('Target variable for station ' + station)

# Add a line every 365 values
for i in range(365, len(df['y']), 365):
    plt.axvline(x=i, color='r', linestyle='--')

plt.show()

# Now we can plot the target variable
plt.plot(range(7), df['y'][11:18])
plt.xlabel('Date')
plt.ylabel('Target variable')
plt.title('Target variable for station ' + station)

plt.show()

