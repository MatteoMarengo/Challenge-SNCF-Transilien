import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('x_train_sncf.csv')
# Perform basic data exploration
# Display the first few rows of the data
print(data.head())

# Get the shape of the data (number of rows, number of columns)
print(data.shape)

# Get summary statistics of the data
print(data.describe())

# Check for missing values
print(data.isnull().sum())

num_stations = len(data.iloc[:, 1].unique())
print("Number of different stations:", num_stations)


