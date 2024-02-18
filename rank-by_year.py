import pandas as pd
import matplotlib.pyplot as plt

# Load the data
# Assuming the filename is 'data.csv' and is in the same directory as the script
df = pd.read_csv('x_train_sncf.csv')
# df = df.sort_values('date')


# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Create a new column with the numerical day of the year
df['day_of_year'] = df['date'].dt.dayofyear

# Now df['day_of_year'] contains the numerical representation of the date

# If you need to reset the day count at the beginning of each year, you can do so by:
# Grouping the data by year and then applying a custom function to enumerate the days within each group
df['numerical_date'] = df.groupby(df['date'].dt.year)['date'].rank(method='first').astype(int)

print(df.iloc[190000:190005])


# Now df['numerical_date'] contains the numerical day count reset at the start of each year

# To plot the data for a specific station throughout the year, you would filter the DataFrame and then plot
# For example, to plot the data for station 'IJ7':
station_data = df[df['station'] == '1J7']
plt.plot(station_data['date'], station_data['vacances'])  # Replace 'vacances' with the correct column name for the y values
plt.xlabel('Date')
plt.ylabel('Frequentation')
plt.title('Frequentation over time for station IJ7')
plt.show()
