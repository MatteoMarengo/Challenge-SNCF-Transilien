import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
x_train = pd.read_csv('x_train_sncf.csv')
y_train = pd.read_csv('y_train_sncf.csv')

# Preprocess the data
# For example, parse the dates and convert them to a numerical value, such as the number of days since a certain date
x_train['date'] = pd.to_datetime(x_train['date'])
x_train['days_since'] = (x_train['date'] - x_train['date'].min()).dt.days

# You would typically convert categorical data to numerical here, but for simplicity, let's assume 'station' is numerical
# You could also drop the 'date' column if it's no longer needed after extracting relevant features from it

# Prepare the features and target variables
X = x_train[['days_since', 'job', 'ferie', 'vacances']]  # Assuming these are the features you want to use
y = y_train['y']

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict on the training data
y_pred = model.predict(X)

# Evaluate the model
mape = np.mean(np.abs((y - y_pred) / y)) * 100

print(f'MAPE: {mape}%')
