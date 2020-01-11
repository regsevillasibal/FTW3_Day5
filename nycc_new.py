
# IMPORT GENERIC PACKAGES
import numpy as np # numerical calc package
import pandas as pd # holds data
import matplotlib.pyplot as plt # plotting library
import seaborn as sns # pretty plotting

# plotting config
sns.set(style='white', rc={'figure.figsize':(20,10)})

from sklearn.linear_model import LinearRegression # linear regression package
from sklearn.model_selection import train_test_split # split dataset
from sklearn.metrics import mean_squared_error as mse # Measurement metric

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# read data into a DataFrame
data = pd.read_csv('nyc-rolling-sales.csv', index_col=0)

data.head()

data.columns

# take a subset of our data
columns = [ 'BOROUGH', 'SALE PRICE','COMMERCIAL UNITS','LAND SQUARE FEET', 'GROSS SQUARE FEET' ]
subset_data = data[columns]

# Get number of (rows, columns)
subset_data.shape

# Get first 5 rows
subset_data.head()

# Convert to float
data['SALE PRICE'] = pd.to_numeric(data['SALE PRICE'], errors='coerce')
data['SALE PRICE'] = data['SALE PRICE'].fillna(0)

data['GROSS SQUARE FEET'] = pd.to_numeric(data['GROSS SQUARE FEET'], errors='coerce')
data['LAND SQUARE FEET'] = pd.to_numeric(data['LAND SQUARE FEET'], errors='coerce')

# Convert to date
data['SALE DATE'] = pd.to_datetime(data['SALE DATE'], errors='coerce')


# Remove 5th and 95th percentile tails
zero = 0
fifth = data['SALE PRICE'].describe(np.arange(0.05, 1, 0.05)).T['15%']
ninetyfifth = data['SALE PRICE'].describe(np.arange(0.05, 1, 0.05)).T['95%']
data = data[(data['SALE PRICE'] > zero) &
             (data['SALE PRICE'] <= ninetyfifth)].copy()

# Handle Missing Values by Dropping (for now)
data.dropna(inplace=True)

data.describe()

# Define Features
features = ['BOROUGH', 'COMMERCIAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','RESIDENTIAL UNITS','LOT', 'BLOCK', 'ZIP CODE' ]

# Set X
X = data[features]

# Set y
y = data['SALE PRICE']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# Initialize model
model = RandomForestRegressor()

# Fit Model
model.fit(X_train, y_train)

# calculate the R-squared
model.score(X_test, y_test)

y_predicted = model.predict(X_test)

# We input new advertising data into the model to predict future sales

# Sample
new_data = [[2, 2, 10000,1,1,1,1,1]]
model.predict(new_data)

"""### Model Error"""

np.sqrt(mse(y_predicted, y_test)) # Root mean squared error

"""This means that the root mean square error of any prediction done by the model against the actual value should be ~2 thousand dollars per campaign. Your predictions should deviate from the real values only by about 2 thousand dollars. This is high but we can further improve on this model by doing feature engineering but for now, this will do."""
