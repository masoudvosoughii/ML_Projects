# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset from a CSV file
dataset = pd.read_csv("/Users/ebrietas/machine learning/multiple_linear_regression/petrol_consumption.csv")

# Separate features (X) and target (y)
# The first 4 columns are data(features) , the 5th column is the target (target)
data = dataset.iloc[:, :4]
target = dataset.iloc[:, 4]

# Split the data into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

# Standardize features (important for SGD)
sc_data = StandardScaler()
x_train = sc_data.fit_transform(x_train)
x_test = sc_data.transform(x_test)

# Create and train the SGD Regressor model
sgd_regressor = SGDRegressor(max_iter=1000, random_state=42, eta0=0.01)
sgd_regressor.fit(x_train, y_train)

# Display number of epochs used
print("epochs :", sgd_regressor.n_iter_)

# Create a DataFrame to display the coefficients of each feature
sgd_coef_df = pd.DataFrame(sgd_regressor.coef_, data.columns, columns=['coefficient'])
print(sgd_coef_df)

# Predict the target values for the test data
y_pred = sgd_regressor.predict(x_test)

# Evaluate the model using common regression metrics
print('MAE :', metrics.mean_absolute_error(y_test, y_pred))
print('MSE :', metrics.mean_squared_error(y_test, y_pred))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))