# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Function to load and preprocess the dataset
def load_data():
    # Load dataset from CSV file
    dataset = pd.read_csv("/Users/ebrietas/petprojects/Projects/machine learning/logistic_regression/diabetes.csv")

    # List of columns where 0 is not a valid value
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

    # Replace 0s with NaN and then fill NaNs with the column mean
    for column in zero_not_accepted:
        dataset[column] = dataset[column].replace(0, np.nan)
        mean = int(dataset[column].mean(skipna=True))
        dataset[column] = dataset[column].replace(np.nan, mean)

    # Split dataset into features (x) and labels (y)
    x = dataset.iloc[:, :8]  # All rows, columns 0-7 (features)
    y = dataset.iloc[:, 8]  # All rows, column 8 (target)

    # Split data into training and testing sets (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale features using StandardScaler
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)  # Fit and transform training data
    x_test = sc_x.transform(x_test)  # Transform test data

    return x_train, x_test, y_train, y_test


# Function to train the K-Nearest Neighbors classifier
def algorithm(x_train, y_train):
    #clf = LogisticRegression()
    clf = SGDClassifier(loss= "log_loss")
    clf.fit(x_train, y_train)  # Train classifier
    return clf


# Function to evaluate the model and print accuracy
def show_results(clf, x_test, y_test):
    y_pred = clf.predict(x_test)  # Predict labels for test set
    acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print("Accuracy: {:.2f}%".format(acc * 100))  # Print accuracy in %



# Main execution
x_train, x_test, y_train, y_test = load_data()  # Load and preprocess data
clf = algorithm(x_train, y_train)  # Train the model
show_results(clf, x_test, y_test)  # Evaluate and print results