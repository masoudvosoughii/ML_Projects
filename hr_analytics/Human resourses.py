# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
data = pd.read_csv('/Users/ebrietas/Downloads/mathdown/turnover.csv')

# Display first 5 rows
print(data.head())

# Display last 5 rows
print(data.tail())

# Show shape of the dataset (rows, columns)
print("Shape of dataset:", data.shape)

# Summary statistics
print(data.describe())

# Dataset info: column names, non-null counts, data types
data.info()

# Check how many salary types are in the dataset
print("Salary types:\n", data.groupby('salary').size())

# Check for missing values
print("Any missing values:", data.isnull().values.any())

# Print column names
column_names = data.columns.tolist()
print("Column names:", column_names)

# Count employees who left the company
print("Employees who left:\n", data[data['left'] == 1].count())

# Count employees who stayed
print("Employees who stayed:", data[data['left'] == 0].shape[0])

# Calculate the mean values for employees who left
a = data[data['left'] == 1].mean(numeric_only=True)
print("Averages (left):\n", a)

# Calculate the mean values for employees who stayed
b = data[data['left'] == 0].mean(numeric_only=True)
print("Averages (stayed):\n", b)

# Calculate average working hours per day
work_hour_left = a['average_montly_hours'] / 30
work_hour_stayed = b['average_montly_hours'] / 30
print("Avg working hours per day (left):", work_hour_left)
print("Avg working hours per day (stayed):", work_hour_stayed)

# Compare promotion in last 5 years
promotion_left = a['promotion_last_5years']
promotion_stayed = b['promotion_last_5years']
print("Promotion (left):", promotion_left)
print("Promotion (stayed):", promotion_stayed)

# Avoid division by zero
if promotion_left != 0:
    print("Promotion rate % more among stayed:", (promotion_stayed / promotion_left) * 100)
else:
    print("Promotion (left) is zero, can't compare.")

# Compute correlation between features
correlation = data.corr(numeric_only=True)
print("Feature correlation:\n", correlation)

# Heatmap of correlations
plt.figure(figsize=(6, 6))
sns.heatmap(correlation, square=True, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Create boxplots for different features by salary
features = [
    'satisfaction_level', 'last_evaluation', 'number_project',
    'average_montly_hours', 'time_spend_company',
    'Work_accident', 'left', 'promotion_last_5years'
]

for feature in features:
    data.boxplot(column=feature, by='salary', grid=False)
    plt.title(f'{feature} by Salary')
    plt.suptitle("")
    plt.xlabel("Salary")
    plt.ylabel(feature)
    plt.show()

# Import ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Convert 'salary' text to numerical values
data['salary'] = data['salary'].map({'low': 1, 'medium': 2, 'high': 3})

# Encode 'sales' (department) as numeric values
sales = sorted(data['sales'].unique())
sales_map = dict(zip(sales, range(len(sales))))
data['sales'] = data['sales'].map(sales_map).astype(int)

# Check the updated dataset
print("Updated dataset preview:\n", data.head())

# Prepare feature matrix X and target vector y
X = data.drop('left', axis=1).values
y = data['left'].values

# Split dataset into train (75%) and test (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict probabilities on test set
y_pred = rf_model.predict_proba(X_test)

# Evaluate accuracy
accuracy = rf_model.score(X_test, y_test)
print("Random Forest Accuracy:", accuracy)

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=10)
print("Cross-validation scores:", cv_scores)
print("Average CV score:", cv_scores.mean())

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': data.drop('left', axis=1).columns,
    'Importance': rf_model.feature_importances_
})
print("Feature Importance:\n", feature_importance)

# Predict "soon leave" for current employees (those who haven't left)
df_current = data[data['left'] == 0].copy()
X_current = df_current.drop('left', axis=1).values
pred_probs = rf_model.predict_proba(X_current)

# Add prediction probabilities as a new column
df_current['Soon leave'] = pred_probs[:, 1]

# Show employees very likely to leave (>= 75%)
high_risk_employees = df_current[df_current['Soon leave'] >= 0.75]
print("Employees likely to leave soon (>= 0.75):\n", high_risk_employees)

# Show employees at moderate risk (>= 50%)
moderate_risk_employees = df_current[df_current['Soon leave'] >= 0.5]
print("Employees possibly leaving soon (>= 0.5):\n", moderate_risk_employees)