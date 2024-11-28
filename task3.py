import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, classification_report

import matplotlib.pyplot as plt
import numpy as np


# Load the dataset
file_path = 'Task 3 and 4_Loan_Data.csv'
data = pd.read_csv(file_path)

# Look at the first few rows
print(data.head())

# Check for missing values and data types
print(data.info())

# Get a statistical summary of numerical columns
print(data.describe())

# Drop rows with missing values
data = data.dropna()

# Verify no missing values remain
print(data.info())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate the target and features
X = data.drop(columns=['default'])  # Replace 'Default' with the actual target column
y = data['default']

# Convert categorical variables to numerical (if any)
X = pd.get_dummies(X, drop_first=True)

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression apporach

# Initialize and train the logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_prob_log = log_model.predict_proba(X_test)[:, 1]  # Get probability of "default = 1"

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred_prob_log)
print("Logistic Regression ROC-AUC:", roc_auc)
print("\nClassification Report:\n", classification_report(y_test, log_model.predict(X_test)))

# extra log reg work
feature_names = X.columns  # Names of your features
coefficients = log_model.coef_[0]  # Coefficients from logistic regression
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
print(feature_importance)


# Decision Tree Approach

# Train decision tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)  # Limit depth to prevent overfitting
tree_model.fit(X_train, y_train)

# Predict probabilities
y_pred_prob_tree = tree_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Decision Tree ROC-AUC:", roc_auc_score(y_test, y_pred_prob_tree))
print("\nClassification Report:\n", classification_report(y_test, tree_model.predict(X_test)))

# extra decision tree work
feature_importances = tree_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)

# graphing

# Create a new column for loan-to-income ratio
data['LoanToIncome'] = data['loan_amt_outstanding'] / data['income']

# Group by LoanToIncome thresholds
data['LoanToIncomeGroup'] = pd.cut(data['LoanToIncome'], bins=[0, 1, 2, 3, 5, 10, np.inf], labels=['<1', '1-2', '2-3', '3-5', '5-10', '>10'])

# Calculate default rates for each group with observed=False explicitly set
default_rates = data.groupby('LoanToIncomeGroup', observed=False)['default'].mean()

data['LoanToIncomeGroup'] = pd.cut(
    data['LoanToIncome'],
    bins=[0, 0.05, 0.06, 0.07, 0.08, 0.1, np.inf],  # Fine-tuned bins
    labels=['<0.05', '0.05-0.06', '0.06-0.07', '0.07-0.08', '0.08-0.1', '>0.1']
)
default_rates = data.groupby('LoanToIncomeGroup', observed=False)['default'].mean()
default_rates = default_rates.fillna(0)  # Replace NaNs with 0

default_rates = default_rates[default_rates > 0]  # Remove empty bins

# Plot the results
default_rates.plot(kind='bar', color='blue')
plt.title('Default Rates by Loan-to-Income Ratio (Adjusted)')
plt.xlabel('Loan-to-Income Ratio Group')
plt.ylabel('Default Rate')
plt.savefig('default_rates_plot.png')  # Save the plot as an image
print("Graph saved as 'default_rates_plot.png'")




