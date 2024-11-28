# loan-default-model
# Overview

This project develops a predictive model to estimate the probability of loan default (PD) for retail banking customers based on borrower characteristics. By analyzing the provided dataset, we built models that help identify high-risk borrowers and assess the potential financial impact of defaults. This tool aids the risk management team in setting aside sufficient capital to mitigate losses.

# Objectives

Predict Default Probabilities: Use borrower features to estimate the likelihood of default.

Identify Risk Factors: Analyze patterns in the data to uncover characteristics associated with higher default probabilities.

Model Comparison: Compare logistic regression and decision tree models to evaluate performance.

Visual Insights: Use data visualizations to interpret the relationships between borrower attributes and default risk.

# Dataset

The dataset contains information about borrowers, including:

Income: Borrower's monthly income.

Loan Amount: Total amount of the loan.

Loan-to-Income Ratio: Ratio of loan amount to income.

Credit History: Indicator of past defaults.

Default: Target variable (1 if the borrower defaulted, 0 otherwise).

# Steps Implemented

1. Data Preprocessing

Loaded the dataset using Pandas.

Checked for missing or invalid data and removed such entries.

Created new features like LoanToIncome and grouped borrowers into bins based on the loan-to-income ratio.

2. Exploratory Data Analysis (EDA)

Visualized the distribution of features to understand data patterns.

Analyzed default rates by Loan-to-Income ratio bins.

Found significant increases in default probabilities as the ratio exceeded certain thresholds.

3. Feature Engineering

Standardized numerical features for consistent scaling.

Encoded categorical variables where applicable.

4. Model Building

# Built two models:

Logistic Regression: Effective for simple linear relationships.

Decision Tree: Captured non-linear interactions and complex patterns.

5. Model Evaluation

Compared models using metrics like:

ROC-AUC: Measures the ability of the model to separate defaulters from non-defaulters.

Precision/Recall: Evaluates model accuracy in predicting defaults.

Visualized the ROC curve and analyzed feature importance.

6. Visualization

Created bar charts showing default rates by Loan-to-Income ratio groups.

Plotted the ROC curves to compare model performance.

# How to Run the Project

Prerequisites

Python 3.8 or later.

Required libraries:

pandas

numpy

matplotlib

scikit-learn

Steps

Clone the repository:

git clone <repository_url>
cd loan-default-model

Install dependencies:

pip install -r requirements.txt

Run the script:

python3 task3.py

Follow prompts to view predictions and visualizations.

Results

Logistic Regression: Achieved an ROC-AUC of ~0.999, indicating excellent predictive accuracy for this dataset.

Decision Tree: Performed slightly less well but still effective with an ROC-AUC of ~0.995.

Default rates increased sharply for Loan-to-Income ratios above 0.07, highlighting it as a key risk indicator.

## Contributors: 
Nicolas Park
