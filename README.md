# Loan Default Prediction Using Machine Learning

## 📌 Project Overview

Loan Default Prediction is a machine learning classification project that aims to predict whether a borrower is likely to default on a loan based on their personal, financial, employment, and credit-related information.

Financial institutions face challenges in identifying borrowers who may fail to repay their loans. By analyzing historical loan data and identifying patterns associated with defaults, machine learning techniques can help assess potential credit risks.

This project explores the application of supervised machine learning to classify borrowers into two categories: **Default** and **Non-Default**.

## 🎯 Project Objectives

- Analyze historical loan data to understand the factors associated with loan defaults.
- Perform Exploratory Data Analysis (EDA) to identify patterns, distributions, and relationships between features.
- Clean and preprocess the dataset for machine learning.
- Convert categorical variables into numerical representations.
- Apply feature scaling to numerical attributes.
- Develop a classification model to predict loan default.
- Evaluate model performance using appropriate classification metrics.

## 📂 Dataset Description

The project uses the Loan Default Prediction dataset.

**Dataset Statistics:**

| Attribute | Description |
|---|---|
| Total Records | 255,347 |
| Total Columns | 18 |
| Input Features | 17 before removing LoanID |
| Target Variable | Default |
| Problem Type | Binary Classification |

### Features Description

The dataset contains borrower information across several categories:

**1. Personal Information**
- Age
- Education
- Marital Status
- Dependents

**2. Financial Information**
- Income
- Loan Amount
- Credit Score
- Interest Rate
- Debt-to-Income Ratio (DTIRatio)

**3. Employment Information**
- Employment Type
- Months Employed

**4. Loan Information**
- Loan Term
- Number of Credit Lines
- Loan Purpose
- Mortgage Status
- Co-signer Status

### Target Variable

The target column is `Default`.

- 0 – Loan Not Defaulted
- 1 – Loan Defaulted

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the structure and characteristics of the dataset before building the machine learning model.

The following analysis was carried out:

- Dataset dimensions and data types inspection.
- Statistical analysis of numerical variables.
- Missing value identification.
- Duplicate record detection.
- Distribution analysis of numerical features.
- Identification of potential outliers using boxplots.
- Analysis of categorical variables.
- Examination of the target variable distribution.
- Visualization of relationships between selected features.

Libraries such as Matplotlib and Seaborn were used for data visualization.

## ⚙️ Data Preprocessing

Data preprocessing was performed to prepare the dataset for machine learning.

The following steps were included:

**1. Data Cleaning**
- Checked the dataset for missing values.
- Checked for duplicate records.
- Removed the unnecessary `LoanID` column because it does not provide meaningful predictive information.

**2. Feature Selection**
- Separated the independent variables from the target variable.
- Selected borrower-related attributes as input features.
- Used `Default` as the target variable.

**3. Encoding Categorical Variables**
- Applied Label Encoding to the Education feature.
- Used One-Hot Encoding for nominal categorical variables.
- Removed reference categories to avoid redundant encoded columns.

**4. Feature Scaling**
- Applied Standard Scaling to numerical features to bring them onto a comparable scale.

**5. Train-Test Split**
- Divided the dataset into training and testing sets.
- Used an 80:20 split.
- Applied stratified sampling to maintain the target class distribution.

## 🤖 Machine Learning Approach

This project is formulated as a supervised binary classification problem.

The objective is to train a machine learning model using historical borrower information and predict whether a new borrower is likely to default on a loan.

The model development and evaluation stages are part of the machine learning workflow.

### Model Evaluation

Model performance can be assessed using the following classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

These metrics help evaluate the model's ability to distinguish between defaulting and non-defaulting borrowers.

## 🛠️ Technologies & Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data preprocessing and machine learning |
| Jupyter Notebook | Development environment |

## 📁 Repository Structure

The repository contains the following files:

- `Loan_Default_Prediction_ML_Project.ipynb` – Jupyter Notebook containing data analysis and machine learning workflow.
- `Loan_default.csv` – Dataset used for the project.
- `README.md` – Project documentation.

## 🚀 Future Improvements

- Experiment with multiple classification algorithms.
- Handle class imbalance using suitable sampling techniques.
- Perform hyperparameter optimization.
- Compare different models based on evaluation metrics.
- Improve prediction performance through feature engineering.
- Develop an interactive loan default prediction application using Streamlit.
- Explore containerization using Docker.
- Deploy the application on AWS.

## 👨‍💻 Project Status

**Status:** Work in Progress

This project is being developed as a practical application of Python, data analysis, and machine learning concepts, with a focus on financial risk prediction.
