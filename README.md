

# Loan Default Prediction: README

## Project Overview:

This project aims to build a machine learning model to predict loan defaults using historical customer data from a German bank. The dataset used for this project contains 17 features related to the customers' financial and demographic profiles. Various machine learning models were applied to analyze and predict loan default risks, with particular attention paid to class imbalance issues.

## Dataset
- File: German_bank.csv
- Description: The dataset contains 17 features and 1,000 rows. Each row represents a customer, and the target variable, 'default,' indicates whether a customer defaulted on their loan.
- Key Features:
  - `checking_balance`: Available balance in the account
  - `months_loan_duration`: Duration since loan taken
  - `credit_history`: Credit history of the customer
  - `amount`: Loan amount
  - `purpose`: Purpose why loan has been taken
  - `savings_balance`: Balance in account
  - `employment_duration` : Duration of employment
  - `percent_of_income` : Percentage of monthly income
  - `years_at_residence` : Duration of current residence
  - `age` : Age of customer
  - `other_credit` : Any other credits taken
  - `housing` : Type of housing, rent or own
  - `existing_loans_count` : Existing count of loans
  - `job` : Job type
  - `dependents` : Any dependents on customer
  - `phone` : Having phone or not
  - `default`: Target variable indicating default status

## Project Structure
The project is divided into the following sections:

1. Introduction:
   - Overview of the project and the importance of predicting loan defaults for financial stability.

2. Loading Libraries and Data:
   - Importing necessary libraries and loading the dataset for analysis.

3. Exploratory Data Analysis (EDA):
   - Initial exploration of the dataset to understand distributions and relationships between features.

4. Visualization:
   - Visualizations such as KDE + histograms for numerical features, bar plots for categorical features, box plots to examine numerical features against the target variable, pair plots for relationships between numerical variables, and a correlation heatmap.

5. Data Preprocessing and Feature Selection:
   - Data cleaning and feature engineering including handling redundant values, encoding categorical variables, and scaling numerical features. 

6. Model Training:
   - Base Models: Applied various base models such as Logistic Regression, Decision Tree, Random Forest, Gradient Boost, XGBoost, KNN Classifier, Naive Bayes, and LightGBM.
   - Penalty for Minority: Addressed class imbalance by applying penalty-based adjustments in models such as Logistic Regression, Decision Tree, and Random Forest.
   - Anomaly Detection: Used Isolation Forest and One-Class SVM for anomaly detection.
   - Hyperparameter Tuning: Performed hyperparameter optimization for Random Forest and XGBoost.
   - Models Performance Evaluation: Evaluated the models using metrics such as accuracy, precision, recall, and ROC AUC, along with visualization of ROC AUC and Precision-Recall curves.

7. Conclusion:
   - Summarized key findings, including the effectiveness of ensemble models, challenges related to class imbalance, and the importance of careful model tuning to avoid overfitting.


## How to Run the Project
1. Requirements:
   - Python 3.x
   - Jupyter Notebook
   - Libraries: pandas, numpy, sklearn, matplotlib, seaborn, xgboost, lightgbm

2. Steps:
   - Open the provided Jupyter Notebook file `UoA_521_Project.ipynb`.
   - Ensure the dataset `German_bank.csv` is placed in the same directory as the notebook.
   - Execute the cells in the notebook sequentially. The code covers data exploration, preprocessing, model training, evaluation, and hyperparameter tuning.

3. Outputs:
   - The notebook generates various visualizations, performance metrics, and tables summarizing model performance.
   - The final tuned models, along with ROC and Precision-Recall curves, are presented at the end of the notebook.

## Future Work
- Experimenting with additional techniques for handling class imbalance, such as SMOTE.
- Expanding the dataset with more diverse and recent customer data to improve model robustness.
- Further fine-tuning of models for even better performance in real-world applications.

## Contact
For any questions or suggestions regarding this project, please feel free to reach out via mohammedsameer@arizona.edu.
