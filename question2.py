# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 11:31:44 2023

@author: farhan
"""

# loading data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



response_file="form-data.csv"
dataframe=pd.read_csv(response_file)
dataframe = dataframe.drop(0)

# Dropping unrelated columns
dataframe=dataframe.drop(columns=['Timestamp','Email Address'])


# Split the "Cumulative Grade Point Average(CGPA)" column into "Minimum CGPA" and "Maximum CGPA" columns
dataframe[['Minimum CGPA', 'Maximum CGPA']] = dataframe['Cumulative Grade Point Average(CGPA)'].str.split('<', expand=True)
dataframe[['Minimum CGPA', 'Maximum CGPA']] = dataframe['Minimum CGPA'].str.split(' to ', expand=True)

dataframe=dataframe.drop('Cumulative Grade Point Average(CGPA)',axis=1)

dataframe['Minimum CGPA'] = pd.to_numeric(dataframe['Minimum CGPA'], errors='coerce')
dataframe['Maximum CGPA'] = pd.to_numeric(dataframe['Maximum CGPA'], errors='coerce')
dataframe[['Minimum CGPA','Maximum CGPA']]=dataframe[['Minimum CGPA','Maximum CGPA']].fillna(0)



dataframe['What was your starting salary range?'] = dataframe['What was your starting salary range?'].str.replace('BDT', '')
dataframe['What was your starting salary range?'] = dataframe['What was your starting salary range?'].str.replace('> ', '')

# Split the "Salary" column into "Minimum Salary" and "Maximum Salary" columns
dataframe[['Minimum Salary', 'Maximum Salary']] = dataframe['What was your starting salary range?'].str.split(' - ', expand=True)

dataframe=dataframe.drop('What was your starting salary range?',axis=1)

dataframe['Minimum Salary'] = dataframe['Minimum Salary'].str.replace('K', '000')

dataframe['Maximum Salary'] = dataframe['Maximum Salary'].str.replace('K', '000')

dataframe[['Minimum Salary','Maximum Salary']]=dataframe[['Minimum Salary','Maximum Salary']].fillna(0)

# Convert the data type of the salary columns

dataframe[['Minimum Salary','Maximum Salary']] = dataframe[['Minimum Salary','Maximum Salary']].astype(int)

dataframe['Average Salary']=(dataframe['Minimum Salary']+dataframe['Maximum Salary'])/2
dataframe=dataframe.drop(columns=['Minimum Salary','Maximum Salary'])


#Need to find corelation and convert categorical columns with encoding
dataframe['Total year now since your graduation  ']=dataframe['Total year now since your graduation  '].str.replace(' years','')
dataframe['Total year now since your graduation  ']=dataframe['Total year now since your graduation  '].str.replace(' year','')


categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns

# Apply Label Encoding to each categorical column
label_encoders = {}
for column in categorical_columns:
    label_encoder = LabelEncoder()
    dataframe[column] = label_encoder.fit_transform(dataframe[column])
    label_encoders[column] = label_encoder

#Finding corelation Between Data
correlation_matrix = dataframe.corr()
correlation_with_target = correlation_matrix['Did you start working as a software engineer after graduation?'].abs().sort_values(ascending=False)

#Selecting features based on corelation Matrix
N = 4
selected_features = correlation_with_target.index[1:N+1].tolist()

#Defining X, Y train test dataset

X = dataframe[selected_features]
y = dataframe['Did you start working as a software engineer after graduation?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print evaluation metrics and relevant features
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_rep)

# Check feature importance (coefficients)
feature_importance = logistic_model.coef_
print("Feature Importance (Coefficients):")
for feature, importance in zip(selected_features, feature_importance[0]):
    print(f"{feature}: {importance}")
    

#Predict data
whether_going_or_not=pd.DataFrame({
        'Were you interested in coding?':[0],
        'Total year now since your graduation  ':[1],
        'What was the name of the company you worked for right after your graduation? if your answer is not write N/A.':[33],
        'Minimum CGPA':[2]
    })

prediction=logistic_model.predict(whether_going_or_not)
if prediction==0:
    print(f"Going to a company to do job is False")
else:
    print(f"Going to a company to do job is True")

    




























