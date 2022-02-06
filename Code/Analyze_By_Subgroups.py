#!/usr/bin/env python
# coding: utf-8

# In[35]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[ ]:


## Investigate the subgroup: Chinese


# In[59]:


features = pd.read_csv('Chinese2.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['no', 'notSure', 'visible_minority', 'anscount'], axis=1)
features = features.dropna()
features = features.iloc[: , 1:]
features = features[['Harassment', 'Ethical.workplace',
                     'Senior.management','A.safe.and.healthy.workplace', 
                'Duty.to.accommodate', 'yes']]


# In[60]:


features = pd.get_dummies(features)


# In[61]:


# Labels are the values we want to predict
labels = np.array(features['yes'])


# In[62]:


features= features.drop('yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)


# In[63]:


print(feature_list)


# In[64]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[65]:


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[66]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[67]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[68]:


mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[69]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[70]:


plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[ ]:


## Investigate the subgroup: Black


# In[ ]:


features = pd.read_csv('Black2.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['no', 'notSure', 'visible_minority', 'anscount'], axis=1)
features = features.dropna()
features = features.iloc[: , 1:]


features = features[['Use.of.official.languages', 'Duty.to.accommodate',
                     'Performance.management','Work.life.balance.and.workload', 
                'Work.related.stress', 'yes']]


# In[ ]:


features = pd.get_dummies(features)


# In[ ]:


labels = np.array(features['yes'])


# In[ ]:


features= features.drop('yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[ ]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)


# In[ ]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[ ]:


mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[ ]:


plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[ ]:


## Investigate the subgroup: Indigenous


# In[ ]:


features = pd.read_csv('indigenous2020.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['dept_e','No', 'Notsure','Q116', 
                          'Support to resolve pay or other compensation issues', 'IndegenousStatus'], axis=1)
features = features.dropna()
features = features[['Organizational performance', 'Pay or other compensation issues', 
                    'Work-life balance and workload', 'Work-related stress', 
                   'A psychologically healthy workplace', 'Yes']]


# In[ ]:


features = pd.get_dummies(features)


# In[ ]:


labels = np.array(features['Yes'])


# In[ ]:


features= features.drop('Yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)


# In[ ]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[ ]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[ ]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[ ]:


mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[ ]:


plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[ ]:


## Investigate the subgroup: Non Indigenous


# In[ ]:


features = pd.read_csv('nonindigenous2020.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['dept_e','No', 'Notsure','Q116', 
                          'Support to resolve pay or other compensation issues', 'IndegenousStatus'], axis=1)
features = features.dropna()
features = features[['Organizational performance', 'Pay or other compensation issues', 
                    'Discrimination', 'Work-related stress', 
                   'Duty to accommodate','Yes']]


# In[ ]:


features = pd.get_dummies(features)


# In[ ]:


labels = np.array(features['Yes'])


# In[ ]:


features= features.drop('Yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[ ]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[ ]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[ ]:


mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[ ]:


plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[ ]:


## Investigate the subgroup: People with Diability


# In[ ]:


features = pd.read_csv('disabilityY2020.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['dept_e','No', 'Notsure','Q118', 
                          'Support to resolve pay or other compensation issues', 'DisabilityStatus'], axis=1)
features = features.dropna()
features = features[['Organizational performance', 'Pay or other compensation issues', 
                    'Employee Engagement', 'Work-related stress', 
                   'Discrimination', 'Yes']]


# In[ ]:


features = pd.get_dummies(features)
labels = np.array(features['Yes'])
features= features.drop('Yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)


# In[ ]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[ ]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[ ]:


## Investigate the subgroup: Not a People with Diability


# In[ ]:


features = pd.read_csv('disabilityN2020.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['dept_e','No', 'Notsure','Q118', 'Support to resolve pay or other compensation issues', 
                          'DisabilityStatus'], axis=1)
features = features.dropna()
features = features[['Organizational performance', 'Work-life balance and workload', 
                   'Harassment', 'Work-related stress', 
                   'A safe and healthy workplace',
                     'Yes']]


# In[ ]:


features = pd.get_dummies(features)
labels = np.array(features['Yes'])
features= features.drop('Yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)


# In[ ]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[ ]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[ ]:


## Investigate the subgroup: Female


# In[ ]:


# Read in data and keep needed columns
features = pd.read_csv('female2020.csv')
features = features.dropna()
features = features.drop(['gender'], axis = 1)
features = features.drop(['No'], axis = 1)
features = features.drop(['Notsure'], axis = 1)
features = features.drop(['anscount'], axis = 1)
features = features.iloc[: , 1:]
# create one hot vector
features = pd.get_dummies(features)


# In[ ]:


# label and feature
labels = np.array(features['Yes'])
features= features.drop('Yes', axis = 1)
feature_list = list(features.columns)
features = np.array(features)


# In[ ]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 20


# In[ ]:


# train
rf = RandomForestRegressor(n_estimators = 1000, random_state = 20)
rf.fit(train_features, train_labels)


# In[ ]:


# test
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


# variable importance
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[ ]:


# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=20)
important_indices = [feature_list.index('Organizational performance'), feature_list.index('Pay or other compensation issues'),feature_list.index('Employee Engagement'), feature_list.index('Work-related stress'),  feature_list.index('Job fit and development')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions 
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


#for plot
importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Organizational performance', 'Pay or other compensation issues', 'Employee Engagement', 'Work-related stress', 'Job fit and development']
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
x_values = list(range(len(importances_import)))
plt.bar(x_values, importances_import, orientation = 'vertical')
plt.xticks(x_values, feature_list_import, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[ ]:


## Investigate the subgroup: Male


# In[ ]:


# Read in data and keep needed columns
features = pd.read_csv('male2020.csv')
features = features.dropna()
features = features.drop(['gender'], axis = 1)
features = features.drop(['No'], axis = 1)
features = features.drop(['Notsure'], axis = 1)
features = features.drop(['anscount'], axis = 1)
features = features.iloc[: , 1:]


# In[ ]:


# create one hot vector
features = pd.get_dummies(features)


# In[ ]:


# label and feature
labels = np.array(features['Yes'])
features= features.drop('Yes', axis = 1)
feature_list = list(features.columns)
features = np.array(features)


# In[ ]:


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 20)


# In[ ]:


# train
rf = RandomForestRegressor(n_estimators = 1000, random_state = 20)
rf.fit(train_features, train_labels)


# In[ ]:


# test
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


# variable importance
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[ ]:


# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=20)
important_indices = [feature_list.index('Employee Engagement'), feature_list.index('Work-related stress'),feature_list.index('Use of official languages'), feature_list.index('Organizational performance'),  feature_list.index('Job fit and development')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions 
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


# plot
importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Employee Engagement', 'Work-related stress', 'Use of official languages', 'Organizational performance', 'Job fit and development']
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
x_values = list(range(len(importances_import)))
plt.bar(x_values, importances_import, orientation = 'vertical')
plt.xticks(x_values, feature_list_import, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

