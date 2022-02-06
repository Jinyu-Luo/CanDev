#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


## Investigate the subgroup: Chinese


# In[3]:


features = pd.read_csv('Chinese2.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['no', 'notSure', 'visible_minority', 'anscount'], axis=1)
features = features.dropna()
features = features.iloc[: , 1:]
features = features[['Harassment', 'Ethical.workplace',
                     'Senior.management','A.safe.and.healthy.workplace', 
                'Duty.to.accommodate', 'yes']]


# In[4]:


features = pd.get_dummies(features)


# In[5]:


# Labels are the values we want to predict
labels = np.array(features['yes'])


# In[6]:


features= features.drop('yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)


# In[7]:


print(feature_list)


# In[8]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[9]:


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[10]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[11]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[12]:


mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[13]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=20)
important_indices = [feature_list.index('Harassment'), feature_list.index('Ethical.workplace'),feature_list.index('Senior.management'), feature_list.index('A.safe.and.healthy.workplace'),  feature_list.index('Duty.to.accommodate')]
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

importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Harassment', 'Ethical.workplace',
                     'Senior.management','A.safe.and.healthy.workplace', 
                'Duty.to.accommodate']

plt.style.use('fivethirtyeight')
x_values = list(range(len(feature_list_import)))
plt.barh( x_values, importances_import, align = 'center')
plt.yticks(x_values, labels = feature_list_import)




plt.title('Variable Importances For Chinese')
plt.ylabel('Variable')
plt.xlabel('Importance')
plt.show()


# In[ ]:


## Investigate the subgroup: Black


# In[14]:


features = pd.read_csv('Black2.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['no', 'notSure', 'visible_minority', 'anscount'], axis=1)
features = features.dropna()
features = features.iloc[: , 1:]


features = features[['Use.of.official.languages', 'Duty.to.accommodate',
                     'Performance.management','Work.life.balance.and.workload', 
                'Work.related.stress', 'yes']]


# In[15]:


features = pd.get_dummies(features)


# In[16]:


labels = np.array(features['yes'])


# In[17]:


features= features.drop('yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[18]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)


# In[19]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[20]:


mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[21]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=20)
important_indices = [feature_list.index('Use.of.official.languages'), feature_list.index('Duty.to.accommodate'),feature_list.index('Performance.management'), feature_list.index('Work.life.balance.and.workload'),  feature_list.index('Work.related.stress')]
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

importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Use.of.official.languages', 'Duty.to.accommodate',
                     'Performance.management','Work.life.balance.and.workload', 
                'Work.related.stress']

plt.style.use('fivethirtyeight')
x_values = list(range(len(feature_list_import)))
plt.barh( x_values, importances_import, align = 'center')
plt.yticks(x_values, labels = feature_list_import)




plt.title('Variable Importances For Black')
plt.ylabel('Variable')
plt.xlabel('Importance')
plt.show()


# In[ ]:


## Investigate the subgroup: Indigenous


# In[39]:


features = pd.read_csv('indigenous2020.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['dept_e','No', 'Notsure','Q116', 
                          'Support to resolve pay or other compensation issues', 'IndegenousStatus'], axis=1)
features = features.dropna()
features = features[['Organizational performance', 'Pay or other compensation issues', 
                    'Work-life balance and workload', 'Work-related stress', 
                   'A psychologically healthy workplace', 'Yes']]


# In[40]:


features = pd.get_dummies(features)


# In[41]:


labels = np.array(features['Yes'])


# In[42]:


features= features.drop('Yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)


# In[43]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[44]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[45]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[46]:


mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[47]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=20)
important_indices = [feature_list.index('Organizational performance'), feature_list.index('Pay or other compensation issues'),feature_list.index('Work-life balance and workload'), feature_list.index('Work-related stress'),  feature_list.index('A psychologically healthy workplace')]
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

importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Organizational performance', 'Pay or other compensation issues', 
                    'Work-life balance and workload', 'Work-related stress', 
                   'A psychologically healthy workplace']

plt.style.use('fivethirtyeight')
x_values = list(range(len(feature_list_import)))
plt.barh( x_values, importances_import, align = 'center')
plt.yticks(x_values, labels = feature_list_import)




plt.title('Variable Importances For Indigenous People')
plt.ylabel('Variable')
plt.xlabel('Importance')
plt.show()


# In[ ]:


## Investigate the subgroup: Non Indigenous


# In[31]:


features = pd.read_csv('nonindigenous2020.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['dept_e','No', 'Notsure','Q116', 
                          'Support to resolve pay or other compensation issues', 'IndegenousStatus'], axis=1)
features = features.dropna()
features = features[['Organizational performance', 'Pay or other compensation issues', 
                    'Discrimination', 'Work-related stress', 
                   'Duty to accommodate','Yes']]


# In[32]:


features = pd.get_dummies(features)


# In[33]:


labels = np.array(features['Yes'])


# In[34]:


features= features.drop('Yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[35]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[36]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[37]:


mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[38]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=20)
important_indices = [feature_list.index('Organizational performance'), feature_list.index('Pay or other compensation issues'),feature_list.index('Discrimination'), feature_list.index('Work-related stress'),  feature_list.index('Duty to accommodate')]
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

importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Organizational performance', 'Pay or other compensation issues', 
                    'Discrimination', 'Work-related stress', 
                   'Duty to accommodate']

plt.style.use('fivethirtyeight')
x_values = list(range(len(feature_list_import)))
plt.barh( x_values, importances_import, align = 'center')
plt.yticks(x_values, labels = feature_list_import)




plt.title('Variable Importances For Non Indigenous People')
plt.ylabel('Variable')
plt.xlabel('Importance')
plt.show()


# In[ ]:


## Investigate the subgroup: People with Disability


# In[49]:


features = pd.read_csv('disabilityY2020.csv')
#features = features.dropna()
print('The shape of our features is:', features.shape)
features = features.drop(['dept_e','No', 'Notsure','Q118', 
                          'Support to resolve pay or other compensation issues', 'DisabilityStatus'], axis=1)
features = features.dropna()
features = features[['Organizational performance', 'Pay or other compensation issues', 
                    'Employee Engagement', 'Work-related stress', 
                   'Discrimination', 'Yes']]


# In[50]:


features = pd.get_dummies(features)
labels = np.array(features['Yes'])
features= features.drop('Yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)


# In[51]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[52]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[53]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=20)
important_indices = [feature_list.index('Organizational performance'), feature_list.index('Pay or other compensation issues'),feature_list.index('Employee Engagement'), feature_list.index('Work-related stress'),  feature_list.index('Discrimination')]
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

importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Organizational performance', 'Pay or other compensation issues', 
                    'Employee Engagement', 'Work-related stress', 
                   'Discrimination']

plt.style.use('fivethirtyeight')
x_values = list(range(len(feature_list_import)))
plt.barh( x_values, importances_import, align = 'center')
plt.yticks(x_values, labels = feature_list_import)




plt.title('Variable Importances For People With Disability')
plt.ylabel('Variable')
plt.xlabel('Importance')
plt.show()


# In[ ]:


## Investigate the subgroup: Not a People with Diability


# In[78]:


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


# In[79]:


features = pd.get_dummies(features)
labels = np.array(features['Yes'])
features= features.drop('Yes', axis = 1)

feature_list = list(features.columns)
features = np.array(features)


# In[80]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[81]:


predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[82]:


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=20)
important_indices = [feature_list.index('Organizational performance'), feature_list.index('Work-life balance and workload'),feature_list.index('Harassment'), feature_list.index('Work-related stress'),  feature_list.index('A safe and healthy workplace')]
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

importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Organizational performance', 'Work-life balance and workload', 
                   'Harassment', 'Work-related stress', 
                   'A safe and healthy workplace']

plt.style.use('fivethirtyeight')
x_values = list(range(len(feature_list_import)))
plt.barh( x_values, importances_import, align = 'center')
plt.yticks(x_values, labels = feature_list_import)




plt.title('Variable Importances For People without Disability')
plt.ylabel('Variable')
plt.xlabel('Importance')
plt.show()


# In[ ]:


## Investigate the subgroup: Female


# In[54]:


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


# In[55]:


# label and feature
labels = np.array(features['Yes'])
features= features.drop('Yes', axis = 1)
feature_list = list(features.columns)
features = np.array(features)


# In[58]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[59]:


# train
rf = RandomForestRegressor(n_estimators = 1000, random_state = 20)
rf.fit(train_features, train_labels)


# In[60]:


# test
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[61]:


# variable importance
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[62]:


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


# In[63]:


#for plot
importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Organizational performance', 'Pay or other compensation issues', 'Employee Engagement', 'Work-related stress', 'Job fit and development']
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
x_values = list(range(len(feature_list_import)))
plt.barh( x_values, importances_import, align = 'center')
plt.yticks(x_values, labels = feature_list_import)
plt.ylabel('Variable'); plt.xlabel('Importance'); plt.title('Variable Importances For Female');


# In[ ]:


## Investigate the subgroup: Male


# In[64]:


# Read in data and keep needed columns
features = pd.read_csv('male2020.csv')
features = features.dropna()
features = features.drop(['gender'], axis = 1)
features = features.drop(['No'], axis = 1)
features = features.drop(['Notsure'], axis = 1)
features = features.drop(['anscount'], axis = 1)
features = features.iloc[: , 1:]


# In[65]:


# create one hot vector
features = pd.get_dummies(features)


# In[66]:


# label and feature
labels = np.array(features['Yes'])
features= features.drop('Yes', axis = 1)
feature_list = list(features.columns)
features = np.array(features)


# In[67]:


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 20)


# In[68]:


# train
rf = RandomForestRegressor(n_estimators = 1000, random_state = 20)
rf.fit(train_features, train_labels)


# In[69]:


# test
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[70]:


# variable importance
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[71]:


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


# In[72]:


# plot
importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Employee Engagement', 'Work-related stress', 'Use of official languages', 'Organizational performance', 'Job fit and development']
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
x_values = list(range(len(feature_list_import)))
plt.barh( x_values, importances_import, align = 'center')
plt.yticks(x_values, labels = feature_list_import)
plt.ylabel('Variable'); plt.xlabel('Importance'); plt.title('Variable Importances For Male');


# In[ ]:




