
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# female

# Read in data and delete unwanted columns
features = pd.read_csv('female2020.csv')
features = features.dropna()
features = features.drop(['gender'], axis = 1)
features = features.drop(['No'], axis = 1)
features = features.drop(['Notsure'], axis = 1)
features = features.drop(['anscount'], axis = 1)
features = features.iloc[: , 1:]

# create one hot vector
features = pd.get_dummies(features)



# seperate label and feature
labels = np.array(features['Yes'])
features= features.drop('Yes', axis = 1)
feature_list = list(features.columns)
features = np.array(features)


# train test split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 20)


# train
rf = RandomForestRegressor(n_estimators = 1000, random_state = 20)
rf.fit(train_features, train_labels)

# test
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# variable importance
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

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

#for plot
importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Organizational performance', 'Pay or other compensation issues', 'Employee Engagement', 'Work-related stress', 'Job fit and development']
%matplotlib inline
plt.style.use('fivethirtyeight')
x_values = list(range(len(importances_import)))
plt.bar(x_values, importances_import, orientation = 'vertical')
plt.xticks(x_values, feature_list_import, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# male

# Read in data and keep needed columns
features = pd.read_csv('male2020.csv')
features = features.dropna()
features = features.drop(['gender'], axis = 1)
features = features.drop(['No'], axis = 1)
features = features.drop(['Notsure'], axis = 1)
features = features.drop(['anscount'], axis = 1)
features = features.iloc[: , 1:]

# create one hot vector
features = pd.get_dummies(features)

# label and feature
labels = np.array(features['Yes'])
features= features.drop('Yes', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 20)

# train
rf = RandomForestRegressor(n_estimators = 1000, random_state = 20)
rf.fit(train_features, train_labels)

# test
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# variable importance
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


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

# plot
importances_import = list(rf_most_important.feature_importances_)
feature_list_import = ['Employee Engagement', 'Work-related stress', 'Use of official languages', 'Organizational performance', 'Job fit and development']
%matplotlib inline
plt.style.use('fivethirtyeight')
x_values = list(range(len(importances_import)))
plt.bar(x_values, importances_import, orientation = 'vertical')
plt.xticks(x_values, feature_list_import, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
