# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 21:31:49 2017

@author: SRIJANPABBI
"""

#%%
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#%%
# Importing the dataset
dataset_test = pd.read_csv('test.csv')
dataset_train = pd.read_csv('train.csv')
dataset_train.head()
dataset_train.info()
#%%
dataset_train = dataset_train.drop(['PassengerId','Name','Ticket','Cabin','Fare'], axis = 1)
dataset_test= dataset_test.drop(['PassengerId','Name','Ticket','Cabin','Fare'], axis = 1)
#%%
#X_test = dataset_test.iloc[:, [0,1,3,4,5,6,8,10]].values
##y_test = dataset_test.iloc[:, 1].values
#X_train = dataset_train.iloc[:, [0,2,4,5,6,7,9,11]].values
#y_train = dataset_train.iloc[:, 1].values

# Taking care of missing data
# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
dataset_train["Embarked"] = dataset_train["Embarked"].fillna("S")
dataset_train["Age"] = dataset_train["Age"].fillna(30)
dataset_test["Embarked"] = dataset_test["Embarked"].fillna("S")
dataset_test["Age"] = dataset_test["Age"].fillna(30)
dataset_test["Pclass"] = dataset_test["Pclass"].fillna(3)

#%%
X_train = dataset_train.iloc[:,1:].values
y_train = dataset_train.iloc[:,0].values
X_test = dataset_test.values
#%%
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
X_train[:,0] = le1.fit_transform(X_train[:,0])
X_test[:,0] = le2.fit_transform(X_test[:,0])
le3 = LabelEncoder()
le4 = LabelEncoder()
X_train[:,1] = le3.fit_transform(X_train[:,1])
X_test[:,1] = le4.fit_transform(X_test[:,1])
le5 = LabelEncoder()
le6 = LabelEncoder()
X_train[:,-1] = le5.fit_transform(X_train[:,-1])
X_test[:,-1] = le6.fit_transform(X_test[:,-1])

#%%

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_test)
X_test = imputer.transform(X_test)

#%%
ohe1 = OneHotEncoder(categorical_features = [0,5])
X_train = ohe1.fit_transform(X_train).toarray()
ohe2 = OneHotEncoder(categorical_features = [0,5])
X_test = ohe2.fit_transform(X_test).toarray()

#%%
# dummy vab trap removal
#X_train = X_train[:,[1,2,4,5,6,7,8,9,10]]
#X_test = X_test[:,[1,2,4,5,6,7,8,9,10]]
X_train = X_train[:,[1,2,4,5,6,7,8,9,]]
X_test = X_test[:,[1,2,4,5,6,7,8,9]]

#%%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#%%
# Predicting the Test set results
y_pred1 = classifier.predict(X_test)
y_pred = (y_pred1>0.5)
final = y_pred.astype(int)
final = np.array(final)
final = final.ravel()
dataset_test = pd.read_csv('test.csv')
dataset_train = pd.read_csv('train.csv')
d = {'PassengerId' : dataset_test.iloc[:,0].values, 'Survived':final}
dataframe = pd.DataFrame(data = d)
dataframe.to_csv("RF_predictions.csv")
#%%
df = pd.read_csv("gender_submission.csv")
y_test = df.iloc[:,1].values
#%%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
