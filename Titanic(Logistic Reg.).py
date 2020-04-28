# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:57:50 2020

@author: Abhishek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================================================================
# Setting up working directory
# =============================================================================
os.chdir("C:/Users/Abhishek/Desktop/Data Science/IVY/Python/CLUSTERING/CASE STUDIES/CASE-1")

train = pd.read_csv("train.csv")

# =============================================================================
# Visualization
# =============================================================================
sns.countplot(x = "Survived",hue = "Pclass",data = train)
plt.show()

sns.countplot(x = "Survived",hue = "Sex",data = train)
plt.show()

train.head()

sns.heatmap(train.isnull())
# Here "Cabin" column has large number of null values. So, drop it.
# Also "Age" column has few number of null values. So, We will replace null with mean values.  

# =============================================================================
# Data Cleaning for train data
# =============================================================================
train = train.drop("Cabin",axis = 1)

# Here, Replacing missing Age values with mean Age value of different Pclass
def fills_na(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 38   
        elif Pclass==2:
            return 30;
        else:
            return 25
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(fills_na,axis=1)
train.info()

train = train.dropna(axis=0)

sex = pd.get_dummies(train['Sex'],drop_first = True)

embark = pd.get_dummies(train['Embarked'],drop_first = True)

train = pd.concat([train,sex,embark],axis=1)

train.drop(['Sex','Embarked','PassengerId','Ticket','Name'],axis = 1,inplace = True)

# =============================================================================
# Splitting train data
# =============================================================================
x_train = train.drop('Survived',axis = 1)
y_train = train['Survived']

# =============================================================================
# Data cleaning for test data
# =============================================================================
test = pd.read_csv("test1.csv")

sns.heatmap(test.isnull())
test['Fare'] = test['Fare'].ffill(axis=0)
test.drop(['Cabin','PassengerId','Name','Ticket'],axis = 1,inplace = True)
test.info()

def fills_na(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 41   
        elif Pclass==2:
            return 28;
        else:
            return 24
    else:
        return Age
    
test['Age'] = test[['Age','Pclass']].apply(fills_na,axis=1)
test.info()

sex1 = pd.get_dummies(test['Sex'],drop_first = True)
embark1 = pd.get_dummies(test['Embarked'],drop_first = True)

test = pd.concat([test,sex1,embark1],axis=1)

test.drop(['Sex','Embarked',],axis = 1,inplace = True)
# =============================================================================
# Splitting test data
# =============================================================================
x_test = test.drop('Survived',axis = 1) 
y_test = test['Survived']

# =============================================================================
# Regression 
# =============================================================================
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# =============================================================================
# Evaluating logostic model
# =============================================================================
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(x_test, y_test)))



#Confusion matrix for train

y_pred_train = classifier.predict(x_train)
from sklearn.metrics import confusion_matrix

confusion_matrix_train = confusion_matrix(y_train,y_pred_train)
print(confusion_matrix_train)
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(classifier.score(x_train, y_train)))