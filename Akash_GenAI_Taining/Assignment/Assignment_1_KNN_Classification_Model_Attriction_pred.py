# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:57:41 2024

@author: akash
"""

import pandas as pd

data = {
    'Age': [29, 35, 40, 28, 45, 25, 50, 30, 37, 26],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Sales Executive', 'Manager', 'Research Scientist', 'Manager', 'Sales Executive', 'Laboratory Technician', 'Research Scientist'],
    'MonthlyIncome': [4800, 6000, 3400, 4300, 11000, 3500, 12000, 5000, 3100, 4500],
    'JobSatisfaction': [3, 4, 2, 3, 4, 1, 4, 2, 2, 3],
    'YearsAtCompany': [4, 8, 6, 3, 15, 2, 20, 5, 9, 2],
    'Attrition': [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

df.head()

print(df.info())


X=df.drop(['JobRole','Attrition'],axis=1)
y=df['Attrition']

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.1,random_state=42)


model=KNeighborsClassifier()
model.fit(X_train,y_train)



y_pred=model.predict(X_test)



print(accuracy_score(y_test, y_pred))
