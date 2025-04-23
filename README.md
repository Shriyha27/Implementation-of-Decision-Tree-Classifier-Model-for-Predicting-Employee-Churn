# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.
```

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:
# Data Head:
![Screenshot 2025-04-23 085236](https://github.com/user-attachments/assets/c40b8c07-60ab-4b62-a9ab-f4416421a503)


# Dataset info :
![Screenshot 2025-04-23 085244](https://github.com/user-attachments/assets/f9017292-c7bb-4747-a882-c990a19dec5b)


# Null Dataset:
![Screenshot 2025-04-23 085226](https://github.com/user-attachments/assets/a96dd5f1-c3ce-434e-b00c-07c091222156)


# Values count in left column:
![Screenshot 2025-04-23 085218](https://github.com/user-attachments/assets/10597973-f0d9-48c6-b0a6-661f35072e9e)


# Dataset transformed head:
![Screenshot 2025-04-23 085212](https://github.com/user-attachments/assets/05893ce5-d2fa-4ded-9cb3-125e9183b394)


# x.head:
![Screenshot 2025-04-23 085206](https://github.com/user-attachments/assets/6e380388-6635-4db6-ae59-4a40e57ac959)


# Accuracy:
![Screenshot 2025-04-23 085200](https://github.com/user-attachments/assets/156b25a5-e2f5-468d-b6a9-13ccd77fd1d0)


# Data prediction:
![Screenshot 2025-04-23 085150](https://github.com/user-attachments/assets/3ef2a510-29f4-4a42-85ec-5b375aaa9423)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
