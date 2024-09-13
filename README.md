# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
 
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Nisha.J
RegisterNumber:212223040133  
```
```
import pandas as pd
data=pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## Head:
![365531328-1efcc686-7eab-4cd4-b9c1-0d796c896970](https://github.com/user-attachments/assets/af4eb469-5696-4689-a3ba-f0ddb0d11339)

## Data copy:
![365531438-5be34e0a-f570-4ee9-bae8-397949b419ef](https://github.com/user-attachments/assets/abbcab7d-6c26-4334-8a08-bee7c954ce11)

## Fit Transform:
![365531612-cd8da353-7c20-4b3f-89ad-01e4c50a1a30](https://github.com/user-attachments/assets/405ba332-763e-460d-8c8c-ca83b6da88dd)

## Logistic Regression:
![365529991-3fbdb25e-f2bb-4e3b-b923-6f6d4160071b](https://github.com/user-attachments/assets/bbf4dfd8-475e-49ba-9166-1ec5b5f58c8d)

## Accuracy Score:
![365530158-3f8aa101-3aa1-47fd-b67a-308ab84f0241](https://github.com/user-attachments/assets/32480e35-3c25-4944-9779-ec62c5bbbd1c)

## Confusion Matrix:
![365530286-c9349b35-b1f5-40cc-8f01-7ed2ebed78f7](https://github.com/user-attachments/assets/dd90eb8f-986f-4f2a-8c7c-440ff7e1efdc)

## Classification Report:
![365530415-509d1b75-62ab-4cbb-9bb4-2bd2d1c104a4](https://github.com/user-attachments/assets/f5cc4099-da25-46b2-8b64-fce770b3fc89)

## Prediction:
![365530615-0221633c-d495-4e7d-a46f-18ace64a6931](https://github.com/user-attachments/assets/6c6837b5-7bc9-4d32-96f8-4f338b738d4f)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
