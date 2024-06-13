import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

diabetes=pd.read_csv('diabetes.csv')
diabetes.columns= ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

X=diabetes.loc[:,diabetes.columns!='Outcome']
y=diabetes['Outcome']

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=66)

training_accuracy=[]
testing_accuracy=[]

for n_neighbors in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    training_accuracy.append(knn.score(X_train,y_train))
    testing_accuracy.append(knn.score(X_test,y_test))

plt.plot(range(1,11),training_accuracy,label="training_accuracy")
plt.plot(range(1,11),testing_accuracy,label="testing_accuracy")
plt.xlabel("no of neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()

knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
new=pd.DataFrame([[1, 89, 66, 23, 94, 28.1, 0.167, 21]],columns=X.columns)
print("prediction",knn.predict(new))
print(f"acc of traing:{knn.score(X_train,y_train)}")
print(f"acc of testing:{knn.score(X_test,y_test)}")

    
