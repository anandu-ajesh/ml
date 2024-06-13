import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

db = pd.read_csv('diabetes.csv')
X = db.drop('Outcome', axis=1)
y = db['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=66)

estimators = range(10, 201, 10)
train_accuracy = []
test_accuracy = []

for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=0)
    rf.fit(X_train, y_train)
    train_accuracy.append(rf.score(X_train, y_train))
    test_accuracy.append(rf.score(X_test, y_test))
    
plt.figure(figsize=(10, 5))
plt.plot(estimators, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(estimators, test_accuracy, label='Testing Accuracy', marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('RandomForest Classifier Accuracy')
plt.legend()
plt.grid(True)
plt.show()

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)


print(f"Train Accuracy: {rf.score(X_train, y_train)}")
print(f"Test Accuracy: {rf.score(X_test, y_test)}")


new = pd.DataFrame([[1, 89, 56, 22, 98, 27.4, 0.123, 23]], columns=X.columns)
prediction = rf.predict(new)
print(f'Prediction: {prediction}')