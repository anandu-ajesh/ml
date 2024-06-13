import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

diabetes = pd.read_csv('diabetes.csv')

X = diabetes.loc[:, diabetes.columns!= 'Outcome']
y = diabetes['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)

# Initialize lists to store training and testing accuracy
training_accuracy = []
test_accuracy = []

# Try SVM with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    sv = SVC(kernel=kernel)
    sv.fit(X_train, y_train)
    training_accuracy.append(sv.score(X_train, y_train))
    test_accuracy.append(sv.score(X_test, y_test))

# Plot training and testing accuracy
plt.plot(kernels, training_accuracy, label="Training Accuracy")
plt.plot(kernels, test_accuracy, label="Testing Accuracy")
plt.xlabel("Kernel")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Apply SVM with rbf kernel on a sample test data
sv = SVC(kernel='rbf')
sv.fit(X_train, y_train)
new = np.array([[1, 89, 66, 23, 94, 28.1, 0.167, 21]])
prediction = sv.predict(new)
print("Prediction: ", prediction)
print('Accuracy of svm classifier on training set: {:.2f}'.format(sv.score(X_train, y_train)))
print('Accuracy of svm classifier on test set: {:.2f}'.format(sv.score(X_test, y_test)))