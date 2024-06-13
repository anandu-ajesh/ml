import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the dataset
db = pd.read_csv('diabetes.csv')
X = db.drop('Outcome', axis=1)
y = db['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=66)

# Initialize lists to store accuracy values
estimators = range(10, 201, 10)
train_accuracy = []
test_accuracy = []

# Train and evaluate RandomForestClassifier with different numbers of estimators
for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=0)
    rf.fit(X_train, y_train)
    train_accuracy.append(rf.score(X_train, y_train))
    test_accuracy.append(rf.score(X_test, y_test))

# Plot the accuracy values
plt.figure(figsize=(10, 5))
plt.plot(estimators, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(estimators, test_accuracy, label='Testing Accuracy', marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('RandomForest Classifier Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Train the RandomForestClassifier with a specific number of estimators
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

# Print training and testing accuracy
print(f"Train Accuracy: {rf.score(X_train, y_train)}")
print(f"Test Accuracy: {rf.score(X_test, y_test)}")

# Apply a sample test-data and predict the result
new = [[1, 89, 56, 22, 98, 27.4, 0.123, 23]]
prediction = rf.predict(new)
print(f'Prediction: {rf.predict(new)}')
