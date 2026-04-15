# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

# Step 2: Load IRIS dataset
iris = datasets.load_iris()

# Take only 2 features for visualization
X = iris.data[:, :2]
y = iris.target

# Use only two classes (binary classification)
X = X[y != 2]
y = y[y != 2]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Initialize model and train
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Step 5: Get hyperplane parameters
w = model.coef_[0]
b = model.intercept_[0]

print("Weights (w):", w)
print("Bias (b):", b)

print("\nSupport Vectors:\n", model.support_vectors_)

# Step 6: Plot graph
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')

# Plot support vectors
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100,
    facecolors='none',
    edgecolors='k',
    label='Support Vectors'
)

# Create decision boundary
x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Decision boundary
x1 = -(w[0] * x0 + b) / w[1]

# Margins
x1_p = -(w[0] * x0 + b - 1) / w[1]
x1_n = -(w[0] * x0 + b + 1) / w[1]

# Plot lines
plt.plot(x0, x1, 'k-', label='Decision Boundary')
plt.plot(x0, x1_p, 'r--', label='Positive Margin')
plt.plot(x0, x1_n, 'b--', label='Negative Margin')

plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Hyperplane and Support Vectors")

plt.show()

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)
