# 1_logistic_regression.py
# This script demonstrates a Logistic Regression model on the Iris dataset.

# --- Step 1: Data Loading and Preprocessing ---
# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
print("Loading the Iris dataset...")
iris = load_iris()
X = iris.data  # The feature data
y = iris.target # The target labels

# Split the data into a training set and a testing set
# A test size of 20% is used (test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets.")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print("-" * 30)

# --- Step 2: Training the Model ---
# Initialize the Logistic Regression model
# The 'max_iter' is set to 200 to ensure convergence
print("Initializing and training the Logistic Regression model...")
model = LogisticRegression(max_iter=200)

# Fit the model to the training data
model.fit(X_train, y_train)
print("Model training complete.")
print("-" * 30)

# --- Step 3: Making Predictions ---
# Use the trained model to make predictions on the test data
print("Making predictions on the test set...")
predictions = model.predict(X_test)
print("Predictions complete.")
print("-" * 30)

# --- Step 4: Evaluating and Displaying the Output ---
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)

# Generate a classification report to see precision, recall, and F1-score
class_report = classification_report(y_test, predictions, target_names=iris.target_names)

print("Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)

# Display the first few predicted vs actual values for a clear comparison
print("\nFirst 10 Predicted vs. Actual values:")
for i in range(10):
    print(f"Predicted: {iris.target_names[predictions[i]]}, Actual: {iris.target_names[y_test[i]]}")
  
