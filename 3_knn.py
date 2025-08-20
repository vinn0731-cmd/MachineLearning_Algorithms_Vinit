# 3_knn.py
# This script demonstrates a K-Nearest Neighbors model on the Digits dataset.

# --- Step 1: Data Loading and Preprocessing ---
# Import necessary libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Digits dataset (a dataset of handwritten digits)
print("Loading the Digits dataset...")
digits = load_digits()
X = digits.data
y = digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets.")
print("-" * 30)

# --- Step 2: Training the Model ---
# Initialize the KNN model with 5 neighbors
print("Initializing and training the KNN model...")
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)
print("KNN model training complete.")
print("-" * 30)

# --- Step 3: Making Predictions ---
# Make predictions on the test set
print("Making predictions on the test set...")
predictions = model.predict(X_test)
print("Predictions complete.")
print("-" * 30)

# --- Step 4: Evaluating and Displaying the Output ---
# Calculate the accuracy and generate a classification report
accuracy = accuracy_score(y_test, predictions)
class_report = classification_report(y_test, predictions, target_names=[str(i) for i in range(10)])

print("Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)
