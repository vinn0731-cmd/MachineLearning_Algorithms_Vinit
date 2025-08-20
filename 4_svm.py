# 4_svm.py
# This script demonstrates a Support Vector Machine model on the Breast Cancer dataset.

# --- Step 1: Data Loading and Preprocessing ---
# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the Breast Cancer dataset
print("Loading the Breast Cancer dataset...")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets.")
print("-" * 30)

# --- Step 2: Training the Model ---
# Initialize the SVM model with a linear kernel
print("Initializing and training the SVM model...")
model = SVC(kernel='linear')

# Train the model
model.fit(X_train, y_train)
print("SVM model training complete.")
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
class_report = classification_report(y_test, predictions, target_names=['malignant', 'benign'])

print("Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)
