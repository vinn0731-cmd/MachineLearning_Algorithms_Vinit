# 5_random_forest.py
# This script demonstrates a Random Forest model on the Wine dataset.

# --- Step 1: Data Loading and Preprocessing ---
# Import necessary libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Wine dataset
print("Loading the Wine dataset...")
wine = load_wine()
X = wine.data
y = wine.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets.")
print("-" * 30)

# --- Step 2: Training the Model ---
# Initialize the Random Forest Classifier with 100 trees
print("Initializing and training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
print("Random Forest model training complete.")
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
class_report = classification_report(y_test, predictions, target_names=wine.target_names)

print("Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)
