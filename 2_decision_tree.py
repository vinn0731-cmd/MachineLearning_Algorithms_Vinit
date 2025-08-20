# 2_decision_tree.py
# This script demonstrates a Decision Tree model on the Titanic dataset.

# --- Step 1: Data Loading and Preprocessing ---
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create a sample Titanic dataset (since there's no built-in one)
data = {'PassengerId': range(1, 11),
        'Survived': [0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
        'Pclass': [3, 1, 3, 1, 3, 2, 3, 3, 1, 2],
        'Age': [22, 38, 26, 35, 35, 27, 2, 27, 14, 4]}
titanic_df = pd.DataFrame(data)

# Define features (X) and target (y)
X = titanic_df[['Pclass', 'Age']]
y = titanic_df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Data loaded and split into training and testing sets.")
print(f"Training data shape: {X_train.shape}")
print("-" * 30)

# --- Step 2: Training the Model ---
# Initialize the Decision Tree Classifier model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)
print("Decision Tree model training complete.")
print("-" * 30)

# --- Step 3: Making Predictions ---
# Make predictions on the test set
predictions = model.predict(X_test)
print("Predictions complete.")
print("-" * 30)

# --- Step 4: Evaluating and Displaying the Output ---
# Calculate the accuracy and generate a classification report
accuracy = accuracy_score(y_test, predictions)
class_report = classification_report(y_test, predictions, target_names=['Did not survive', 'Survived'])

print("Model Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(class_report)
