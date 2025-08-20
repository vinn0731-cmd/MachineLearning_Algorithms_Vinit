```
# MachineLearning_Algorithms_Vinit
Python Script Implementation with five learning Algorithms
Loading the Iris dataset...
Data split into training and testing sets.
Training data shape: (120, 4)
Testing data shape: (30, 4)
------------------------------
Initializing and training the Logistic Regression model...
Model training complete.
------------------------------
Making predictions on the test set...
Predictions complete.
------------------------------
Model Evaluation:
Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30


First 10 Predicted vs. Actual values:
Predicted: versicolor, Actual: versicolor
Predicted: setosa, Actual: setosa
Predicted: virginica, Actual: virginica
Predicted: versicolor, Actual: versicolor
Predicted: versicolor, Actual: versicolor
Predicted: setosa, Actual: setosa
Predicted: versicolor, Actual: versicolor
Predicted: virginica, Actual: virginica
Predicted: versicolor, Actual: versicolor
Predicted: versicolor, Actual: versicolor
```


```
Data loaded and split into training and testing sets.
Training data shape: (7, 2)
------------------------------
Decision Tree model training complete.
------------------------------
Predictions complete.
------------------------------
Model Evaluation:
Accuracy: 0.67

Classification Report:
                 precision    recall  f1-score   support

Did not survive       0.00      0.00      0.00         1
       Survived       0.67      1.00      0.80         2

       accuracy                           0.67         3
      macro avg       0.33      0.50      0.40         3
   weighted avg       0.44      0.67      0.53         3

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
```


```
Loading the Digits dataset...
Data split into training and testing sets.
------------------------------
Initializing and training the KNN model...
KNN model training complete.
------------------------------
Making predictions on the test set...
Predictions complete.
------------------------------
Model Evaluation:
Accuracy: 0.9861

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        33
           1       1.00      1.00      1.00        28
           2       1.00      1.00      1.00        33
           3       1.00      1.00      1.00        34
           4       0.98      1.00      0.99        46
           5       0.98      0.96      0.97        47
           6       0.97      1.00      0.99        35
           7       1.00      0.97      0.99        34
           8       1.00      1.00      1.00        30
           9       0.95      0.95      0.95        40

    accuracy                           0.99       360
   macro avg       0.99      0.99      0.99       360
weighted avg       0.99      0.99      0.99       360
```


```
Loading the Breast Cancer dataset...
Data split into training and testing sets.
------------------------------
Initializing and training the SVM model...
SVM model training complete.
------------------------------
Making predictions on the test set...
Predictions complete.
------------------------------
Model Evaluation:
Accuracy: 0.9561

Classification Report:
              precision    recall  f1-score   support

   malignant       0.97      0.91      0.94        43
      benign       0.95      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114
```
