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


