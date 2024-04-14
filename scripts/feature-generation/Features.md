## Estimator creation with supervised learning models.

### Usage:
Script callers may supply any argument to signal that only an estimator for complement Naive Bayes should be created. Calling this script with no arguments will generate estimators for all 3 models {cNB, SVM, GBDT} and may take a long time due to hyperparameter tuning (in some cases 30-40 minutes).

Complement Naive Bayes was generally the best option in most of the trials that were ran.

### Output:
This script produces desired estimators with .joblib file extensions in the estimators directory. Confusion matrices are created for each model used in the graphs directory.