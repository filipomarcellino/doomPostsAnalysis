## Data analysis

### Requirements:
The desired estimator .joblib file in the estimators directory.

### Usage:
Script callers may supply supply an argument for the desired model to be used. Options are {cNB, SVM, GBDT}. If no argument is found, cNB is used by default.

### Output:
This script does linear regression for both positive and negative sentiment proportions (by month) over time and reports the p value. Graphs for how this proportion changes with time are produced along with a histogram of when the data was collected.