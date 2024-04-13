# Project goal
Sentiment analysis of computer science related reddit postings from 2019 to 2024

## Pipeline

### collect-subreddit-posts.py
Input: client_ID, client_secret, user_agent, (optional: target_subreddit, otherwise defaults to predetermined list)

Output: csv files containing posts from each subreddit


### clean-posts-data.py
Input: none

Output: data/cleaned/all_subreddits.csv and data/cleaned/analysis.csv


### generate-labels.py
Input: None

Output: data/labeled/all_subreddits_labelled.csv

Note: This script assumes you have ollama running and that the data/cleaned directory is accessible from the project root. Read Labels.md for further instructions.


### generate-TFIDF-features.py
Input: Providing no arguments creates all 3 estimators (which may take a long time). Providing any arguments only creates the estimator that was found to perform best on this task.

Output: Desired estimators with .joblib file extensions. Confusion matrices.


### analyze-target_subreddit.py
Input: Estimator to be used for determining sentiment. Options are {cNB, SVM, GBDT} (default = cNB).

Output: p values and sentiment regression graphs.

Note: the desired estimator must exist in the estimators directory.


## Group: 
Liam, Taras, Filip

