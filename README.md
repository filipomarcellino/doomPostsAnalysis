# GOAL
Sentiment Analysis of cs reddit postings from 2019 to 2024

## WorkFlow

### collect-subreddit-posts.py
Input: client_ID, client_secret, user_agent, subreddit_list

Output: csv files containing posts from each subreddit

### clean-posts-data.py
Input: filenames of the csv files containing unfiltered posts

Output: 2 csv files
all_subreddits.csv (training data from all subreddits other than r/cs_majors), analysis

### generate-labels.py
Input: all_subreddits.csv training data

Output: all_subreddits_labelled.csv
Note script assumes you have proper input and output directory folders as specified in code. (need these to run)

### generate-TFDIF-features.py
Input: all_subreddits_labelled.csv

Output: 3 different models in .joblib file extensions, cMats

### analyze-target_subreddit.py
Input: analysic.csv

Output: p values and sentiment regression graphs.



## Group: 
Liam, Taras, Filip






