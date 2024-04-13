Group: Liam, Taras, Filip

# GOAL
Sentiment Analysis of cs reddit postings from 2019 to 2024

## WorkFlow

collect-subreddit-posts.py
Input: client_ID, client_secret, user_agent, subreddit_list
Output: csv files containing posts from each subreddit

clean-posts-data.py
Input: filenames of the csv files containing unfiltered posts
Output: 2 csv files
all_subreddits.csv (training data from all subreddits other than r/cs_majors)

generate-labels.py
Input: all_subreddits.csv training data
Output: all_subreddits_labelled.csv
Note script assumes you have proper input and output directory folders as specified in code. (need these to run)

generate-TFDIF-features.py
Input: all_subreddits_labelled.csv
Output: 3 different models in .joblib file extensions

analyze-target_subreddit.py
Input: analysic.csv
Output: p values and sentiment regression graphs.

### Data
big list of CSVs -> stuff from step 1 (api calls)
{all_subreddits, analysis} -> stuff from step 2 (filtering)
{all_subreddits_labeled} -> stuff from step 3 (labeling)
{joblib files, cMats} -> stuff from step 4 (generate TFIDF)
{graphs} -> stuff from step 5 (analysis.py)
