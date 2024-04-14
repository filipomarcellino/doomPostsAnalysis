## Using PRAW to collect Reddit posts

### Requirements:
As of writing you must supply credentials to call the Reddit API. There are guides that can help you do this.
https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps
https://praw.readthedocs.io/en/stable/getting_started/authentication.html#oauth

### Usage:
Script callers must supply arguments for:
- client_ID
- client_secret
- user_agent

Script callers may supply arguments for:
- target_subreddit (otherwise, default to built in list of subreddits)

### Output:
This script produces a new csv file {subreddit_name}-posts.csv for any subreddit that data is collected for.
