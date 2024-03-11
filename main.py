import praw

reddit = praw.Reddit(client_id='q5zkFdD03ScNaSThk0J8TQ',
                     client_secret='JQjrEA5Y6CnzrbOoUng1oBZ66xqqYQ',
                     user_agent='sentimentAnalysis')
subreddit = reddit.subreddit('cscareerquestions')
top_posts = subreddit.top(time_filter='day', limit=10)
for post in top_posts:
    print(post.title)
