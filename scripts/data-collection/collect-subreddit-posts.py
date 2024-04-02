import csv
import datetime
import praw
import sys
import os


def main():
    # Client ID, Secret, and User agent must be supplied as arguments.
    client_ID = sys.argv[1]
    client_secret = sys.argv[2]
    user_agent = sys.argv[3]

    # Target subreddit
    subreddit_list = [sys.argv[4]]

    # If this is not a valid subreddit name, iterate through all
    # the subreddits in the list.
    if len(sys.argv[4]) < 3:
        subreddit_list = ['cscareerquestions', 'csMajors', 'cscareerquestionsCAD', 'cscareers',
                      'cscareerquestionsEU', 'developersIndia', 'cscareerquestionsuk', 'computerscience',
                      'cscareerquestionsOCE', 'learnprogramming', 'DevelEire', 'dataengineering',
                      'ITCareerQuestions', 'compsci', 'AskComputerScience', 'ExperiencedDevs', 'webdev']

    # Setup PRAW auth and subreddit to target.
    reddit = praw.Reddit(client_id=client_ID,
                         client_secret=client_secret,
                         user_agent=user_agent)

    # get directory for data
    parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    data_dir = os.path.join(parent_dir, r"data")

    # for all subreddits

    for subreddit_name in subreddit_list:

        print(f"Collecting data for {subreddit_name}")

        subreddit = reddit.subreddit(subreddit_name)

        # filename for this CSV file
        file_name = os.path.join(data_dir, f"{subreddit_name}-posts.csv")

        # Fetch all the posts for this subreddit
        posts = fetch_posts(subreddit)

        print(f"Your API auth has {(reddit.auth.limits).get('remaining')} requests remaining. These will "
          f"reset at timestamp {(reddit.auth.limits).get('reset_timestamp')} (unix time).")

        # Write to CSV in the Data folder
        write_posts_to_csv(posts, file_name)
        continue

    # after loop is done, exit
    sys.exit()


def fetch_posts(subreddit, limit=None):
    """
    This function takes from:
    - top of {all time, year, month}
    - controversial {all time, year, month}
    - hot
    - rising
    - new

    This function skips:
    - "Top of {week, day, hour}" as these are likely covered by new
    - "Gilded" as these are likely covered by top
    - "Controversial {week, day, hour}" as these are likely covered by new
    """

    # This should be fine as using a numpy array would require
    # knowing the max length of a post if we wanted to avoid using
    # object dtype (which is just python pointers).
    posts = []

    # To track duplicate posts, we will use the (unique) ID for the post.
    # Each ID we use will be stored in the set.
    #
    # Since IDs are small, using a set here instead of storing hashes is fine.
    # We would likely need to store more data with the latter solution.
    duplicate_posts = set()

    # Now, we start collecting data to be stored in posts.
    #
    # Reminder (PRAW docs):
    #   limit – The number of content entries to fetch.
    #   Most of Reddit’s listings contain a maximum of 1000 items
    #   and are returned 100 at a time. This class will automatically
    #   issue all necessary requests (default: 100).

    print("Fetching from top")

    # Grab as many posts as possible from top of all,year,month
    # Each iteration should use at most 10 requests (100 posts each).
    # Thus, we use 30 requests for this operation
    for filter_by in ['all', 'year', 'month']:
        for post in subreddit.top(time_filter=filter_by, limit=limit):
            # only add this post to the list if we have never seen the ID.
            if post.id not in duplicate_posts:
                posts.append(post)
                duplicate_posts.add(post.id)

    print("Fetching from controversial")

    # Grab as many posts as possible from controversial of all,year,month
    # Each iteration should use at most 10 requests (100 posts each).
    # Thus, we use 30 requests for this operation
    for filter_by in ['all', 'year', 'month']:
        for post in subreddit.controversial(time_filter=filter_by, limit=limit):
            # only add this post to the list if we have never seen the ID.
            if post.id not in duplicate_posts:
                posts.append(post)
                duplicate_posts.add(post.id)

    print("Fetching from hot")

    # Grab as many posts as possible from hot
    # This operation should use at most 10 requests (100 posts each)
    for post in subreddit.hot(limit=limit):
        # only add this post to the list if we have never seen the ID.
        if post.id not in duplicate_posts:
            posts.append(post)
            duplicate_posts.add(post.id)

    print("Fetching from rising")

    # Grab as many posts as possible from rising
    # This operation should use at most 10 requests (100 posts each)
    for post in subreddit.rising(limit=limit):
        # only add this post to the list if we have never seen the ID.
        if post.id not in duplicate_posts:
            posts.append(post)
            duplicate_posts.add(post.id)

    print("Fetching from new")

    # Grab as many posts as possible from new
    # This operation should use at most 10 requests (100 posts each)
    for post in subreddit.new(limit=limit):
        # only add this post to the list if we have never seen the ID.
        if post.id not in duplicate_posts:
            posts.append(post)
            duplicate_posts.add(post.id)

    # At this point, all posts should be collected. Return them.
    return posts


def read_existing_post_ids(file_name):
    try:
        with open(file_name, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            return {rows[0] for rows in reader if rows}
    except FileNotFoundError:
        return set()


def write_posts_to_csv(posts, file_name):
    """Write post details to a CSV file, avoiding already recorded data."""
    existing_ids = read_existing_post_ids(file_name)
    try:
        with open(file_name, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # Write header if file is new
                writer.writerow(['ID', 'title', 'content', 'date'])
            for post in posts:
                if post.id not in existing_ids:
                    creation_date = datetime.datetime.fromtimestamp(post.created_utc,
                                                                    tz=datetime.timezone.utc).strftime(
                        '%Y-%m-%d %H:%M:%S')
                    content = post.selftext if post.selftext else "N/A"
                    writer.writerow([post.id, post.title, content, creation_date])

    except FileNotFoundError:
        # subreddit names are always valid file names, so this will only
        # occur if some random garbage was given as the subreddit name.
        print("File operation failed as your subreddit name was invalid.")


if __name__ == "__main__":
    main()
