import csv
import datetime
import praw

reddit = praw.Reddit(client_id='q5zkFdD03ScNaSThk0J8TQ',
                     client_secret='JQjrEA5Y6CnzrbOoUng1oBZ66xqqYQ',
                     user_agent='sentimentAnalysis')


def read_existing_post_ids(file_name):
    try:
        with open(file_name, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            return {rows[0] for rows in reader if rows}
    except FileNotFoundError:
        return set()


def write_posts_to_csv(posts, file_name):
    """Write post details to a CSV file, avoiding duplicates."""
    existing_ids = read_existing_post_ids(file_name)
    with open(file_name, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Write header if file is new
            writer.writerow(['ID', 'Title', 'Content', 'Creation Date'])
        for post in posts:
            if post.id not in existing_ids:
                creation_date = datetime.datetime.fromtimestamp(post.created_utc, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                content = post.selftext if post.selftext else "N/A"
                writer.writerow([post.id, post.title, content, creation_date])


def fetch_posts_for_year(subreddit, year, limit=100):
    """Fetch posts for a specific year and return a filtered list."""
    posts = []
    for post in subreddit.top(time_filter='all', limit=limit*5):  # Increased limit for a broader fetch
        post_year = datetime.datetime.fromtimestamp(post.created_utc, tz=datetime.timezone.utc).year
        if post_year == year:
            posts.append(post)
            if len(posts) >= limit:
                break
    return posts


def fetch_and_write_top_posts_for_years(subreddit_name, years, limit):
    subreddit = reddit.subreddit(subreddit_name)
    for year in years:
        file_name = f'{subreddit_name}_top_posts_{year}.csv'  # Each year gets its own file
        posts = fetch_posts_for_year(subreddit, year, limit)
        print(f"Fetched {len(posts)} posts for the year {year}.")
        write_posts_to_csv(posts, file_name)
        print(f'Updated {file_name} with top posts from {year}.')


def main():
    subreddit_name = 'cscareerquestions'
    years = [2020, 2021, 2022, 2023, 2024]
    fetch_and_write_top_posts_for_years(subreddit_name, years, 100)
    print('CSV files have been updated with the top 100 posts for each year from 2020 to 2024.')


if __name__ == "__main__":
    main()
