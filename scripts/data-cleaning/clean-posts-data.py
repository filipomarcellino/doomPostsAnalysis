import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split


def main():

    # get directory for data
    parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    data_dir = os.path.join(parent_dir, r"data")

    subreddit_list = ['csMajors', 'cscareerquestionsCAD', 'cscareerquestionsEU', 'developersIndia', 'cscareerquestionsuk', 'computerscience', 'ExperiencedDevs', 'webdev']

    # Initialize an empty DataFrame to hold all records
    all_subreddits_data = pd.DataFrame()

    # iterate through all the data sources
    for subreddit_name in subreddit_list:

        print(f"Starting work on {subreddit_name}-posts.csv")

        # filename for this CSV file.
        file_name = os.path.join(data_dir, f"{subreddit_name}-posts.csv")

        # Attempt to read file. Skip this subreddit if not found.
        try:
            subreddit_data = pd.read_csv(file_name)
        except FileNotFoundError:
            print(f"{subreddit_name}-posts.csv was not found.\n")
            continue

        # Handle Missing Values - Remove rows where 'Title' or 'Content' is missing.
        subreddit_data.dropna(subset=['title', 'content'], inplace=True)

        # Normalize Text Data - Convert to lowercase and trim whitespaces.
        subreddit_data['title'] = subreddit_data['title'].str.lower().str.strip()
        subreddit_data['content'] = subreddit_data['content'].str.lower().str.strip()

        # Randomly sample 1000 posts from each subreddit
        if len(subreddit_data) >= 1000:
            limited_subreddit_data = subreddit_data.sample(n=1000, random_state=42)
        else:
            limited_subreddit_data = subreddit_data

        # Append to the main DataFrame
        all_subreddits_data = pd.concat([all_subreddits_data, limited_subreddit_data], ignore_index=True)
        print(f"Cleaned {subreddit_name}-posts.csv data and appended to main dataframe")


    # Perform train-validation split
    train_data, validation_data = train_test_split(all_subreddits_data, test_size=0.2, random_state=42)

    # Export to data directory
    train_data.to_csv(os.path.join(data_dir, "cleaned", "all_subreddits_train.csv"), index=False)
    validation_data.to_csv(os.path.join(data_dir, "cleaned", "all_subreddits_validation.csv"), index=False)
    print("Train and validation data exported")

    sys.exit()


if __name__ == "__main__":
    main()