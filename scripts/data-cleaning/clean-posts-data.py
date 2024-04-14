import pandas as pd
import os
import sys


def main():

    # get directory for data
    parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    data_dir = os.path.join(parent_dir, r"data")

    subreddit_list = ['csMajors', 'cscareerquestionsCAD', 'cscareerquestionsEU', 'developersIndia', 'cscareerquestionsuk', 'computerscience', 'ExperiencedDevs', 'webdev']
    analysis_subreddit = 'csMajors'

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
        # This removed around 4000 entries from our dataset
        subreddit_data.dropna(subset=['title', 'content'], inplace=True)

        # Cutoff the text data
        # The title already has a cutoff
        subreddit_data['content'] = subreddit_data['content'].str.slice(0, 1000)

        # Remove the value &#x200B;
        # Around 1200 entries had this
        subreddit_data['content'] = subreddit_data['content'].str.replace(r'&#x200B;', '', regex=True)

        # If this is the analysis subreddit
        if (subreddit_name == analysis_subreddit):
            # Write to csv
            subreddit_data.to_csv(os.path.join(data_dir, "cleaned", "analysis.csv"), index=False)
            print(f'Cleaned {subreddit_name}-posts.csv data. This is the analysis dataset.\n')
        else:
            # Else, append to the main DataFrame
            all_subreddits_data = pd.concat([all_subreddits_data, subreddit_data], ignore_index=True)
            print(f"Cleaned {subreddit_name}-posts.csv data and appended to main dataframe\n")


    # Export to data directory
    all_subreddits_data.to_csv(os.path.join(data_dir, "cleaned", "all_subreddits.csv"), index=False)

    sys.exit()


if __name__ == "__main__":
    main()
