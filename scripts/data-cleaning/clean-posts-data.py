import pandas as pd
import os
import sys


def main():

    # get directory for data
    parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    data_dir = os.path.join(parent_dir, r"data")

    subreddit_list = ['cscareerquestions', 'csMajors', 'cscareerquestionsCAD', 'cscareers',
                      'cscareerquestionsEU', 'developersIndia', 'cscareerquestionsuk', 'computerscience',
                      'cscareerquestionsOCE', 'learnprogramming', 'DevelEire', 'dataengineering',
                      'ITCareerQuestions', 'compsci', 'AskComputerScience', 'ExperiencedDevs', 'webdev']

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

        # Do data cleaning here.

        # Turn into two sets of data here.

        print("\n")

    sys.exit()


if __name__ == "__main__":
    main()