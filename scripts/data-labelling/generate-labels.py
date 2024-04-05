import sys
import pandas as pd
import ollama
import re
import os

model_file ='''
FROM llama2
PARAMETER temperature 0
PARAMETER num_predict 8
'''

def ask_ollama(text):
    response = ollama.chat(model='sentiment', messages=[
        {
            'role': 'user',
            'content': f'Judge if following reddit post is about computer science career, is it a positive, negative, neutral or neither if not about career?: "{text}" - answer in 1 word',
        },
    ])
    return response['message']['content']


def determine_sentiments(data):
    sentiments = []

    print("Generating labels. This may take some time.")

    ollama.create(model='sentiment', modelfile=model_file)

    for _, row in data.iterrows():
        combined_text = f"{row['title']} {row['content']}"
        sentiment = ask_ollama(combined_text)

        sentiment = sentiment.lower()

        # match one of the sentiments with potential whitespace in front of it
        sentiment_pattern = r'\s*\b(negative|neutral|positive)\b\s*'

        # Match the first sentiment
        sentiment_match = (re.match(sentiment_pattern, sentiment))

        # If we found an instance of positive, neutral, negative
        if sentiment_match != None:
            # Set sentiment to this string and strip whitespace that might exist
            sentiment = sentiment_match[0]
            sentiment = sentiment.strip()

        # Add sentiment to list
        if sentiment == "neutral":
            sentiments.append('neutral')
        elif sentiment == 'positive':
            sentiments.append('positive')
        elif sentiment == 'negative':
            sentiments.append('negative')
        else:
            sentiments.append('neither')
    return sentiments


def main():

    # get directory for data folder
    parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    data_dir = os.path.join(parent_dir, r'data')

    # get directory for cleaned data folder
    cleaned_dir = os.path.join(data_dir, r'cleaned')

    # get directory for labeled data folder
    labeled_dir = os.path.join(data_dir, r'labeled')

    # get filepaths
    data_filepath = os.path.join(cleaned_dir, r'all_subreddits.csv')
    output_file_path = os.path.join(labeled_dir, r"all_subreddits_labelled.csv")

    data = pd.read_csv(data_filepath)
    data['sentiment'] = determine_sentiments(data)

    data.to_csv(output_file_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
