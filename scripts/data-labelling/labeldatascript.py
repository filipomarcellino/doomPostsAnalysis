import sys
import pandas as pd
import ollama


def ask_ollama(text):
    response = ollama.chat(model='llama2', messages=[
        {
            'role': 'user',
            'content': f'Does this post have a positive, negative or neutral outlook on computer science career? "{text}" - answer in 1 word',
        },
    ])
    return response['message']['content']


def determine_sentiments(data):
    sentiments = []
    for _, row in data.iterrows():
        title_sentiment = ask_ollama(row['Title'])
        content_sentiment = ask_ollama(row['Content'])

        if 'positive' in title_sentiment.lower() or 'positive' in content_sentiment.lower():
            sentiments.append('Positive')
        elif 'negative' in title_sentiment.lower() and 'negative' in content_sentiment.lower():
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')

    return sentiments


def main(data_path):
    data = pd.read_csv(data_path)

    data['Sentiment'] = determine_sentiments(data)

    output_file_path = f"{data_path}_labelled.csv"
    data.to_csv(output_file_path, index=False)
    print("Done.")


if __name__ == "__main__":
    file = sys.argv[1]
    main(file)
