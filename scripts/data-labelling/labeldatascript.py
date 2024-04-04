import sys
import pandas as pd
import ollama


def ask_ollama(text):
    response = ollama.chat(model='llama2', messages=[
        {
            'role': 'user',
            'content': f'Judge if following reddit post is about computer science career, is it a positive, negative, neutral or neither if not about career?: "{text}" - answer in 1 word',
        },
    ])
    return response['message']['content']


def determine_sentiments(data):
    sentiments = []
    for _, row in data.iterrows():
        combined_text = f"{row['Title']} {row['Content']}"
        sentiment = ask_ollama(combined_text)

       if sentiment.lower() == "neutral":
            sentiments.append('Neutral')
        elif sentiment.lower() == 'positive':
            sentiments.append('positive')
        elif sentiment.lower() == 'negative':
            sentiments.append('Negative')
        else:
            sentiments.append('Neither')

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
