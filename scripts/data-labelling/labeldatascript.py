import sys
import pandas as pd
import ollama


def ask_ollama(text):
    response = ollama.chat(model='llama2', messages=[
        {
            'role': 'user',
            'content': f'If the following text is a post about computer science career, is it positive, negative, neutral or neither ?: "{text}" - answer in 1 word',
        },
    ])
    return response['message']['content']


def determine_sentiments(data):
    sentiments = []
    for _, row in data.iterrows():
        combined_text = f"{row['Title']} {row['Content']}"
        sentiment = ask_ollama(combined_text)

        if sentiment == "Neither":
            sentiments.append('Neither')
        elif sentiment == 'Positive':
            sentiments.append('Positive')
        elif sentiment == 'Negative':
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
