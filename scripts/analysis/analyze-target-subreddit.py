import sys
import os
import joblib
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag



def main(model_choice):

    # subreddit to analyze by default
    analysis_filename = "analysis.csv"

     # get directory for data folder
    parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    data_dir = os.path.join(parent_dir, r'data')

    # get directory for cleaned data folder
    cleaned_dir = os.path.join(data_dir, r'cleaned')

    # get directory for the estimators folder
    estimators_dir = os.path.join(parent_dir, r'estimators')

    # set path
    analysis_filename = os.path.join(cleaned_dir, analysis_filename)

    analysis_data = pd.read_csv(analysis_filename)

    # load model of choice
    model_filename = None
    match model_choice:
        case 'cNB':
            model_filename = os.path.join(estimators_dir, "cNB_estimator.joblib")
        case 'SVM':
            model_filename = os.path.join(estimators_dir, "SVM_estimator.joblib")
        case 'GBDT':
            model_filename = os.path.join(estimators_dir, "GBDT_estimator.joblib")
        case _:
            print("Please specify a valid model as the second argument.")
            print("Valid model choices are: cNB, SVM, GBDT")
            sys.exit()


    print("loading model")
    model = joblib.load(model_filename)

    print("Transforming data")
    # Take relevant data out
    X = analysis_data['title'] + ' ' + analysis_data['content']

    # stopwords used in generate-TFIDF-features.py
    custom_stopwords = stopwords.words('english')
    kept_stopwords = ["not", "no"]
    custom_stopwords = list(filter(lambda x: x not in kept_stopwords, custom_stopwords))
    custom_stopwords.extend(["'re", "'s", "'d", "'ll", "'ve", "'m"])

    # lemmatizer used in generate-TFIDF-features
    WN_lemmatizer = WordNetLemmatizer()

    # Apply processing to the data to prepare it for TFIDF
    X = X.apply(lambda x: combined_processing(x, custom_stopwords=custom_stopwords, lemmatizer=WN_lemmatizer))
    analysis_data['label'] = model.predict(X)

    print(analysis_data['label'])

    return


# We want to get the text ready for the TFIDF feature generation
# Doing it like this prevents issues where scikit doesn't deal with stop words
# correctly because we're using NLTK
# (read: run screen is filled with raised errors about stopwords)
def combined_processing(text, custom_stopwords, lemmatizer):
    """
    Function to do all processing before TFIDF feature extraction.
    We tokenize, remove stopwords (excluding negation), and lemmatize
    """

    # First step, set to lower
    text = text.lower()
    # remove extra whitespace
    text = text.strip()

    # tokenize with NLTK tokenizer
    text = word_tokenize(text)

    # Remove special characters except apostrophe
    remove_special = r"[^A-Za-z']"

    # Map doesn't work nicely so we do this, probably terrible for resource usage
    text = [re.sub(remove_special, '', word) for word in text]
    # remove empty '' strings
    # yes, this does do a double pass over the list for this operation,
    # but I don't know what else to do in this language so it is what it is
    tokenized_text = [word for word in text if (word != '')]

    # remove stopwords
    tokenized_text = [word for word in tokenized_text if (word not in custom_stopwords)]

    # Now to run lemmatization on the text
    #
    # We're going to use WordNet because, well, it's in NLTK so it's just
    # easier to use the stuff from the same library.
    #
    # This function has an input "pos" that helps it (read the docs if you want)
    # https://www.nltk.org/api/nltk.stem.wordnet.html?highlight=wordnetlemmatizer
    tagged_text = pos_tag(tokenized_text)

    # These tags are not in the form that is accepted by the lemmatizer
    # There are 5 options listed on the docs, "n", "v", "a", "r", "s
    # Tags have a bunch of starting symbols that can be found
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    #
    # I have gone through the list and found that if the tags starts with:
    # J -> adjective
    # N -> noun
    # R -> adverb
    # V -> verb
    #
    # I could not find a direct way to map something to "satelite adjective"
    #
    # So, we want to convert from (word, pos_tag) -> (word, wordnet_tag).
    # Apply a mapping for this

    # word[0],word[1] -> word[0],wordnet_tag(word[0])
    tagged_text = [(word[0],map_tags(word[1])) for word in tagged_text]

    final_text = []

    # After mapping, lets run the lemmatizer with our tags
    for word in tagged_text:
        if word[1] != None:
            final_text.append(lemmatizer.lemmatize(word[0], word[1]))
        # If there was no tag for the lemmatizer, don't run it.
        else:
            final_text.append(word[0])

    return final_text


def map_tags(tag):
    """Utility function to map tags to wordnet tags"""
    # Determine what wordnet tag to convert to
    # We only really need the first letter
    match tag[0]:
        case "J":
            return wordnet.ADJ
        case "N":
            return wordnet.NOUN
        case "R":
            return wordnet.ADV
        case "V":
            return wordnet.VERB
        # return None, we skip this tag if true
        case _:
            return None


def override_function(token):
    return token


if __name__ == "__main__":
    model_choice = None
    if(len(sys.argv) == 1):
        model_choice = "cNB"
    else:
        model_choice = sys.argv[1]
    main(model_choice)