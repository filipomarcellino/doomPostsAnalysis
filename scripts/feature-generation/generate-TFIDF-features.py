import pandas as pd
import os
import re
import sys
import nltk
import joblib
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

PROCESS_DATA_FAST = False

# This script finds the best parameters for TFIDF for our classification
# methods that use them.
#
# I have chosen complement Naive Bayes and SVMs for this task because I have
# seen that it had decent results in this paper
#
# In general, complement Naive Bayes seems to have the best performance
# in terms of classification of negative sentiments, but
# it comes at the cost of misclassification of some of the neutral
# sentiments
#
# (You will need to use institutional access for this I believe)
# Complement Naive Bayes Classifier for Sentiment Analysis of Internet Movie Database
# https://link.springer.com/content/pdf/10.1007/978-3-031-21743-2.pdf

def main():
    # get directory for data folder
    parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    data_dir = os.path.join(parent_dir, r'data')

    # get directory for cleaned data folder
    labeled_dir = os.path.join(data_dir, r'labeled')

    # get directory for saving confusion matrix graphs
    graph_dir = os.path.join(parent_dir, r'graphs')

    # get directory for saving estimators
    estimator_dir = os.path.join(parent_dir, r'estimators')

    # plot filenames
    cNB_plotfile = os.path.join(graph_dir, r'cNB_confusion.png')
    SVM_plotfile = os.path.join(graph_dir, r'SVM_confusion.png')
    GBDT_plotfile = os.path.join(graph_dir, r'GBDT_confusion.png')

    # model savefile names
    cNB_savefile = os.path.join(estimator_dir, r'cNB_estimator.joblib')
    SVM_savefile = os.path.join(estimator_dir, r'SVM_estimator.joblib')
    GBDT_savefile = os.path.join(estimator_dir, r'GBDT_estimator.joblib')

    # Get data filepath
    data_file_path = os.path.join(labeled_dir, r"all_subreddits_labelled.csv")

    # read data into CSV
    data = pd.read_csv(data_file_path)

    # download stuff we may need if it doesn't exist
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    # We want to remove stopwords at some point, but we DO NOT want to remove words like
    # not (and words with n't) so we are going to prevent that.
    #
    # When we call tokenize, we will get n't by itself, so it's fine for us to just remove
    # not from the list of words we remove
    custom_stopwords = stopwords.words('english')
    kept_stopwords = ["not", "no"]

    # I feel like filter is probably better here
    # Anyways, we should have removed not and no
    custom_stopwords = list(filter(lambda x: x not in kept_stopwords, custom_stopwords))

    # Our tokenization creates tokens such as 's or 're for contractions
    # We should add this to our stopwords
    custom_stopwords.extend(["'re", "'s", "'d", "'ll", "'ve", "'m"])

    WN_lemmatizer = WordNetLemmatizer()

    print("Transforming data")

    # Sometimes there are still NaN entries, a bit strange
    # We just filter them
    data.dropna(subset=['title', 'content'], inplace=True)

    # convert neither into neutral, it has the same meaning in
    # this context
    data[data['sentiment'] == 'neither'] = 'neutral'

    # When counting on the dataset we used run on just generate-labels
    # the label distribution is
    #
    # neither: 819
    # negative: 973
    # neutral: 19684
    # positive: 2202
    #
    # Clearly, we want to identify negative and positive sentiment
    # more than the neutral sentiment. We should probably fix
    # this imbalance a bit.

    # resample the imbalanced data
    # We will sample neutral an equal number of times
    num_pos = (data[data['sentiment'] == 'positive']).shape[0]

    # Keep the same amount of the other data
    data_app = data[data['sentiment'].isin(['positive', 'negative', 'neither'])]

    # Sample
    data = data[data['sentiment'] == 'neutral'].sample(n=num_pos)

    # Append this to get our undersampled dataframe
    data = data._append(data_app)

    # Take relevant data out
    X = data['title'] + ' ' + data['content']

    # Apply processing to the data to prepare it for TFIDF
    X = X.apply(lambda x: combined_processing(x, custom_stopwords=custom_stopwords, lemmatizer=WN_lemmatizer))
    y = data['sentiment']

    print("Hypertuning complement naive bayes. This may take some time.")

    # Split training and validation sets with 20% of the data as validation data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20)

    # We turn the text data into TF-IDF data
    # Then we run it through complement Naive Bayes
    #
    # Why? Because it is designed for imbalanced datasets
    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB
    #
    # Our input is already tokenized so we will override this
    # Same with the preprocessor
    comp_bayes_pipe = Pipeline(
        steps=[
        ('tfidf', TfidfVectorizer(stop_words=None, tokenizer=override_function,
                                  token_pattern=None, lowercase=False,
                                  preprocessor=override_function, use_idf=True, norm='l2')),
        (('cNB'), ComplementNB())
        ]
    )

    # Lets setup parameters to hypertune for this
    #
    # We will check through options:
    # Unigrams, unigrams and bigrams, or unigrams, bigrams, trigrams
    # Remove terms in more than {0.1, 0.2, ..., 0.9} percent of examples
    # Remove terms in less than {1, 3, 5, 7, 10, 15, 25, 50} examples
    # Use L1 or L2 norm
    # Use sublinear_tf or don't
    # Alpha for cNB, the default is 1.0 so we will search in powers of 10 around this

    comp_bayes_grid = {
        'tfidf__ngram_range':((1,1), (1,2)),
        'tfidf__max_df':(0.2, 0.4, 0.7, 0.8, 0.9, 1.0),
        'tfidf__min_df':(1,3,5,10),
        'cNB__alpha':(1.0, 0.1, 0.01, 0.001)
    }

    # Now do random gridsearch
    # n_iter is just how many samples we test
    # We need to pick a scoring metric that is good for unbalanced data
    # Preferably, we would want negative/positive most accurately identified
    #
    # We have some options for scoring that can be found here
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    #
    # From the docs I expect f1_micro to be better, but it often
    # can be inconsistent even if sometimes it outperforms f1_macro.

    comp_bayes_search = RandomizedSearchCV(
        estimator=comp_bayes_pipe,
        param_distributions=comp_bayes_grid,
        n_iter=100,
        n_jobs=-1,
        verbose=1,
        scoring='f1_micro'
    )

    comp_bayes_search.fit(X_train, y_train)

    best_cNB_estimator = comp_bayes_search.best_estimator_

    f1_cNB_score = best_cNB_estimator.score(X_valid, y_valid)
    print(f"Best validation score for complement naive bayes: {f1_cNB_score}")

    y_pred_cNB = best_cNB_estimator.predict(X_valid)

    cNB_confusion = confusion_matrix(y_true=y_valid, y_pred=y_pred_cNB)
    disp_cNB = ConfusionMatrixDisplay(confusion_matrix=cNB_confusion, display_labels=best_cNB_estimator.classes_)

    disp_cNB.plot()
    plt.savefig(cNB_plotfile)
    joblib.dump(best_cNB_estimator, cNB_savefile)

    if PROCESS_DATA_FAST:
        print("Skipping other models for speed.")
        return

    # Now, lets try using a SVM classifier

    print("Hypertuning SVM classifier. This may take some time.")

    # We turn the text data into TF-IDF data for the same reason
    SVM_pipe = Pipeline(
        steps=[
            ('tfidf', TfidfVectorizer(stop_words=None, tokenizer=override_function,
                                      token_pattern=None, lowercase=False,
                                      preprocessor=override_function, use_idf=True, norm='l2')),
            (('uSVM'), SVC(class_weight='balanced'))
        ]
    )

    # Lets setup hyperparameters
    # Read the docs if you want to see what each does
    #
    # The most important hyperparameter for SVM will be
    # the value for C,
    SVM_grid = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_df': (0.2, 0.4, 0.7, 0.8, 0.9, 1.0),
        'tfidf__min_df': (1, 3, 5, 10),
        'uSVM__C':(0.1, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0),
        'uSVM__shrinking':(True, False),
        'uSVM__decision_function_shape':('ovo', 'ovr')
    }

    SVM_search = RandomizedSearchCV(
        estimator=SVM_pipe,
        param_distributions=SVM_grid,
        n_iter=100,
        n_jobs=-1,
        verbose=1,
        scoring='f1_micro'
    )

    SVM_search.fit(X_train, y_train)

    best_SVM_estimator = SVM_search.best_estimator_

    f1_SVM_score = best_SVM_estimator.score(X_valid, y_valid)
    print(f"Best validation score for SVM: {f1_SVM_score}")

    y_pred_SVM = best_SVM_estimator.predict(X_valid)
    SVM_confusion = confusion_matrix(y_true=y_valid, y_pred=y_pred_SVM)
    disp_SVM = ConfusionMatrixDisplay(confusion_matrix=SVM_confusion, display_labels=best_SVM_estimator.classes_)
    disp_SVM.plot()

    plt.savefig(SVM_plotfile)
    joblib.dump(best_SVM_estimator, SVM_savefile)

    # Lets try with gradient boosting as well, since our results
    # are not exactly the best
    #
    # This takes a long time (30 minutes sometimes on my bad hardware)

    print("Hypertuning gradient boosting trees classifier. This may take a long time.")

    # We turn the text data into TF-IDF data for the same reason
    GBDT_pipe = Pipeline(
        steps=[
            ('tfidf', TfidfVectorizer(stop_words=None, tokenizer=override_function,
                                      token_pattern=None, lowercase=False,
                                      preprocessor=override_function, use_idf=True, norm='l2')),
            (('GBDT'), GradientBoostingClassifier())
        ]
    )

    # Setup hyperparameters
    GBDT_grid = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_df': (0.2, 0.4, 0.7, 0.8, 0.9, 1.0),
        'tfidf__min_df': (1, 3, 5, 10),
        'GBDT__learning_rate': (0.05, 0.1, 0.2, 0.3),
        'GBDT__n_estimators': (80, 100, 120),
        'GBDT__min_samples_split': (0.005, 0.01, 0.015), # 0.5% to 1.5%
        'GBDT__min_samples_leaf': (2, 5, 10, 25, 50),
        'GBDT__max_depth': (3,4,5,6),
        'GBDT__subsample':(0.7, 0.8, 0.9, 1.0)
    }

    #
    GBDT_search = RandomizedSearchCV(
        estimator=GBDT_pipe,
        param_distributions=GBDT_grid,
        n_iter=100,
        n_jobs=-1,
        verbose=1,
        scoring='f1_micro'
    )

    GBDT_search.fit(X_train, y_train)

    best_GBDT_estimator = GBDT_search.best_estimator_

    f1_GBDT_score = best_GBDT_estimator.score(X_valid, y_valid)
    print(f"Best validation score for GBDT: {f1_GBDT_score}")

    y_pred_GBDT = best_GBDT_estimator.predict(X_valid)
    GBDT_confusion = confusion_matrix(y_true=y_valid, y_pred=y_pred_GBDT)
    disp_GBDT = ConfusionMatrixDisplay(confusion_matrix=GBDT_confusion, display_labels=best_GBDT_estimator.classes_)
    disp_GBDT.plot()

    plt.savefig(GBDT_plotfile)
    joblib.dump(best_GBDT_estimator, GBDT_savefile)

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
    # If we get args, skip the other two models that
    # take a lot longer than complement Naive Bayes.
    if len(sys.argv) > 1:
        PROCESS_DATA_FAST = True
    main()
