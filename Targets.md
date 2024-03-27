# Candidate subreddits for NLP training

The categories below are not rigorous and only give an indication of what to expect in terms of how much negative sentiment data can be extracted. Higher negative sentiment subreddits are (probably) more likely to hold posts that will be negative.

## Subreddits directly related to computer science

These subreddits can probably be used for gathering training data or analysis without as much intervention or filtering to ensure the topic relates to computer science. This is because these subreddits are directly related to computer science.

### Minimal negative sentiment
compsci, AskComputerScience, ExperiencedDevs, webdev

### Regular negative sentiment
computerscience, cscareerquestionsOCE, learnprogramming, DevelEire, dataengineering, ITCareerQuestions

### High negative sentiment
cscareerquestions, csMajors, cscareerquestionsCAD, cscareers, cscareerquestionsEU, developersIndia, cscareerquestionsuk

## Subreddits that can possibly be used with filtering

These subreddits can probably be used if needed for gathering training data after filtering for computer science related keywords that indicate a similar cause of fear (for example, worries that AI will automate whatever job is being talked about).

### Regular negative sentiment
careerguidance, selfimprovement, Layoffs, singularity

### High negative sentiment
recruitinghell, ArtistHate, jobs

Some subreddit specific notes for filtering:

ArtistHate - most negative/doom posting is under "venting" and the general reason is due to generative AI. This is quite similar to the cause of similar posting you would see for computer science (i.e AI coding).

recruitinghell, careerguidance, jobs, Layoffs - Filled with plenty of complaining. Needs filtering for computer science and computer science adjacent topics.

selfimprovement - Some posts are related to being discouraged at the rise of AI coding.

# Discovery method

Subreddits identified above were discovered using tools such as (https://anvaka.github.io/redsim/) and manual discovery. Many subreddits that were searched through do not have a very promising amount of posts for training the model due to missing examples of one of the labels.