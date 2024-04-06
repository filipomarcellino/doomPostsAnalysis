## Using Ollama and Python for label generation

### Requirements:
- Install Ollama - https://ollama.com
- pip3 install ollama (Make sure to do this in the root directory)

### To generate labels:
- Ensure the all_subreddits.csv file is in the data/cleaned directory
- python generate-labels.py

This script will produce a new csv file all_subreddits_labeled.csv with a sentiment column. You can locate the new csv file in the data/labeled directory.

