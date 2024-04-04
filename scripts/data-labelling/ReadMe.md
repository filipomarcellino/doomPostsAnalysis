Using Ollama and Python for Local AI LLM System (Ollama, Llama2, Python) to label data


IMPORTANT NOTE:
Make sure Ollama is running on the local system before running the Python script!


Steps:
Install Ollama - https://ollama.com (model to run on machine)

In terminal - ollama run llama2 

pip3 install ollama (python library)

proceed to run python script



For running labelling script:

python labelscript.py data.csv

will produce new csv file data_labelled.csv with sentiment column

