import ollama
import os
import platform

# Make sure ollama2 is installed on computer before running or will not work

def clear_screen():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')


def ask(input_data):
    query = f'{input_data} - answer in 30 or fewer words'
    # note: I am using llama2 here, if you have beefy machine can use llama2:7b
    response = ollama.chat(model='llama2', messages=[
        {
            'role': 'user',
            'content': query,
        },
    ])
    response = response['message']['content']

    return response


os.system('clear')
while True:
    question = input('Enter Question: ')
    clear_screen()
    answer = ask(question)
    print(f'Question: {question}')
    print(answer)
