import requests
import time
import re

# function for Huggingface API calls
def query(payload, model_path, headers):
    API_URL = "https://api-inference.huggingface.co/models/" + model_path
    while True:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == requests.codes.ok:
            break
        else:
            # Not connected to internet maybe?
            print('Unsuccessful request, status code '+ str(response.status_code))
            print(response.json()) #debug only
            if response.status_code==404:
                print('Are you connected to the internet?')
                print(API_URL)
                break
            if response.status_code==400:
                print(headers)
                break
            if response.status_code==503:
                time.sleep(response.json()['estimated_time'])
            else:
                print(payload)
    return response.json()

def generate_text(prompt, model_path, text_generation_parameters, headers):
    start_time = time.time()
    payload = {"inputs": prompt, "parameters": text_generation_parameters}
    output_list = query(payload, model_path, headers)
    end_time = time.time()
    duration = round(end_time - start_time, 1)
    print(f'{len(output_list)} sample(s) of text generated in {duration} seconds.')
    if output_list and 'generated_text' in output_list[0].keys():
        return(output_list[0]['generated_text'])
    else:
        print(output_list)
        return('')

def clean_text(generated_text):
    truncate = 0
    cleanStr = ''
    # look for double-quotes
    truncate = generated_text.find('"')
    if truncate>-1:
        cleanStr = generated_text[:truncate]
    # if we can't find double-quotes, look for punctuation
    elif re.search(r'[?.!]', generated_text):
            trimPart = re.split(r'[?.!]', generated_text)[-1]
            cleanStr = generated_text.replace(trimPart,'')
    # if we can't find punctuation, use the last space
    else:
        truncate = generated_text.rfind(' ')
        if truncate>-1:
            cleanStr = generated_text[:truncate+1]
    if not cleanStr:
        print('Bad generation')
    return cleanStr

def rank_text(candidate_texts, positive_keywords, reranking_model):
    return ranked_candidates
