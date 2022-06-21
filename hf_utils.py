import requests
import time
import re

# function for Huggingface API calls
def query(payload, model_path, headers):
    API_URL = "https://api-inference.huggingface.co/models/" + model_path
    for retry in range(3):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == requests.codes.ok:
            try:
                results = response.json()
                return results
            except:
                print('Invalid response received from server')
                print(response)
                return None
        else:
            # Not connected to internet maybe?
            if response.status_code==404:
                print('Are you connected to the internet?')
                print('URL attempted = '+API_URL)
                break
            if response.status_code==503:
                print(response.json()['error'])
                time.sleep(response.json()['estimated_time'])
                continue
            if response.status_code==504:
                print('504 Gateway Timeout')
            else:
                print('Unsuccessful request, status code '+ str(response.status_code))
                print(response.json()) #debug only
                print(payload)

def generate_text(prompt, model_path, text_generation_parameters, headers):
    start_time = time.time()
    options = {'use_cache': False, 'wait_for_model': True}
    payload = {"inputs": prompt, "parameters": text_generation_parameters, "options": options}
    output_list = query(payload, model_path, headers)
    if not output_list:
        print('Generation failed')
    end_time = time.time()
    duration = round(end_time - start_time, 1)
    stringlist = []
    if output_list and 'generated_text' in output_list[0].keys():
        print(f'{len(output_list)} sample(s) of text generated in {duration} seconds.')
        for gendict in output_list:
            stringlist.append(gendict['generated_text'])
    else:
        print(output_list)
    return(stringlist)
