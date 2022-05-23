import logging
import time
import json
import requests
import hf_utils

from utils import ROOT_DIR

class ToxicityHelper():

    # this version uses the hitomi-team toxicity classifier

    def __init__(self, HF_key, threshold_map):
        self._headers = {"Authorization": "Bearer "+HF_key}
        self._threshold_map = threshold_map # e.g. {'nsfw': 0.9, 'hate': 0.9, 'threat': 0.9}

    def _detoxify_predict(self, text):
        payload = {"inputs": text}
        output_list = hf_utils.query(payload, hitomi-team/discord-toxicity-classifier, self._headers)
        return_dict = {}
        if len(output_list)!=1:
            print(output_list)
            return return_dict
        temp_results = output_list[0]
        return_dict = {'nsfw': 0, 'hate': 0, 'threat': 0}
        for result in temp_results:
            labelseq = json.loads(result['label'])
            score = result['score']
            return_dict['nsfw'] = return_dict['nsfw'] + result['score'] * labelseq[0]
            return_dict['hate'] = return_dict['hate'] + result['score'] * labelseq[1]
            return_dict['threat'] = return_dict['threat'] + result['score'] * labelseq[2]
        return return_dict

    def text_above_toxicity_threshold(self, input_text):
        try:
            results = self._detoxify_predict(input_text)
        except:
            logging.exception(f"Exception when trying to run detoxify prediction on {input_text}")

        if self._threshold_map.keys() != results.keys():
            logging.warning(f"Detoxify results keys and threshold map keys do not match. The toxicity level of the input text cannot be calculated.")
            return True

        for key in self._threshold_map:
            if results[key] > self._threshold_map[key]:
                return True
