### Script for one-shot Reddit bots
## Unlike ssi-bot, these are *not* necessarily finetuned on data from any subreddit
## Rather, they are prompted with a "character" to play (name + backstory)

import praw
import csv
import random
import re
import time
from datetime import datetime
import os
from hf_utils import generate_text, clean_text
import toxicity_helper
import yaml

## Load config details from YAML
def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as error:
            print(error)
    return None

class reddit_bot:
    def __init__(self, config_file):
        config = load_yaml(config_file)
        if not config:
            print('Cannot load config file; check path and formatting')
            break
        config['HF_headers'] = {"Authorization": "Bearer "+os.environ[config['HF_key_var']]}
        config['reddit_pass'] = os.environ[config['reddit_pass_var']]
        config['reddit_ID'] = os.environ[config['reddit_ID_var']]
        config['reddit_secret'] = os.environ[config['reddit_secret_var']]
        reddit = praw.Reddit(
            user_agent=config['bot_username'],
            client_id=config['reddit_ID'],
            client_secret=config['reddit_secret'],
            username=config['bot_username'],
            password=config['reddit_pass'],
        )
        reddit.validate_on_submit = True
        sub = reddit.subreddit(config['bot_subreddit'])
        hands = threading.Thread(target=self.submission_loop, args=())
        eye_1 = threading.Thread(target=self.watch_submissions, args=())
        eye_2 = threading.Thread(target=self.watch_comments, args=())

    def make_post(self):
        # make a post to a subreddit
        headers = self.config['HF_headers']
        params = self.config['text_generation_parameters']
        prompt = '\n'.join(config['bot_backstory'],self.config['post_prefix'],"Post title: ")
        title = clean_text(generate_text(self.config['textgen_model'],headers,prompt,params))
        prompt = prompt+title+"\nPost body: "
        body = clean_text(generate_text(self.config['textgen_model'],headers,prompt,params))

    def generate_reply(self, comment):
        # reply to a comment
        return(reply)

    def make_comment(self, submission):
        # reply to a submission
        return(comment)

    def watch_submissions(self):
        # watch for posts
        for submission in self.sub.stream.submissions(pause_after=0):
            # decide whether to reply to a post

    def watch_comments(self):
        # watch for comments
        for comment in self.sub.stream.comments(pause_after=0):
            if not comment:
                continue
            if comment.author == self.reddit.user.me():
                continue
            already_replied = False
            comment.submission.comments.replace_more(limit=None)
            for reply in comment.replies:
                if reply.author == reddit.user.me():
                    already_replied = True
                    break
            if already_replied:
                continue
            # parent should have author attribute regardless of submission/comment
            comment_parent == comment.parent()
            if self.config['followup_only'] and comment_parent.author != self.reddit.user.me():
                continue
            reply_probability = 0 # by default, don't reply
            if comment.parent_id[:2]=='t3' and self.config['force_top_reply']:
                reply_probability = 1
            elif trigger_words:
                for word in trigger_words:
                    if word in comment.body.lower():
                        reply_probability = self.config['reply_chance']
            elif self.config['reply_choice_model']:
                text_a = comment_parent.body
                text_b = comment.body
                # need to convert this to HF API call
                predictions, raw_outputs = model.predict([[text_a,text_b]])
                reply_probability = predictions[0]
            if random.random() < reply_probability:


    def submission_loop(self):
        while True:
            self.make_post()
            sleep(self.config['post_frequency']*3600)

    def run(self):
        self.hands.start()
        self.eye_1.start()
        self.eye_2.start()

def main():
    bot = reddit_bot("bot_config.yaml")
    bot.run()

if __name__ == "__main__":
    main()
