### Script for one-shot Reddit bots using Huggingface models
## Unlike ssi-bot, these are *not* necessarily finetuned on data from any subreddit
## Rather, they are prompted with a "character" to play (name + backstory)

import praw
import csv
import random
import re
import time
from datetime import datetime, date
import os, sys
from hf_utils import generate_text, clean_text, query
from tagging_mixin import TaggingMixin
from toxicity_helper import ToxicityHelper
import yaml
import threading
from nltk import word_tokenize

_default_negative_keywords = [
    ('ar', 'yan'), ('ausch, witz'),
    ('black', ' people'),
    ('child p', 'orn'), ('concentrati', 'on camp'),
    ('fag', 'got'),
    ('hit', 'ler'), ('holo', 'caust'),
    ('inc', 'est'), ('israel'),
    ('jew', 'ish'), ('je', 'w'), ('je', 'ws'),
    (' k', 'ill'), ('kk', 'k'),
    ('lol', 'i'),
    ('maste', 'r race'), ('mus', 'lim'),
    ('nation', 'alist'), ('na', 'zi'), ('nig', 'ga'), ('nig', 'ger'),
    ('pae', 'do'), ('pale', 'stin'), ('ped', 'o'),
    ('rac' 'ist'), (' r', 'ape'), ('ra', 'ping'),
    ('sl', 'ut'), ('swas', 'tika'),
]

_negative_keywords = ["".join(s) for s in _default_negative_keywords]

## Load config details from YAML
def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as error:
            print(error)
    return None

def words_below(string,max_words):
    # check to see if an input string would exceed token budget
    token_list = word_tokenize(string)
    if len(token_list)>max_words:
        return False
    else:
        return True

def words_below(string,max_words):
    # check to see if an input string would exceed token budget
    wordlike_list = string.split()
    if len(wordlike_list)>max_words:
        return False
    else:
        return True

class reddit_bot:
    def __init__(self, config_file):
        self.config = load_yaml(config_file)
        if not self.config:
            print('Cannot load config file; check path and formatting')
            sys.exit()
        self.HF_key = os.environ[self.config['HF_key_var']]
        self.headers = {"Authorization": "Bearer "+self.HF_key}
        self.config['reddit_pass'] = os.environ[self.config['reddit_pass_var']]
        self.config['reddit_ID'] = os.environ[self.config['reddit_ID_var']]
        self.config['reddit_secret'] = os.environ[self.config['reddit_secret_var']]
        self.reddit = praw.Reddit(
            user_agent=self.config['bot_username'],
            client_id=self.config['reddit_ID'],
            client_secret=self.config['reddit_secret'],
            username=self.config['bot_username'],
            password=self.config['reddit_pass'],
        )
        self.me = self.reddit.user.me()
        self.reddit.validate_on_submit = True
        self.sub = self.reddit.subreddit(self.config['bot_subreddit'])
        self.submission_writer = threading.Thread(target=self.submission_loop, args=())
        self.submission_reader = threading.Thread(target=self.watch_submissions, args=())
        self.comment_reader = threading.Thread(target=self.watch_comments, args=())
        self.today = date.today()
        self.tally = 0 # to compare with daily input character budget
        self.SSI = TaggingMixin() # handler for legacy SSI tagging functions
        self.detox = ToxicityHelper(self.HF_key,{'nsfw': 0.9, 'hate': 0.9, 'threat': 0.9})
        self.negative_keywords = _negative_keywords + self.config['negative_keywords']

    def bad_keyword(self,text):
        return [keyword for keyword in self.negative_keywords if re.search(r"\b{}\b".format(keyword), text, re.IGNORECASE)]

    def check_budget(self,string):
        # check to see if an input string would exceed character budget
        # first, check the date; reset it and the tally if changed
        if date.today() != self.today:
            # reset the character budget and date
            self.today = date.today()
            self.tally = 0
        character_cost = len(string)
        if (self.tally + character_cost) < self.config['character_budget']:
            return True
        else:
            return False

    def make_post(self):
        submission = None
        # ssi-bot style GPT-2 model text post generation
        prompt = '<|soss'
        if self.check_budget(prompt):
            self.tally += len(prompt)
            print("Generating a post on r/"+self.sub.display_name)
            post_params = self.config['text_generation_parameters']
            post_params['return_full_text'] = True
            generated_text = generate_text(prompt,self.config['post_textgen_model'],post_params,self.headers)
            print(f"GENERATED: {generated_text}")
            if generated_text:
                if self.check_budget(generated_text) and words_below(generated_text,500):
                    self.tally += len(generated_text)
                    if self.detox.text_above_toxicity_threshold(generated_text):
                        print("Generated text failed toxicity check, discarded.")
                    else:
                        # To do: image post logic, text-to-image models
                        post = self.SSI.extract_submission_from_generated_text(generated_text)
                        if post:
                            if post['title'] and post['selftext']:
                                submission = self.sub.submit(title=post['title'],selftext=post['selftext'])
                                print("Post successful!")
                            else:
                                print("Either title or selftext is missing!")
                        else:
                            print("Failed to extract post from generated text!")
                else:
                    print("Too many characters in generated text, discarded.")
            else:
                print("Text generation failed!")
        else:
            print("Not enough characters left in budget to make a post!")
        return submission

    def generate_reply(self, comment):
        print("Generating a reply to comment:\n"+comment.body)
        reply = None
        # accumulate comment thread for context
        at_top = False
        prompt = 'Reply by u/{}: "'.format(self.config['bot_username'])
        thread_item = comment
        for level in range(self.config['max_levels']):
            prompt = '\n'.join(['Comment by u/{}: "{}"'.format(thread_item.author.name, thread_item.body),prompt])
            if thread_item.parent_id[:2]=='t3':
                # next thing is the post, not a comment
                # To do: image recognition/description for link posts
                thread_post = comment.submission
                if thread_post.is_self:
                    thread_OP = thread_post.author.name
                    post_title = thread_post.title
                    post_body = thread_post.selftext
                    prompt = '\n'.join(['Post by u/{} titled "{}": "{}"'.format(thread_OP,post_title,post_body),prompt])
                break
            else:
                thread_item = thread_item.parent()
        prompt = '\n'.join([self.config['bot_backstory'],prompt])
        if self.check_budget(prompt) and words_below(prompt, 1000):
            self.tally += len(prompt)
            print(f"PROMPT: {prompt}")
            reply_params = self.config['text_generation_parameters']
            reply_params['return_full_text'] = False
            cleanStr = clean_text(generate_text(prompt,self.config['reply_textgen_model'],reply_params,self.headers))
            print(f"GENERATED: {cleanStr}")
            if cleanStr:
                if self.check_budget(cleanStr) and words_below(cleanStr,500):
                    self.tally += len(cleanStr)
                    if not self.detox.text_above_toxicity_threshold(cleanStr):
                        reply = comment.reply(body=cleanStr)
                        print("Reply successful!")
                    else:
                        print("Text is toxic, skipping...")
                else:
                    print("Unable to check toxicity, skipping...")
            else:
                print("Generation failed, skipping...")
        else:
            print("Prompt is too long, skipping...")
        return reply

    def make_comment(self, submission):
        comment = None
        # reply to a submission
        thread_OP = submission.author.name
        post_title = submission.title
        print("Commenting on submission:\n"+post_title)
        post_body = comment.submission.selftext
        prompt = 'Comment by u/{}: "'.format(self.config['bot_username'])
        prompt = '\n'.join(['Post by u/{} titled "{}": "{}"'.format(thread_OP,post_title,post_body),prompt])
        prompt = '\n'.join([self.config['bot_backstory'],prompt])
        if self.check_budget(prompt) and words_below(prompt,1000):
            self.tally += len(prompt)
            print(f"PROMPT: {prompt}")
            reply_params = self.config['text_generation_parameters']
            reply_params['return_full_text'] = False
            cleanStr = clean_text(generate_text(prompt,self.config['reply_textgen_model'],reply_params,self.headers))
            print(f"GENERATED: {cleanStr}")
            if cleanStr:
                if self.check_budget(cleanStr) and words_below(cleanStr):
                    self.tally += len(cleanStr)
                    if not self.detox.text_above_toxicity_threshold(cleanStr):
                        reply = comment.reply(cleanStr)
                        print("Comment successful!")
                    else:
                        print("Text is toxic, skipping...")
                else:
                    print("Unable to check toxicity, skipping...")
            else:
                print("Generation failed, skipping...")
        else:
            print("Prompt is too long, skipping...")
        return comment

    def watch_submissions(self):
        # watch for posts
        while True:
            for submission in self.sub.stream.submissions(pause_after=0,skip_existing=True):
                # decide whether to reply to a post
                if not submission:
                    continue
                if submission.author == self.me:
                    continue
                if self.bad_keyword(submission.title) or (submission.is_self and self.bad_keyword(submission.selftext)):
                    continue
                already_replied = False
                submission.comments.replace_more(limit=None)
                for comment in submission.comments:
                    if comment.author == reddit.user.me():
                        already_replied = True
                        break
                if already_replied:
                    continue
                reply_probability = self.config['reply_chance']
                if self.config['trigger_words']:
                    if submission.is_self:
                        post_string = ' '.join([submission.title.lower(),submission.selftext.lower()])
                    else:
                        post_string = submission.title.lower()
                    for word in self.config['trigger_words']:
                        if word.lower() in post_string:
                            reply_probability = reply_probability + self.config['trigger_boost']
                if random.random() < reply_probability:
                    print("Generating a comment on submission "+submission.id)
                    self.make_comment(submission)
            # except:
            #     print('Error reading comments, are you connected to the Internet?')
            #     time.sleep(60)

    def watch_comments(self):
        # watch for comments
        while True:
            for comment in self.sub.stream.comments(pause_after=0,skip_existing=True):
                if not comment:
                    continue
                if comment.author == self.me:
                    continue
                if self.bad_keyword(comment.body):
                    print("Bad keyword found, skipping...")
                    continue
                already_replied = False
                comment.replies.replace_more(limit=None)
                for reply in comment.replies:
                    if reply.author == self.me:
                        already_replied = True
                        break
                if already_replied:
                    continue
                # parent should have author attribute regardless of submission/comment
                comment_parent = comment.parent()
                if self.config['followup_only'] and comment_parent.author != self.me:
                    continue
                if comment.parent_id[:2]=='t3':
                    if self.config['force_top_reply']:
                        print("Top-level reply forced.")
                        self.generate_reply(comment)
                    else:
                        reply_probability = self.config['reply_chance']
                        if self.config['trigger_words']:
                            print("Trigger word found!")
                            for word in self.config['trigger_words']:
                                if word.lower() in comment.body:
                                    reply_probability = reply_probability + self.config['trigger_boost']
                        if random.random() < reply_probability:
                            self.generate_reply(comment)
                else:
                    print('Checking comment "{}"'.format(comment.body))
                    prompt = comment_parent.body + "<|endoftext|>" + comment.body
                    if self.check_budget(prompt) and words_below(prompt, 1000):
                        self.tally += len(prompt)
                        payload = {"inputs": prompt}
                        score = query(payload, "microsoft/DialogRPT-width", self.headers)[0][0]['score']
                        if score >= self.config['min_reply_score']:
                            self.generate_reply(comment)
                        else:
                            print("Comment not selected for reply, skipping...")
            # try:
            # except:
            #     print('Error reading comments, are you connected to the Internet?')
            #     time.sleep(60)

    def submission_loop(self):
        while (self.config['post_frequency']>0):
            post = self.make_post()
            # try:
            # except:
            #     print('Error making a post, are you connected to the Internet?')
            #     time.sleep(60)
            if post:
                time.sleep(self.config['post_frequency']*3600)
        print("No submissions scheduled, exiting")

    def run(self):
        print("Launching submission writer")
        self.submission_writer.start()
        # don't bother running submission reader if bot is followup-only
        if not self.config['followup_only']:
            print("Launching submission reader")
            self.submission_reader.start()
        else:
            print("Bot set to follow-up only, will not read submissions.")
        print("Launching comment reader")
        self.comment_reader.start()

def main():
    bot = reddit_bot("bot_config.yaml")
    bot.run()

if __name__ == "__main__":
    main()
