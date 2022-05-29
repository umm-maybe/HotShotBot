### Script for one-shot Reddit bots using Huggingface models
## Unlike ssi-bot, these are *not* necessarily finetuned on data from any subreddit
## Rather, they are prompted with a "character" to play (name + backstory)

import praw
import csv
import random
import re
import time
from datetime import datetime, date
import os
from hf_utils import generate_text, clean_text, query
from tagging_mixin import TaggingMixin
from toxicity_helper import ToxicityHelper
import yaml

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
    # check to see if an input string would exceed character budget
    wordlike_list = string.split()
    if len(wordlike_list)>max_words:
        return False
    else:
        return True

class reddit_bot:
    def __init__(self, config_file):
        config = load_yaml(config_file)
        if not config:
            print('Cannot load config file; check path and formatting')
            break
        HF_key = os.environ[config['HF_key_var']]
        config['HF_headers'] = {"Authorization": "Bearer "+HF_key}
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
        me = reddit.user.me()
        headers = self.config['HF_headers']
        params = self.config['text_generation_parameters']
        reddit.validate_on_submit = True
        sub = reddit.subreddit(config['bot_subreddit'])
        submission_writer = threading.Thread(target=self.submission_loop, args=())
        submission_reader = threading.Thread(target=self.watch_submissions, args=())
        comment_reader = threading.Thread(target=self.watch_comments, args=())
        today = date.today()
        tally = 0 # to compare with daily input character budget
        SSI = TaggingMixin() # handler for legacy SSI tagging functions
        detox = ToxicityHelper(HF_key,{'nsfw': 0.9, 'hate': 0.9, 'threat': 0.9})
        negative_keywords = _negative_keywords + self.config['negative_keywords']

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
            generated_text = generate_text(prompt,post_textgen_model,self.params,self.headers)
            if generated_text:
                if self.check_budget(generated_text) and words_below(generated_text,500):
                    self.tally += len(generated_text)
                    if not self.detox.text_above_toxicity_threshold(generated_text):
                        # To do: image post logic, text-to-image models
                        post = self.SSI.extract_submission_from_generated_text(generated_text)
                        submission = self.sub.submit(title=post['title'],selftext=post['selftext'])
        return submission

    def generate_reply(self, comment):
        reply = None
        # accumulate comment thread for context
        at_top = False
        level = 0
        prompt = 'Reply by u/{}: "'.format(self.config['bot_username'])
        curr_comment = comment
        while not at_top and levels <= self.config['max_levels']:
            level += 1
            prompt = '\n'.join('Comment by u/{}: "{}"'.format(curr_comment.author.name, curr_comment.body),prompt)
            if curr_comment.parent_id[:2]=='t3':
                # it's the post, not a comment
                at_top = True
                thread_OP = comment.submission.author.name
                post_title = comment.submission.title
                if comment.submission.is_self:
                    post_body = comment.submission.selftext
                    prompt = '\n'.join('Post by u/{} titled "{}": "{}"'.format(thread_OP,post_title,post_body),prompt)
                # To do: image recognition/description for link posts
            else:
                curr_comment = comment.parent()
        # effectively, this prevents the bot from responding below max_levels
        # the reason for this is that without the post, there's not enough context
        if at_top:
            prompt = '\n'.join(self.config['bot_backstory'],prompt)
            if self.check_budget(prompt) and words_below(prompt, 1000):
                self.tally += len(prompt)
                cleanStr = clean_text(generate_text(prompt,reply_textgen_model,self.params,self.headers))
                if cleanStr:
                    if self.check_budget(cleanStr) and words_below(cleanStr,500):
                        self.tally += len(cleanStr)
                        if not self.detox.text_above_toxicity_threshold(cleanStr):
                            reply = comment.reply(cleanStr)
        return reply

    def make_comment(self, submission):
        comment = None
        # reply to a submission
        thread_OP = submission.author.name
        post_title = submission.title
        post_body = comment.submission.selftext
        prompt = 'Comment by u/{}: "'.format(self.config['bot_username'])
        prompt = '\n'.join('Post by u/{} titled "{}": "{}"'.format(thread_OP,post_title,post_body),prompt)
        prompt = '\n'.join(self.config['bot_backstory'],prompt)
            if self.check_budget(prompt) and words_below(prompt,1000):
                self.tally += len(prompt)
                cleanStr = clean_text(generate_text(prompt,reply_textgen_model,self.params,self.headers))
                if cleanStr:
                    if self.check_budget(cleanStr) and words_below(cleanStr):
                        self.tally += len(cleanStr)
                        if not self.detox.text_above_toxicity_threshold(cleanStr):
                            reply = comment.reply(cleanStr)
        return comment

    def watch_submissions(self):
        # watch for posts
        while True:
            try:
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
                            post_string = ' '.join(submission.title.lower(),submission.selftext.lower())
                        else:
                            post_string = submission.title.lower()
                        for word in self.config['trigger_words']:
                            if word.lower() in post_string:
                                reply_probability = reply_probability + self.config['trigger_boost']
                    if random.random() < reply_probability:
                        self.make_comment(submission)
            except:
                print('Error reading comments, are you connected to the Internet?')
                time.sleep(60)

    def watch_comments(self):
        # watch for comments
        while True:
            try:
                for comment in self.sub.stream.comments(pause_after=0,skip_existing=True):
                    if not comment:
                        continue
                    if comment.author == self.me:
                        continue
                    if self.bad_keyword(comment.body):
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
                    comment_parent == comment.parent()
                    if self.config['followup_only'] and comment_parent.author != self.me:
                        continue
                    if comment.parent_id[:2]=='t3':
                        if self.config['force_top_reply']:
                            self.generate_reply(comment)
                        else:
                            reply_probability = self.config['reply_chance']
                            if self.config['trigger_words']:
                                for word in self.config['trigger_words']:
                                    if word.lower() in comment.body:
                                        reply_probability = reply_probability + self.config['trigger_boost']                            reply_probability
                            if random.random() < reply_probability:
                                self.generate_reply(comment)
                    else:
                        prompt = comment_parent.body + "<|endoftext|>" + comment.body
                        if check_budget(prompt) and words_below(prompt, 1000):
                            self.tally += len(prompt)
                            payload = {"inputs": prompt}
                            score = query(payload, "microsoft/DialogRPT-width", self.headers)[0][0]['score']
                            if score >= self.config['min_reply_score']:
                                self.generate_reply(comment)
            except:
                print('Error reading comments, are you connected to the Internet?')
                time.sleep(60)

    def submission_loop(self):
        while True:
            try:
                self.make_post()
            except:
                print('Error making a post, are you connected to the Internet?')
                time.sleep(60)
            time.sleep(self.config['post_frequency']*3600)

    def run(self):
        self.submission_writer.start()
        # don't bother running submission reader if bot is followup-only
        if not self.config['followup_only']:
            self.submission_reader.start()
        self.comment_reader.start()

def main():
    bot = reddit_bot("bot_config.yaml")
    bot.run()

if __name__ == "__main__":
    main()
