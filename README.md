# Hot-Shot Bots
This project is intended to provide a codebase for hybrid few-shot GPT Reddit bots to be run on bot-friendly subreddits (e.g. r/SubSimGPT2Interactive).  

Whereas in the past SSI bots used GPT-2 models that had been fine-tuned on archives of data downloaded for one or more subreddits for all posts and replies, these bots use fine-tuned GPT-2 models *for posts only*.  Comments/replies are generated using the standard pretrained version of a larger language model (e.g. GPT-J) which is given a prompt including, as context: 1) a backstory describing the kind of Redditor making the reply, 2) the original post and 3) the comment thread.

## Why this works
Larger language models (meaning ones with billions of parameters, generally speaking) have seen so much text from across the Internet (and specifically Reddit) that the writing style you want is probably hiding within the parameter weights somewhere, and you can call upon that style to be activated using a well-structured prompt. In this case, we're prompting it with generated content from a smaller GPT-2 model that has been fine-tuned on the subreddits whose style we want to simulate.

These larger language models are also capable of role-play, meaning they can generate text that they think is consistent with a character or situation that you define in their prompt.  Essentially, they've seen enough examples of role-play in the wild that you can increase the probability of seeing a certain writing style or vocabulary by describing the kind of character or situation in which that language is found.  We call this the bot's "backstory".

For more background information on large language models and their few-shot learning behavior, see the following papers:

* [Attention Is All You Needd](https://arxiv.org/abs/1706.03762)
* [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
* [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
* [GPT-J-6B: 6B JAX-Based Transformer](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/)

## Variants
It is entirely possible to configure this type of bot to be "reply-only", in which case no fine-tuned GPT-2 model need ever be used.  The bot will never make posts, simply replying based upon its backstory and the post/comment thread to which it is commenting.

It is also feasible, though probably unnecessary, to fine-tune GPT-J.  If you are interested in doing this, here is a guide which other people have followed with some success:

[How to Fine-Tune GPT-J - The Basics
](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md)

## Requirements
To make the use of really large language models possible for plebian bot operators using lowly home servers and affordable computing cloud instances, we take advantage of the [Huggingface Accelerated CPU Inference API](https://huggingface.co/inference-api).  Anyone can [sign up](https://huggingface.co/join) and get access to 30,000 free input characters a month, which may not be much, but it's something.  Paid plans currently start at 1M characters a month for a price which is comparable to that of most VPS hosting providers charge for machines capable of running [the most popular codebase](https://github.com/zacc/ssi-bot) for interactive Reddit GPT-2 bots.

Neither the actual Huggingface Transformers library, nor Torch are actually required to be installed, and you don't have to have access to a machine capable of running inference on the large language models being used.  In our tests a single bot consumed just 45MB of RAM and 1.5% of CPU on a mid-2015 MacBook Pro.

A design decision was also made not to use any form of local database to duplicate data that Reddit already stores for us and makes available free of charge via PRAW.  Thus there are minimal disk read/writes, and minimal storage requirements (though maybe more network I/O than you might otherwise expect).

## Notes
A couple of BERT models are used to keep things running on the rails:

* A [Dialog Ranking Pretrained Transformer](https://huggingface.co/microsoft/DialogRPT-width) is used to decide whether to reply to comments; and
* A [toxicity classifier](hitomi-team/discord-toxicity-classifier) is used to prevent hateful, threatening, or lewd speech from being posted by the bot on Reddit.

Other re-ranking models will be added as options in the near future.

For more information on BERT see the following paper:

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)


## Setup
If you want to generate posts using a fine-tuned GPT-2 model, first follow the instructions [here](https://github.com/zacc/ssi-bot).  You can use the iPython notebook included in that repository to fine-tune a small- or medium-sized GPT-2 model for free on Google Colab.  Downloading and filtering Reddit data to make a good bot will be your biggest challenge, but there are tools and advice for that as well.  Also, follow its instructions for creating a Reddit username and getting an API ID and secret for it to use.

Once your GPT-2 model has been fine-tuned, upload it to the Huggingface model hub.  This is a free service that will make it possible to generate text from your model using the Huggingface API.  Instructions can be found [here](https://huggingface.co/docs/hub/adding-a-model#using-the-web-interface-and-command-line).

**Note**: you may initially encounter problems with generation that can be solved by deleting the version of  `tokenizer_config.json` found in the folder you downloaded from Google Colab.

Clone this repository to the machine from which you wish to run your bot.  Create a virtual environment using `venv` (or `conda`, if you prefer), activate it, and run `pip install -Ur requirements.txt` to make sure all dependencies are met.

## Configuration
All parameters are set in a YAML file named `bot_config.yaml`.  These are, for the most part, well-commented and/or self-explanatory.  Nonetheless, here are some tips for the easily confused:

* Environment variables must be created on your system to store the Reddit password, ID and secret for your bot, as well as your Huggingface API key (which can be obtained by visiting [this link](https://huggingface.co/settings/tokens)).  Reference the names of these variables, rather than the actual values.
* Negative keywords are used to block replies to a post or comment; a default list of these is incorporated within the bot code.
* Positive keywords can be used to modify the baseline chance of your bot making a reply when the re-ranker does not apply (e.g. top-level comments).  Or, you can force the bot to respond to all top-level comments.
* You can also limit your bot to only follow up on replies to posts or comments that it previously made.
* The `character_budget` is a daily limit on how many characters may be sent to the accelerated inference API; the bot will prevent itself from going above this number.  This is so that you don't unwittingly face massive charges from Huggingface.

## Operation
Once your bot is configured, you can run it by using the following command: `python3 bot.py`

At the bottom of the terminal (or `tmux` instance) in which the bot is running, there is a status report which should look something like the following:

`READ: post=0	reply=0	| WRITE: post=0	reply=0	| SPEND=0%`

The values shown will be updated as the bot runs, so that you can verify that it is actually receiving data (otherwise it should be fairly quiet).  The SPEND value is the ratio between the characters sent to the inference API thus far in that day, and the total budget you have assigned; it is reset to zero every day.

## Credits
Some code was borrowed from the [ssi-bot](https://github.com/zacc/ssi-bot) repository created by [u/tateisukannanirase](https://www.reddit.com/user/tateisukannanirase/).




