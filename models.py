import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import custom_datasets
from multiprocessing.pool import ThreadPool
import time
import openai


DEVICE = "cuda"


def load_base_model_and_tokenizer(name, args):
    if args.openai_model is None:
        print(f'Loading BASE model {name}...')
        base_model_kwargs = {}
        if 'gpt-j' in name or 'neox' in name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in name:
            base_model_kwargs.update(dict(revision='float16'))
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=args.cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=args.cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer

def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])

def _openai_sample(arg_tuple):
    p, args, openai = arg_tuple

    if args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": args.openai_model, "max_tokens": 200 }
    if args.do_top_p:
        kwargs['top_p'] = args.top_p

    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text

def chatgpt_generate(prompt, args, openai, max_tokens=150):
    responses = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": args.chatgpt_preset},
            {"role": "user", "content": prompt},
        ],
    max_tokens = max_tokens,
    presence_penalty = args.chatgpt_presence_penalty,
    frequency_penalty = args.chatgpt_frequency_penalty,
    temperature= args.chatgpt_temperature,
    top_p=args.chatgpt_top_p,
    )

    response = responses['choices'][0]['message']['content']
    token_usage = responses['usage']['total_tokens']
    return response, token_usage

def sample_from_chatGPT(texts, args, surrogate_tokenizer, min_words=55, prompt_tokens=30, openai=None):
    if args.dataset == 'pubmed':
        texts = [(t[:t.index(custom_datasets.SEPARATOR)], args) for t in texts]
    else:
        tokenized = surrogate_tokenizer(texts)
        prefix = [''.join(surrogate_tokenizer.decode(ids[:prompt_tokens])) for ids in tokenized['input_ids']]

        if args.prompt is not None:
            texts = [(args.prompt + f'complete the text with at least {min_words} words: ' + t, args, openai) for t in prefix]
        else:
            texts = [(f'complete the text with at least {min_words} words: ' + t, args, openai) for t in prefix]


    pool = ThreadPool(args.batch_size)
    results = pool.starmap(chatgpt_generate, texts)
    decoded, tokens_used = map(list, zip(*results))
    # API_TOKEN_COUNTER += sum(tokens_used)

    decoded = [pre + samp for pre, samp in zip(prefix, decoded)]
    if args.prompt is not None: #strip prompt
        decoded = [x[len(args.prompt) :] for x in decoded]
    return decoded

# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, base_model, base_tokenizer, args, min_words=55, prompt_tokens=30, openai=None):
    # encode each text as a list of token ids
    if args.dataset == 'pubmed':
        texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    else:
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)

    if args.prompt is not None:
        addl_prompt = base_tokenizer(args.prompt, return_tensors="pt", padding=True).to(DEVICE)
        batch_size = len(texts)
        stacked_prompt = {key : torch.tile(value, (batch_size, 1)) for key, value in addl_prompt.items()}
        all_encoded = {key: torch.cat((stacked_prompt[key],value[:, :prompt_tokens]), dim = 1) for key, value in all_encoded.items()}
    else:
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

    if args.openai_model:
        # decode the prefixes back into text
        prefixes = base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(args.batch_size)

        arg_list = [(pf, args, openai) for pf in prefixes]
        decoded = pool.map(_openai_sample, arg_list)
    else:
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {}
            if args.do_top_p:
                sampling_kwargs['top_p'] = args.top_p
            elif args.do_top_k:
                sampling_kwargs['top_k'] = args.top_k
            min_length = 50 if args.dataset in ['pubmed'] else 150
            outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
            decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

    # if args.openai_model:
    #     global API_TOKEN_COUNTER
    #
    #     # count total number of tokens with GPT2_TOKENIZER
    #     total_tokens = sum(len(GPT2_TOKENIZER.encode(x)) for x in decoded)
    #     API_TOKEN_COUNTER += total_tokens
    if args.prompt is not None: #strip prompt
        decoded = [x[len(args.prompt) :] for x in decoded]
    return decoded
