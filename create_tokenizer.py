# !/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging

import datasets
from datasets import load_dataset

import transformers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tokenizer")

    parser.add_argument("--vocab_size", type=int, required=True, help="Size of the vocabulary")
    parser.add_argument("--load_dir", type=str, required=True, help="Directory which has raw data.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory which will be used to save tokenizer.")
    parser.add_argument("--pre_process_dataset", action='store_true', required=True,
                        help="whether to preprocess dataset or not. If set to true, will preprocess and resave the dataset.")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger.info(f"Starting tokenizer training with args {args}")

    logger.info(f"Loading  dataset")
    # raw_datasets = load_dataset(args.load_dir)
    # we need following special tokens
    tokens_special = ['<s>', '</s>', '<mask>', '<pad>', '<unk>'] + [f'<extra_id_{i}>' for i in range(0, 100)]
    iterator = []
    if args.pre_process_datatset:
        with open(args.load_dir, 'r') as f:
            data = json.load(f)
        for i, k in enumerate(data.keys()):
            iterator.append(data[k]['TEXT'])
        # iterator = (data[k]['TEXT'] for i, k in enumerate(data))
    else:
        raw_datasets = load_dataset('json', args.load_dir)
        iterator = (item["text"] for item in raw_datasets)
    logger.info(f"Building tokenizer (might take a couple of minutes)")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Task 3.1: train a BPE tokenizer
    # You only need ["UNK"] and ["PAD"] special tokens.
    # DO NOT just copy everything from the tutorial.
    # Use vocab_size=args.vocab_size.
    # When you run this sript use 8192. The model should converge faster with a smaller vocab size.
    # Tokenizer training tutorial: https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
    # API reference for tokenizer.trainer class: 

    # YOUR CODE STARTS HERE (our implementation is about 6 lines)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]", ))
    tokenizer_trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=tokens_special)
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train_from_iterator(iterator, trainer=tokenizer_trainer)
    # YOUR CODE ENDS HERE

    # wrap the tokenizer to make it usable in HuggingFace Transformers
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    logger.info(f"Saving tokenizer to {args.save_dir}")
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
