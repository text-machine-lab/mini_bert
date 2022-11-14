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
from datasets import load_dataset, load_from_disk

import transformers
from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer, SentencePieceBPETokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import RobertaTokenizer, AutoTokenizer

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
    parser.add_argument("--load_dir", type=str, default="formatted_data", help="path to raw dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory which will be used to save tokenizer.")
    parser.add_argument("--tokenizer_type", type=str, default="BPE", required=True,
                        help="type of tokenizer to be trained choose from roberta, byte_level, sentence_piece, otherwise BPE")
    parser.add_argument(
        "--pre_process_dataset",
        default=False,
        action="store_true",
        help="If data is not formatted into hugging face format this should be true",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger.info(f"Starting tokenizer training with args {args}")

    logger.info(f"Loading  dataset")
    # raw_datasets = load_dataset(args.load_dir)
    bos_token = '<s>'
    eos_token = '</s>'
    mask_token = '<mask>'
    pad_token = '<pad>'
    unknown_token = '<unk>'
    cls_token = '<cls>'
    # we need following special tokens
    tokens_special = [f'<extra_id_{i}>' for i in range(0, 100)]
    iterator = []
    if args.pre_process_dataset:
        with open(args.load_dir, 'r') as f:
            data = json.load(f)
    else:
        data = load_from_disk(args.load_dir)

    for i, k in enumerate(data.keys()):
        iterator.append(data[k]['TEXT'])

    logger.info(f"Building tokenizer (might take a couple of minutes)")

    # Use vocab_size=args.vocab_size.
    #  The model should converge faster with a smaller vocab size.

    if args.tokenizer_type == "roberta":
        print("training roberta tokenizer on our dataset")
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        tokenizer.train_new_from_iterator(iterator, vocab_size=args.vocab_size)
        logger.info(f"Saving tokenizer to {args.save_dir}")

    elif args.tokenizer_type == "byte_level":
        print("training bytelevel bpe tokenizer on our dataset")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(iterator, vocab_size=args.vocab_size)
        logger.info(f"Saving tokenizer to {args.save_dir}")
    elif args.tokenizer_type == "sentence_piece":
        print("training sentence piece bpe tokenizer on our dataset")
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train_from_iterator(iterator, vocab_size=args.vocab_size)
        logger.info(f"Saving tokenizer to {args.save_dir}")
    else:
        print("training BPE tokenizer on our dataset")
        tokenizer = Tokenizer(BPE(unk_token=unknown_token))
        tokenizer_trainer = BpeTrainer(vocab_size=args.vocab_size,
                                       special_tokens=[unknown_token, bos_token, eos_token, mask_token, pad_token,
                                                       cls_token])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(iterator, trainer=tokenizer_trainer)

    # wrap the tokenizer to make it usable in HuggingFace Transformers
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token=unknown_token,
                                                     bos_token=bos_token,
                                                     eos_token=eos_token,
                                                     mask_token=mask_token,
                                                     pad_token=pad_token,
                                                     cls_token=cls_token)
    logger.info(f"Saving tokenizer to {args.save_dir}")
    try:
        tokenizer.save_pretrained(args.save_dir)
    except:
        #this might be roberta
        tokenizer.save_model(args.save_dir)


if __name__ == "__main__":
    main()
