import re
from copy import deepcopy
import random

import pandas as pd
import torch
from datasets import load_dataset


def postprocess_text(preds, labels, original_code=None):
    """Use this function to postprocess generations and labels before BLEU computation."""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def filter_example(example, vocab_set, contractions, additional_exclusions=True):
    """Filters a particular example in a dataset
    Args:
        example: an example in the dataset
        vocab_set: the set of aochildes vocabulary to filter by
        contractions: a list of contractions to be filtered out
        additional_exclusions: a flag to specify whether to exclude additional proper nouns
    """
    features = example.keys()
    if 'sentence' in features:
        sentence = example['sentence']
        
    else:
        t1 = list(features)[0]
        t2 = list(features)[1]
        sentence = example[t1]+" "+example[t2]
    
    new_sentence = sentence.split(' ')
            
    if additional_exclusions:
        new_sentence = [w for n,w in enumerate(new_sentence) if (w == w.lower() or n==0)]
    new_sentence = [re.sub('[0-9!:&“”—\-_,@#$?;’.\'\(\)"]', '', w.lower()) for w in new_sentence]
    new_sentence = [w for w in new_sentence if w != '' and w not in contractions]
    
    for word in new_sentence:
        if word not in vocab_set:
            return False
    return True

def group_sentences(
        split,
        max_len=128,
        sentence_split_ratio=1.2,
):
    print('grouping sentences...')
    list_out = []
    cur_sequence = ''
    idx = 0
    for item in split:
        new_sequence = cur_sequence + item['TEXT'] + ' '
        if (len(new_sequence.split(' ')) * sentence_split_ratio) >= max_len:
            list_out.append(cur_sequence)
            cur_sequence = item['TEXT'] + ' '
        else:
            cur_sequence = new_sequence
    print('done')

    return list_out

def filter_glue_dataset(
    task_name, cache_dir, 
    use_auth_token=None, 
    aochildes_vocab_path="AOChildes_word_frequency.csv"
):
    """Filters GLUE datasets based on AOChildes vocabulary
    Args:
        task_name: name of GLUE dataset
        cache_dir: dir where data is cached
        use_auth_token: we will not need this mostly, but used by run_glue script
        aochildes_vocab_path: the path of AOChildes vocabulary
    """
    datasets = load_dataset(
        "glue",
        task_name,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    vocab_freq = pd.read_csv(aochildes_vocab_path)
    vocab_set = set(vocab_freq['word'])
    contractions = set(['nt','s','re','t','d','ll'])
    for key in datasets.keys():
        datasets[key] = datasets[key].filter(lambda example: filter_example(example, vocab_set, contractions))
    return datasets

def pad(sequence_list, pad_id):
    """Pads sequence_list to the longest sequence in the batch with pad_id.

    Args:
        sequence_list: a list of size batch_size of numpy arrays of different length
        pad_id: int, a pad token id

    Returns:
        torch.LongTensor of shape [batch_size, max_sequence_len]
    """
    # print(sequence_list)
    max_len = max(len(x) for x in sequence_list)
    if max_len < 1:
        max_len = 1
    padded_sequence_list = []
    for sequence in sequence_list:
        padding = [pad_id] * (max_len - len(sequence))
        padded_sequence = sequence + padding
        padded_sequence_list.append(padded_sequence)

    return torch.LongTensor(padded_sequence_list)


def sample_small_debug_dataset(raw_datasets):
    raw_datasets = sample_small_debug_dataset(raw_datasets, 100)
    return raw_datasets


def sample_small_debug_dataset(raw_datasets, sample_size):
    random_indices = random.sample(list(range(len(raw_datasets["train"]))), sample_size)
    subset = raw_datasets["train"].select(random_indices)
    raw_datasets["train"] = deepcopy(subset)
    if "validation" in raw_datasets:
        raw_datasets["validation"] = deepcopy(subset)
    if "test" in raw_datasets:
        raw_datasets["test"] = deepcopy(subset)
    return raw_datasets

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
