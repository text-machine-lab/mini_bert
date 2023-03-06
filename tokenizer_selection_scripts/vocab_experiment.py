import json
import transformers
from transformers import AutoTokenizer, AutoConfig
from tokenizers import Tokenizer, SentencePieceBPETokenizer, ByteLevelBPETokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, load_from_disk
import os


def train_tokenizer(tokenizer_name, iterator, special_tokens, vocab_size):
    
    if tokenizer_name == "BPE":
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer_trainer = BpeTrainer(vocab_size=int(vocab_size),  special_tokens=special_tokens)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(iterator, trainer=tokenizer_trainer, vocab_size=vocab_size)
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    
    elif tokenizer_name == "WordPiece":
        tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
        tokenizer_trainer = WordPieceTrainer(vocab_size=int(vocab_size),  special_tokens=special_tokens)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(iterator, trainer=tokenizer_trainer, vocab_size=vocab_size)
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    
    elif tokenizer_name == "SentencePiece":
        tokenizer = SentencePieceBPETokenizer(unk_token="<unk>")
        tokenizer.train_from_iterator(iterator, vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        
    elif tokenizer_name == 'ByteLevelBPE':
        tokenizer = Tokenizer(ByteLevelBPETokenizer(unk_token="<unk>"))
        tokenizer.train_from_iterator(iterator, vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer = tokenizer.train_new_from_iterator(iterator, vocab_size=vocab_size)
    
    #
    #print(type(tokenizer))
    
    return tokenizer
        
def calculate_split_ratio(list_sents, tok_in):
    
    split_ratio_sent = []
    split_ratio_word = []
    sent_lens = []
    for sent in list_sents:
        sent_len = 0
        for w in sent.split(' '):
            toked = tok_in.tokenize(w)
            split_ratio_word.append(toked.__len__())
            sent_len += toked.__len__()
        
        #
        sent_lens.append(sent_len)
        sent_ratio = sent_len / sent.split(' ').__len__()
        split_ratio_sent.append(sent_ratio)
    
    return np.mean(sent_lens), np.mean(split_ratio_sent), np.mean(split_ratio_word)

def get_text_of_instance(ex_in):
    
    text = ''
    for feat in ex_in:
        if feat in ['sentence', 'sentence1', 'sentence2', 'question', 'question1', 'question2', 'hypothesis', 'premise']:
            text += ex_in[feat]
            text += ' '
    
    return text

def get_glue_tasks():
    return ['cola', 'sst2', 'mrpc', 'stsb', 'rte', 'wnli']#['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']

def calculate_unk_occurances(tok_in):
    subsets = get_glue_tasks()
    count_unk = {i: 0 for i in subsets}
    
    for subset in subsets:
        data_i = load_dataset(
            'glue',
            subset,
        )
        
        #
        total_instances = 0
        total_unk = 0
        for split in data_i:
            for instance in data_i[split]:
                total_instances += 1
                text = get_text_of_instance(instance)
                text_toked = tok_in.tokenize(text)
                if ('<unk>' in text_toked) or ("[UNK]" in text_toked):
                    total_unk += 1
        
        #
        count_unk[subset] = total_unk / total_instances
    
    return count_unk

def get_iterator(dict_data):
    list_sents = []
    for split in ['train', 'test', 'validation']:
        list_sents += dict_data[split]['TEXT']
    
    return list_sents

def get_random_examples(list_in, size):
    print(type(list_in))
    indices = np.random.choice(range(len(list_in)), size, replace=False)
    data_sampled = []
    for index in indices:
        data_sampled.append(list_in[index])
    
    return data_sampled

def score_tokenizer_output(tok_in):
    
    ref_dict = {
        'cooking': {
            'parts': ['cook', 'ing']
        },
        'dangerous': {
            'parts': ['danger', 'ous']
        },
        'pretext': {
            'parts': ['pre', 'text']
        },
        'fitness': {
            'parts': ['fit', 'ness']
        },
        'antisocial': {
            'parts': ['anti', 'social']
        },
        'podium': {
            'parts': ['pod', 'ium']
        },
        'universe': {
            'parts': ['uni', 'verse']
        },
        'european': {
            'parts': ['europ', 'ean']
        },
        'decode': {
            'parts': ['de', 'code']
        },
        'subvert': {
            'parts': ['sub', 'vert']
        },
        'proactive': {
            'parts': ['pro', 'active']
        },
        'concentric': {
            'parts': ['con', 'centr', 'ic']
        },
        'octopus': {
            'parts': ['octo', 'pus']
        },  
    }
    
    #
    score = 0
    for word in ref_dict:
        toked = tok_in.tokenize(word)
        #
        if (len(toked) == 1) or (len(toked) == len(word)):
            score -= 1
        else:
            for t_ in toked:
                if t_ in ref_dict[word]['parts']:
                    score += 1
    
    return score


def run_vocabulary_experiment():
    
    # read pre-training data
    """
    with open('./../../../../from_shala/vocabulary_analysis/data_filtration/Filtration_15Nov2022/ALL_FILTERED_DATA/processed_data.json', 'r') as f:
        data = json.load(f)
    """
    print(f"\nReading data...")
    path_data = './../pretraining_data_free_text_08Jan2022'
    print(f"Reading data from {path_data}")
    data = load_from_disk(path_data)
    print("DONE!")

    #
    tokens_special = ['<s>', '</s>', '<mask>', '<pad>', '<unk>', '<cls>'] + [f'<extra_id_{i}>' for i in range(0, 100)]
    
    #
    tokenizer_names = [
        # 'BPE',
        # 'WordPiece',
        'SentencePiece',
        # 'ByteLevelBPE',
        #'bert-base-uncased',
        #'phueb/BabyBERTa-1', 
        #'prajjwal1/bert-tiny', 
        #'distilbert-base-uncased', 
        # 'roberta-base',
        #'t5-base',
        #'albert-base-v2',
    ]
    vocab_sizes = [1000, 5000, 10000, 20000, 30000, 50000, 70000, 100000]#[i for i in range(5000, 15001, 1000)]
    total_exp = (len(tokenizer_names) * len(vocab_sizes)) + (len(tokenizer_names))
    df_exp = pd.DataFrame(
        -1,
        index=range(total_exp),
        columns=['tokenizer', 'vocab_size', 'sentence_length', 'sentence_split_ratio', 'word_split_ratio', 'manual_evaluation']+get_glue_tasks(),
    )

    # get iterator
    print(f"\nCreating an iterator from the data...")
    iterator = get_iterator(data)
    print("DONE!")
    print(f"\nSampling examples for evaluation...")
    sampled_examples = get_random_examples(iterator, 5000)
    print("DONE!")
    

    #
    tokenizers = {}
    df_idx = 0
    for name in tqdm(tokenizer_names):

        if name in ['BPE', 'WordPiece', 'SentencePiece', 'roberta-base', 'bert-base-uncased', 't5-base', 'ByteLevelBPE']:

            for vocab_size in vocab_sizes:
                df_idx += 1

                # train a tokenizer
                tok_i  = train_tokenizer(name, iterator, tokens_special, vocab_size)
                tokenizers[f'{name}_{vocab_size}'] = deepcopy(tok_i)

                # calculate stats
                sent_len, sr_sent, sr_word = calculate_split_ratio(sampled_examples, tok_i)

                # calculate <unk> instances in glue
                #unk_count = calculate_unk_occurances(tok_i)

                # manual evaluation
                manual_score = score_tokenizer_output(tok_i)

                #
                df_exp.loc[df_idx, 'tokenizer'] = name
                df_exp.loc[df_idx, 'vocab_size'] = vocab_size
                df_exp.loc[df_idx, ['sentence_length', 'sentence_split_ratio', 'word_split_ratio']] = [sent_len, sr_sent, sr_word]
                df_exp.loc[df_idx, 'manual_evaluation'] = manual_score
                #for subset in get_glue_tasks():
                #    df_exp.loc[df_idx, subset] = unk_count[subset]
                
                #
                #if vocab_size == 12000:
                if not os.path.exists(os.path.join('./Tokenizer_files_free_text', f'{name}_{vocab_size}')):
                    os.makedirs(os.path.join('./Tokenizer_files_free_text', f'{name}_{vocab_size}'))

                dir_save = os.path.join('./Tokenizer_files_free_text', f'{name}_{vocab_size}')
                try:
                    tokenizer[f'{name}_{vocab_size}'].save_pretrained(dir_save)
                    print('save_pretrain')
                except:
                    try:
                        tokenizers[f'{name}_{vocab_size}'].save_vocabulary(dir_save)
                        print('vocab')
                    except:
                        print('could not save tokenizer')
                #
                with open(os.path.join(dir_save, 'config.json'), 'w') as f:
                    json.dump({}, f)

        #
        if not name in ['BPE', 'WordPiece', 'SentencePiece']:
            df_idx += 1

            #
            tok_i = AutoTokenizer.from_pretrained(name)
            tokenizers[f'{name}_pretrained'] = deepcopy(tok_i)

            # calculate stats
            sent_len, sr_sent, sr_word = calculate_split_ratio(sampled_examples, tok_i)

            # calculate <unk> instances in glue
            unk_count = calculate_unk_occurances(tok_i)

            # manual evaluation
            manual_score = score_tokenizer_output(tok_i)

            #
            df_exp.loc[df_idx, 'tokenizer'] = name
            df_exp.loc[df_idx, 'vocab_size'] = 'pretrained'
            df_exp.loc[df_idx, ['sentence_length', 'sentence_split_ratio', 'word_split_ratio']] = [sent_len, sr_sent, sr_word]
            df_exp.loc[df_idx, 'manual_evaluation'] = manual_score
            
            # @TODO: uncomment following line if we want to check occurrence of unk token
            #for subset in get_glue_tasks():
            #    df_exp.loc[df_idx, subset] = unk_count[subset]
            
        # save data
        df_exp.to_csv('Vocabulary_experiment_results_free_text_08Jan23_2.csv')
    
    return

def filter_for_tokenizer_type(df_in, tok_names):
    
    df_list = []
    for name in tok_names:
        df_list.append(df_in.loc[df_in.loc[:, 'tokenizer'] == name, :])
    
    return pd.concat(df_list)

def plot_compare_vocabulary_sizes(df_in, tok_type):
    glue_tasks = get_glue_tasks()
    
    #
    fig, ax = plt.subplots(figsize=(12, 6), dpi=64)
    sns.barplot(
        data=df_in,
        x='vocab_size',
        y='sentence_length',
        hue='tokenizer'
    )
    plt.xlabel('Vocabulary size', fontsize=22)
    plt.ylabel(f'Length of tokenized sentence', fontsize=22)
    plt.xticks(rotation=90)

    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)

    plt.savefig(f'./EDA_plots/{tok_type}_VocabSize_vs_SentenceLength.jpg')
    
    
    #
    for task in glue_tasks:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=64)
        sns.barplot(
            data=df_in,
            x='vocab_size',
            y=task,
            hue='tokenizer'
        )
        plt.xlabel('Vocabulary size', fontsize=22)
        plt.ylabel(f'Fraction of instances with <unk> token', fontsize=22)
        plt.xticks(rotation=90)

        # Set tick font size
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)

        plt.savefig(f'./EDA_plots/{tok_type}_VocabSize_vs_UnkToken_{task}.jpg')
    
    
    #
    fig, ax = plt.subplots(figsize=(12, 6), dpi=64)
    sns.barplot(
        data=df_in,
        x='vocab_size',
        y='word_split_ratio',
        hue='tokenizer'
    )
    plt.xlabel('Vocabulary size', fontsize=22)
    plt.ylabel(f'Number of tokens per word', fontsize=22)
    plt.xticks(rotation=90)

    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)

    plt.savefig(f'./EDA_plots/{tok_type}_VocabSize_vs_WordSplitRatio.jpg')
    
    #
    fig, ax = plt.subplots(figsize=(12, 6), dpi=64)
    sns.barplot(
        data=df_in,
        x='vocab_size',
        y='manual_evaluation',
        hue='tokenizer'
    )
    plt.xlabel('Vocabulary size', fontsize=22)
    plt.ylabel(f'Manual evaluation score', fontsize=22)
    plt.xticks(rotation=90)

    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)

    plt.savefig(f'./EDA_plots/{tok_type}_VocabSize_vs_ManualEval.jpg')
    
    return
    

if __name__ == '__main__':
    run_vocabulary_experiment()
    