import scipy as sp
import numpy as np
import pandas as pd
import json
import os
import glob
from datasets import load_dataset
from tqdm import tqdm
#from english_words import english_words_set
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
#import matplotlib.pyplot as plt
#import seaborn as sns
#from scipy.stats import zipf
#from scipy.optimize import curve_fit
import nltk
from nltk.corpus import words
from nltk.tokenize import sent_tokenize
from nltk import FreqDist, word_tokenize, wordpunct_tokenize
import re 
import random
import datasets
from copy import deepcopy

# regexps for text preprocessing
ONLY_ALPHA = re.compile(r'([^\s\w]|_)+')
NUMBERS = re.compile(r'\b\d+')
MULTISPACE = re.compile(r'[^\S\r\n]{2,}')
AT_DIGIT = re.compile(r'@[,.]@')
AT_HYPHEN = re.compile(r'@-@')
WORD = re.compile(r'\w+')
HLINK = re.compile(r'http\S+')

def preprocess(line):
    #line = line.replace('``', '"')
    #line = line.replace("''", '"')
    #line = AT_HYPHEN.sub('-', line)
    line = line.replace('-', ' ')
    line = AT_DIGIT.sub('#NUMBER', line)
    line = NUMBERS.sub('#NUMBER ', line)
    line = HLINK.sub('#HLINK', line)
    #line = MULTISPACE.sub(' ', line)
    #line = line.lstrip(' ')
    return line

def KL_divergence(p1, p2):
    return sum(p1 * np.log((p1/p2)))

def sample_articles(list_indices, sample_size, seed_value=0):
    np.random.seed(seed_value)
    return np.random.choice(a=list_indices, size=sample_size, replace=False).tolist()

def to_paragraphs(list_docs):
    list_paras = []
    for doc_i in list_docs:
        list_paras += doc_i.split('\n\n')
    
    return list_paras

def to_sentences(list_docs):
    list_sents = []
    for doc_i in list_docs:
        list_sents += sent_tokenize(doc_i)
    
    return list_sents

def to_spans(
    list_docs,
    span_length=110,
    stride=30,
):
    list_spans = []
    for doc in list_docs:
        words = doc.split(' ')
        for start in range(0, len(words), stride):
            end = min(start+span_length, len(words))
            span = ' '.join(words[start:end])
            list_spans.append(span)
    
    return list_spans

def to_count_vectors(list_docs, english_vocab=[]):
    
    #
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list_docs)
    doc_term = pd.DataFrame(X.toarray())
    doc_term.columns = vectorizer.get_feature_names_out()
    
    # 
    if len(english_vocab) > 0:
        common_words = set(doc_term.columns.tolist()).intersection(set(english_vocab))
        keep_words = sorted(list(common_words))
    else:
        keep_words = sorted(doc_term.columns.tolist())
    
    return doc_term.loc[:, keep_words]

def filter_doc_term(
    dt_in, 
    keep_words, 
    sentence_length=40, #4 if we are splitting at sentence-level
):
        
    # remove sentences with unwanted words
    all_words = dt_in.columns.tolist()
    rem_words = set(all_words) - set(keep_words)
    dt_out = dt_in.loc[dt_in.loc[:, rem_words].sum(axis=1) == 0, :]
    
    # remove sentences with less than 6 words
    dt_out = dt_out.loc[dt_out.loc[:, :].sum(axis=1) > sentence_length, :]
    
    # sort columns
    sorted_words = sorted(dt_out.columns.tolist())
    dt_out = dt_out.loc[:, sorted_words]
    
    return dt_out

def identify_limited_vocab_docs(list_docs, limited_vocab, english_vocab, batch_size=1000):    
    df_list = []
    selected_indices = []
    selected_docs = []
    for start_ in range(0, len(list_docs), batch_size):
        end_ = min(len(list_docs), start_ + batch_size)
        batch = list_docs[start_:end_]
        batch_ = [preprocess(i) for i in batch]
        
        #
        df_i = to_count_vectors(batch_, english_vocab)
        df_i = df_i.reset_index(drop=True)
        df_i.index = [i for i in range(start_, end_, 1)]
        df_i = filter_doc_term(
            df_i, 
            keep_words=limited_vocab,
            sentence_length=40, #=4 if we are splitting at sentence-level
        )
        
        #
        if df_i.shape[0] > 0:
            selected_indices += df_i.index.tolist()
            #df_list.append(df_i)
    
    #
    #doc_term_mat = pd.concat(df_list)
    #doc_term_mat = doc_term_mat.fillna(0)
    #doc_term_mat = doc_ter_mat.reset_index(drop=True)
    selected_docs += [list_docs[i] for i in selected_indices]
    
    return selected_docs, selected_indices#, doc_term_mat

def get_vocabs():
    
    vocab = pd.read_csv('CHILDES_vocab_age.csv')
    vocab = vocab.dropna()

    #
    age_discrete = []
    for age in vocab.loc[:, 'age']:
        age_lb = 0
        for age_ub in range(7):
            if age_lb < age <= age_ub:
                age_discrete.append(age_ub)
            age_lb = age_ub

    #
    vocab['age_discrete'] = age_discrete
    #print('='*10)
    #print(vocab.groupby('age_discrete').count().loc[:, 'word'])

    #
    vocab_of_interest = [i.lower() for i in vocab.word.unique().tolist()]
    english_vocab = pd.read_csv('wikipedia_vocab_regex_based.csv').iloc[:, 1].values.tolist()#list(set(words.words()).union(set(vocab_of_interest)))
    #english_vocab = list(set(words.words()).union(set(vocab_of_interest)))

    #
    #print('='*10)
    #print(f'words in vocab of interest: {len(vocab_of_interest)}')
    #print(f'words in english vocab: {len(english_vocab)}')#
    
    return vocab_of_interest, english_vocab
    
def get_filtered_sentences(
    list_docs, 
    save_every_steps, 
    filename_save,
    batch_size=10000, #10k if we split at sentence-level, 1k if split at span level
    save_=True
):
    
    #
    vocab_of_interest, english_vocab = get_vocabs()
    
    #
    all_filtered_sentences = []
    all_sentences = 0
    batch_idx = 0
    #for start in tqdm(range(0, len(list_docs), batch_size)):
    for start in range(0, len(list_docs), batch_size):
        end = min(len(list_docs), start+batch_size)
        batch_idx += 1
        articles = list_docs[start:end]
        articles = [i.lower() for i in articles]
        
        # @TODO: pay attention here, do we want to split one article by sentences or by spans?
        #sentences = to_sentences(articles)
        sentences = to_spans(articles)
        
        #
        filtered, filtered_idx = identify_limited_vocab_docs(
            list_docs=sentences, 
            limited_vocab=vocab_of_interest,
            english_vocab=[],#english_vocab,
        )
        all_filtered_sentences += filtered
        all_sentences += len(sentences)

        if batch_idx%save_every_steps == 0:
            percentage = len(all_filtered_sentences)/all_sentences
            all_filtered_sentences = list(set(all_filtered_sentences))
            if save_:
                with open(filename_save, 'w') as f:
                    json.dump(all_filtered_sentences, f)
                #
                print(f'Total number of sentences processed: {all_sentences}')
                print(f'Total number of sentences selected: {len(all_filtered_sentences)}')
                print(f'Percentage of selected sentences: {percentage*100}%')
    
    #
    percentage = len(all_filtered_sentences)/all_sentences
    all_filtered_sentences = list(set(all_filtered_sentences))
    if save_:
        with open(filename_save, 'w') as f:
            json.dump(all_filtered_sentences, f)

        #
        print(f'Total number of sentences processed: {all_sentences}')
        print(f'Total number of sentences selected: {len(all_filtered_sentences)}')
        print(f'Percentage of selected sentences: {percentage*100}%')
    
    return all_filtered_sentences


def filter_wikipedia():
    
    # get wiki data
    wiki_data = load_dataset("wikipedia", "20220301.en")
    wiki_docs = wiki_data['train']['text']
    
    #
    _ = get_filtered_sentences(wiki_docs, 3, './Filtration_25Dec2022/Wikipedia/wikipedia_filtered_sentences.json')
    
    return


def filter_bookcorpus():
    
    # get bookcorpus data
    bc_data = load_dataset("bookcorpus")
    bc_docs = bc_data['train']['text']
    
    #
    _ = get_filtered_sentences(bc_docs, 100, './Filtration_25Dec2022/BookCorpus/bookcorpus_filtered_sentences.json')
    
    return

def filter_cbt():
    
    # get bookcorpus data
    cbt_data = load_dataset("cbt", "raw")
    cbt_docs = []
    for split in ['train', 'test', 'validation']:
        cbt_docs += [i for i in cbt_data[split]['content']]
    
    #
    _ = get_filtered_sentences(cbt_docs, 100, './Filtration_25Dec2022/CBT/cbt_filtered_sentences.json')
    
    
    return

def filter_simplified_wiki():
    
    # collect all file names
    read_path = os.path.join('.', 'wikipedia-extractor', 'text')
    all_files = []
    all_files += glob.glob(os.path.join(read_path, 'AA')+'/*')
    all_files += glob.glob(os.path.join(read_path, 'AB')+'/*')
    
    # collect all page content
    all_docs = []
    for file in tqdm(all_files):
        with open(file, 'r') as f:
            text_i = f.read()
        #
        docs_i = text_i.split('</doc>')
        for idx, doc_i in enumerate(docs_i):
            parts = doc_i.split('\">\n')
            all_docs.append(parts[-1])
    
    #
    _ = get_filtered_sentences(all_docs, 10, './Filtration_25Dec2022/WikipediaSimplified/simplified_wikipedia_filtered_sentences.json')
    
    return

def filter_c4():
    
    for split in ['train', 'validation']:
        print('\n')
        print(f'Starting filtration of {split} split of the C4 data')
        print('\n')
        dataset = load_dataset("c4", "en", split=split, streaming=True)
        batch_filtered = []
        batch_size = 100000 #10k if we are splitting at sentence-level
        batch_idx = 0
        starting_point = 0#12000000 # 12 mil.
        for start in tqdm(range(starting_point, dataset.dataset_size, batch_size)):
            end = min(dataset.dataset_size, start + batch_size)
            batch_idx += 1
            
            # batch of docs
            batch = []
            dataset_i = dataset.skip(start)
            for idx, i in enumerate(iter(dataset_i)):
                if (idx%batch_size == 0) and (idx != 0):
                    batch += [i['text']]
                    break
                else:
                    batch += [i['text']]

            # get filtered sentences
            filtered_sent = get_filtered_sentences(
                list_docs=batch, 
                save_every_steps=100000, 
                filename_save=f'{end}.json',
                save_=False,
            )
            batch_filtered += filtered_sent

            # save sentences
            if batch_idx%1 == 0:
                print("Saving current collection of filtered sentences")
                path_ = './Filtration_25Dec2022/C4'
                filename_save = os.path.join(path_, f'c4_{split}_{start}_{end}_filtered_sentences.json')
                with open(filename_save, 'w') as f:
                    json.dump(batch_filtered, f)

                #
                batch_filtered = []
            
    
    return

def check_written_processed_files():
    
    try:
        with open('./Filtration_25Dec2022/ALL_FILTERED_DATA/processed_files.json', 'r') as f:
            processed_files = json.load(f)
    except:
        processed_files = {}

    try:
        with open('./Filtration_25Dec2022/ALL_FILTERED_DATA/processed_data.json', 'r') as f:
            dict_data = json.load(f)
    except:
        dict_data = {}
        
    
    return dict_data, processed_files

def featurize_docs(list_docs, dict_in, processed_files, file, vocab_of_interest, age2words, idx_file, batch_size=10000):
    
    #
    data_source = file.split('_')[0]
    
    #
    for start in tqdm(range(0, len(list_docs), batch_size)):
        end = min(start + batch_size, len(list_docs))
        if end <= processed_files[file]:
            continue
        
        # scoop out batch
        batch = list_docs[start: end]
        
        # convert current batch of docs to a doc-term matrix
        df_batch = to_count_vectors(batch)
        df_batch.loc[:, :] = (df_batch.loc[:, :].values > 0).astype(int)
        
        # find oov and iov words for the current batch of docs
        all_words = df_batch.columns.tolist()
        iov = sorted(list(set(vocab_of_interest).intersection(set(all_words))))
        oov = sorted(list(set(all_words) - set(vocab_of_interest)))
        
        # find length of the sentence (FEATURE WE WANT)
        df_batch['SENTENCE_LENGTH'] = df_batch.loc[:, all_words].sum(axis=1)
        
        # change 1s present in the DF to identify occurence of a word by corresponding age of the word 
        # (This age of the word is take from the AIChildes dataset)
        for age in age2words:
            age_w = age2words[age]
            age_w = set(age_w).intersection(iov)
            df_batch.loc[:, age_w] = df_batch.loc[:, age_w].values * age
            
        # calculated required features
        df_batch['TOTAL_AGE'] = df_batch.loc[:, iov].sum(axis=1)
        df_batch['OOV_COUNT'] = df_batch.loc[:, oov].sum(axis=1)
        df_batch['AVG_AGE'] = df_batch.loc[:, 'TOTAL_AGE'].values / df_batch.loc[:,'SENTENCE_LENGTH'].values
        df_batch['DATA_SOURCE'] = data_source
        df_batch['OOV_WORDS'] = [[] for x in range(len(df_batch))]
        df_batch['TEXT'] = batch
        
        # in addition to above features, we will also keep a list of OOV words for every sentence
        # NOTE: This might be little time consuming but will be useful to double check things later on
        df_oov = df_batch.loc[df_batch.loc[:, 'OOV_COUNT'].values > 0, oov]
        for i_oov in df_oov.index:
            w_oov = np.where(df_oov.loc[i_oov, oov].values > 0)[0].tolist()
            df_batch.at[i_oov, 'OOV_WORDS'] = np.array(oov)[w_oov].tolist()
        
        # Finally, filter the DF (take only the columns of interest) 
        # and convert the filtered DF to a dictionary
        df_batch.index = [f'{data_source}_{idx_file}_{i}' for i in range(start, end, 1)]
        cols_of_interest = ['TEXT', 'DATA_SOURCE', 'SENTENCE_LENGTH', 'TOTAL_AGE', 'AVG_AGE', 'OOV_COUNT', 'OOV_WORDS']
        df_batch = df_batch.loc[:, cols_of_interest]
        dict_batch = df_batch.T.to_dict()
        
        # merge dictionaries
        dict_in = {**dict_in, **dict_batch}
        
        # update the index of processed file
        processed_files[file] = max(processed_files[file], end)
    
    
    return dict_in, processed_files

def get_age2vocab():
    
    #
    vocab = pd.read_csv('CHILDES_vocab_age.csv')
    vocab = vocab.dropna()
    age_discrete = []
    for age in vocab.loc[:, 'age']:
        age_lb = 0
        for age_ub in range(7):
            if age_lb < age <= age_ub:
                age_discrete.append(age_ub)
            age_lb = age_ub
    vocab['age_discrete'] = age_discrete
    
    #
    age2words = {i: [] for i in vocab.loc[:, 'age_discrete'].unique().tolist()}
    for age in vocab.loc[:, 'age_discrete'].unique():
        v_ = vocab.loc[vocab.loc[:, 'age_discrete'] == age, :].loc[:, 'word'].unique().tolist()
        age2words[age] += v_
    
    return age2words

def process_filtered_sentences():
    
    # read vocabulary and create a map age2words
    vocab_of_interest, vocab_english = get_vocabs()
    age2words = get_age2vocab()
    
    # check how much data is previously processed
    dict_data, processed_files = check_written_processed_files()
    
    # read all .json files
    all_files = []
    print('here')
    for dir_ in ['BookCorpus', 'CBT', 'Wikipedia', 'WikipediaSimplified', 'C4']:
        path_ = os.path.join('./Filtration_25Dec2022', dir_)
        all_files += glob.glob(f'{path_}/*.json')
    
    # iterate over all files
    for idx_file, file in enumerate(all_files):
        
        # read the file
        with open(file, 'r') as f:
            data_i = json.load(f)
        
        # check if the current file is already processed
        if file in processed_files:
            max_processed_index = processed_files[file]
            if max_processed_index >= (len(data_i)-1):
                print(f'Skipping {file} because it is already processed')
                continue
        else:
            processed_files[file] = 0
        
        
        # @TODO: remove this when done checking the function
        #if ('c4_train_' in file):# or ('bookcorpus_' in file) or ('wikipedia' == file.split('_')[0]):
        #    continue
        
        
        print(f'Starting extraction from the {file} file')
        dict_data, processed_files = featurize_docs(data_i, dict_data, processed_files, file, vocab_of_interest, age2words, idx_file)
        
        # save the files
        with open('./Filtration_25Dec2022/ALL_FILTERED_DATA/processed_data.json', 'w') as f:
            json.dump(dict_data, f, indent=4)
        
        with open('./Filtration_25Dec2022/ALL_FILTERED_DATA/processed_files.json', 'w') as f:
            json.dump(processed_files, f, indent=4)
    
    return

def main(dataset_name):
    
    if dataset_name == 'wikipedia':
        filter_wikipedia()
    elif dataset_name == 'bookcorpus':
        filter_bookcorpus()
    elif dataset_name == 'wikipedia_simplified':
        filter_simplified_wiki()
    elif dataset_name == 'cbt':
        filter_cbt()
    elif dataset_name == 'c4':
        filter_c4()
    elif dataset_name == 'process_sentences':
        process_filtered_sentences()
    
    return


def sample_unconstrained_spans(list_docs, limit_spans, save_path, filename, limit_docs, save=True, save_every_steps=1000):
    
    #
    if not limit_docs:
        sample_size = 3
    else:
        sample_size = int(limit_spans / limit_docs) + 1
    
    #
    filename_save = os.path.join(save_path, filename)
    
    #
    all_text = ' '.join(list_docs)
    all_spans = to_spans([all_text])
    sampled_spans = list(np.random.choice(all_spans, size=min(sample_size, len(all_spans)), replace=False))
    
    #
    #print((type(all_text), len(all_text)))
    #print((type(all_spans), len(all_spans)))
    #print(all_spans[:10])
    #print((type(sampled_spans), len(sampled_spans)))
    
    """
    for doc_idx, doc in enumerate(list_docs):
        text_i = doc.lower()
        spans_i = to_spans([text_i])
        sampled_i = np.random.choice(spans_i, size=min(sample_size, len(spans_i)), replace=False)
        print(len())
        
        # store
        all_spans += spans_i
        sampled_spans += list(sampled_i)
        
        #
        if len(sampled_spans) >= limit_spans:
            break

        # save sentences
        if save and (doc_idx%save_every_steps == 0):
            print("Saving current collection of sentences...")
            with open(filename_save, 'w') as f:
                json.dump(sampled_spans, f)
    """
    #
    if save:
        print(f"\nSaving final collection of {len(sampled_spans)} sentences...")
        with open(filename_save, 'w') as f:
            json.dump(sampled_spans, f)
    
    return sampled_spans

def sample_simplified_wikipedia(limit_spans, save_path, filename):
    
    #
    print('\n')
    print(f'Starting sampling spans from the Simplified-Wiki data...')
    print('\n')
    
    # collect all file names
    read_path = os.path.join('.', 'wikipedia-extractor', 'text')
    all_files = []
    all_files += glob.glob(os.path.join(read_path, 'AA')+'/*')
    all_files += glob.glob(os.path.join(read_path, 'AB')+'/*')
    
    # collect all page content
    all_docs = []
    for file in tqdm(all_files):
        with open(file, 'r') as f:
            text_i = f.read()
        #
        docs_i = text_i.split('</doc>')
        for idx, doc_i in enumerate(docs_i):
            parts = doc_i.split('\">\n')
            all_docs.append(parts[-1])
    
    #
    _ = sample_unconstrained_spans(
        list_docs=all_docs, 
        limit_spans=limit_spans, 
        save_path=save_path, 
        filename=filename, 
        limit_docs=1,
        save=True,
        save_every_steps=1000,
    )
    
    
    return


def sample_cbt(limit_spans, save_path, filename):
    
    #
    print('\n')
    print(f'Starting sampling spans from the CBT data...')
    print('\n')
    
    # get bookcorpus data
    cbt_data = load_dataset("cbt", "raw")
    cbt_docs = []
    for split in ['train', 'test', 'validation']:
        cbt_docs += [i for i in cbt_data[split]['content']]
    
    #
    _ = sample_unconstrained_spans(
        list_docs=cbt_docs, 
        limit_spans=limit_spans, 
        save_path=save_path, 
        filename=filename, 
        limit_docs=1,
        save=True,
        save_every_steps=1000,
    )
    
    
    return


def sample_bookcorpus(limit_spans, save_path, filename):
    
    # get bookcorpus data
    bc_data = load_dataset("bookcorpus")
    bc_docs = bc_data['train']['text']
    step_size = 100000
    
    #
    print('\n')
    print(f'Starting sampling spans from the BookCorpus data...')
    print('\n')
    
    #
    all_sampled = []
    total_ = 0
    for batch_idx, start_idx in tqdm(enumerate(range(0, len(bc_docs), step_size))):
        end_idx = min(start_idx + step_size, len(bc_docs))
        batch = bc_docs[start_idx:end_idx]
        sampled_i = sample_unconstrained_spans(
            list_docs=batch, 
            limit_spans=limit_spans, 
            save_path=save_path, 
            filename=filename, 
            limit_docs=limit_spans/(step_size-1),
            save=False,
            save_every_steps=1000,
        )
        
        #
        all_sampled += sampled_i
        total_ += len(sampled_i)
        
        #
        if total_ >= limit_spans:
            break
        
        #
        if batch_idx%2 == 0:
            print("Saving current collection of sentences...")
            filename_save = os.path.join(save_path, filename)
            filename_save = filename_save.replace(".json", f"_{start_idx}_{end_idx}.json")
            with open(filename_save, 'w') as f:
                json.dump(all_sampled, f)
            
            # restart
            all_sampled = []
        
    
    #
    print(f"\nSaving final collection of {total_} sentences...")
    filename_save = os.path.join(save_path, filename)
    filename_save = filename_save.replace(".json", f"_{start_idx}_{end_idx}.json")
    with open(filename_save, 'w') as f:
        json.dump(all_sampled, f)
    
    
    return
    

def sample_wikipedia(limit_spans, save_path, filename):
    
    # get wiki data
    wiki_data = load_dataset("wikipedia", "20220301.en")
    wiki_docs = wiki_data['train']['text']
    
    #
    print('\n')
    print(f'Starting sampling spans from the Wikipedia data...')
    print('\n')
    
    #
    sampled_spans = []
    total_ = 0
    start_idx = 0
    for doc_idx, doc in tqdm(enumerate(wiki_docs)):
        sampled_i = sample_unconstrained_spans(
            list_docs=doc, 
            limit_spans=limit_spans, 
            save_path=save_path, 
            filename=filename, 
            limit_docs=limit_spans*10,
            save=False,
            save_every_steps=1000,
        )
        
        # store
        sampled_spans += sampled_i
        total_ += len(sampled_i)

        #
        if total_ >= limit_spans:
            break
        
        #
        # save sentences
        if (doc_idx+1)%500000 == 0:
            print("Saving current collection of sentences...")
            filename_save = os.path.join(save_path, filename)
            filename_save = filename_save.replace(".json", f"_{start_idx}_{doc_idx}.json")
            with open(filename_save, 'w') as f:
                json.dump(sampled_spans, f)

            # restart
            start_idx = doc_idx + 1
            sampled_spans = []
    
    #
    print(f"\nSaving final collection of {total_} sentences...")
    filename_save = os.path.join(save_path, filename)
    filename_save = filename_save.replace(".json", f"_{start_idx}_{doc_idx}.json")
    with open(filename_save, 'w') as f:
        json.dump(sampled_spans, f)
        
    
    return


def sample_c4(limit_spans, save_path, filename):
    
    #
    sampled_spans = []
    total_ = 0
    for split in ['train']:
        print('\n')
        print(f'Starting sampling spans from the {split} split of the C4 data...')
        print('\n')
        dataset = load_dataset("c4", "en", split=split, streaming=True)
        
        
        #
        start_idx = 0
        for doc_idx, doc in tqdm(enumerate(iter(dataset))):
            text_i = [doc['text'].lower()]
            sampled_i = sample_unconstrained_spans(
                list_docs=text_i, 
                limit_spans=limit_spans, 
                save_path=save_path, 
                filename=filename, 
                limit_docs=10000000,
                save=False,
                save_every_steps=1000,
            )
            
            # store
            sampled_spans += sampled_i
            total_ += len(sampled_i)
            
            #
            if total_ >= limit_spans:
                break
            
            # save sentences
            if (doc_idx+1)%500000 == 0:
                print("Saving current collection of sentences...")
                filename_save = os.path.join(save_path, filename)
                filename_save = filename_save.replace(".json", f"_{start_idx}_{doc_idx}.json")
                with open(filename_save, 'w') as f:
                    json.dump(sampled_spans, f)
                
                # restart
                start_idx = doc_idx + 1
                sampled_spans = []
        
        #
        if len(sampled_spans) >= limit_spans:
            break
    
    #
    """
    if (len(sampled_spans) < limit_spans):# and (limit_spans < len()):
        remaining = limit_spans - len(sampled_spans)
        sampled_i = np.random.choice(all_spans, size=remaining, replace=False)
        sampled_spans += list(sampled_i)
    """
    
    #
    print(f"\nSaving final collection of {total_} sentences...")
    filename_save = os.path.join(save_path, filename)
    filename_save = filename_save.replace(".json", f"_{start_idx}_{doc_idx}.json")
    with open(filename_save, 'w') as f:
        json.dump(sampled_spans, f)
    
    
    return

def unconstrained_list_to_dict(list_in, data_in, file_name, start_key, data_source, target_size, batch_size=10000):
    for idx, i in enumerate(list_in):
        if idx <= target_size:
            data_in[start_key+idx] = {"TEXT": i, "DATA_SOURCE": data_source}
        else:
            break
    
    return data_in, idx

def process_unconstrained_data(fixed_seed_value=0):
    
    # limits for the data size for each corpus (matches the distribution over different corpora for the constrained language data)
    data_size_limit = {
        "CBT": 12000 + 20000,
        "WikipediaSimplified": 28000 + 50000,
        "Wikipedia": 707960 + 130000,
        "BookCorpus": 1770000 + 300000,
        "C4": 3980000+3000000 + 500000,
    }
    
    #
    random.seed(fixed_seed_value)
    all_files = []
    final_data = {}
    source_dist = {}
    for dir_ in ['CBT', 'WikipediaSimplified', 'Wikipedia', 'BookCorpus', 'C4']:
        path_ = os.path.join('./Unconstrained_language_10May2023', dir_)
        files = glob.glob(f'{path_}/*.json')
        random.shuffle(files)
        all_files += files
        source_dist[dir_] = 0
        target_size = deepcopy(data_size_limit[dir_])
        for file in tqdm(files):

            # read the file
            with open(file, 'r') as f:
                data_i = json.load(f)

            #
            final_data, size_i = unconstrained_list_to_dict(
                list_in=data_i, 
                data_in=final_data, 
                file_name=file,
                data_source=dir_,
                target_size=target_size,
                start_key=len(final_data),
            )
            
            #
            source_dist[dir_] += size_i
            target_size -= size_i
            if source_dist[dir_] >= data_size_limit[dir_]:
                break

        #
        with open('./Unconstrained_language_10May2023/ALL_FILTERED_DATA/processed_data.json', 'w') as f:
            json.dump(final_data, f, indent=4)

        with open('./Unconstrained_language_10May2023/ALL_FILTERED_DATA/processed_files.json', 'w') as f:
            json.dump(all_files, f, indent=4)
    
    #
    print(f"Distribution over various corpora...")
    for k_, v_ in source_dist.items():
        print(f"\nCorpora: {k_}")
        print(f"Number of examples: {v_}")
    
    return

def sample_unconstrained_data(fixed_seed_value=0):
    
    #
    random.seed(fixed_seed_value)
    np.random.seed(fixed_seed_value)
    
    #
    save_path = "./Unconstrained_language_10May2023"
    
    """
    # CBT
    sample_cbt(
        limit_spans=80000, 
        save_path=save_path, 
        filename="cbt_unconstrained_sentences.json",
    )
    
    # Simplified wikipedia
    sample_simplified_wikipedia(
        limit_spans=190000, 
        save_path=save_path, 
        filename="simplified_wikipedia_unconstrained_sentences.json",
    )
    
    # wikipedia
    sample_wikipedia(
        limit_spans=4800000, 
        save_path=save_path, 
        filename="wikipedia_unconstrained_sentences.json",
    )
    
    # book corpus
    sample_bookcorpus(
        limit_spans=12000000, 
        save_path=save_path, 
        filename="bookcorpus_unconstrained_sentences.json",
    )
    """
    
    # c4
    sample_c4(
        limit_spans=30000000,
        save_path=save_path, 
        filename="c4_unconstrained_sentences.json",
    )
    
    
    
    return

def group_sentences(
    dict_data,
    max_len=128,
    sentence_split_ratio=1.2,
    key='TEXT',
    data_size_limit=9281490,
):
    print('grouping sentences...')
    list_out = []
    cur_sequence = ''
    idx = 0
    for k_, v_ in tqdm(dict_data.items()):
        idx += 1
        new_sequence = cur_sequence + v_[key] + ' '
        if (len(new_sequence.split(' ')) * sentence_split_ratio) >= max_len:
            list_out.append(cur_sequence)
            cur_sequence = v_[key] + ' '
        else:
            cur_sequence = new_sequence

        #
        if len(list_out) >= data_size_limit:
            break
    
    print('done')
    print(f"Total examples in the grouped sentences are: {len(list_out)}")
    
    return list_out

def to_hf_data(
    datafile="./Unconstrained_language_10May2023/ALL_FILTERED_DATA/processed_data.json", 
    savedir='./../unconstrained_language_12May2023',
    fixed_seed_value=0,
):
    
    #
    random.seed(fixed_seed_value)
    
    # read data
    with open(datafile, 'r') as f:
        data_new = json.load(f)
    
    #
    data_new_list = group_sentences(data_new)
    random.shuffle(data_new_list)
    
    #
    train = []#data_old_list[:-150000]
    validation = []#data_old_list[-150000:-75000]
    test = []#data_old_list[-75000:]

    #
    train += data_new_list[:-200000]
    validation += data_new_list[-200000:-100000]
    test += data_new_list[-100000:]

    #
    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)
    
    #
    train_df = pd.DataFrame({
        "TEXT": train,
        #"FLAG": flag_train,
    })

    val_df = pd.DataFrame({
        "TEXT": validation,
        #"FLAG": flag_val,
    })

    test_df = pd.DataFrame({
        "TEXT": test,
        #"FLAG": flag_test,
    })
    
    #
    data_hf = datasets.DatasetDict()
    data_hf['train'] = datasets.Dataset.from_pandas(train_df)
    data_hf['validation'] = datasets.Dataset.from_pandas(val_df)
    data_hf['test'] = datasets.Dataset.from_pandas(test_df)
    
    #
    #data_hf.save_to_disk(savedir)
    
    #
    d_u = datasets.load_from_disk(savedir)
    
    return


if __name__ == "__main__":
    import argparse
    
    """
    #
    parser = argparse.ArgumentParser(description='Data filtration')
    parser.add_argument(
        '--dataset_name',
        default="c4",
        type=str,
        required=True,
        help='the path to workspace'
    )
    args, unknownargs = parser.parse_known_args()
    args = parser.parse_args()
    
    main(args.dataset_name)
    #filter_wikipedia()
    #filter_bookcorpus()
    #filter_cbt()
    #filter_simplified_wiki()
    #filter_c4()
    #process_filtered_sentences()
    """
    
    #
    #sample_unconstrained_data(fixed_seed_value=0)
    process_unconstrained_data()
    to_hf_data()