from copy import deepcopy
import random
import torch


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
    if 'sentence' in example.features:
            sentence = example['sentence']
        
        else:
            t1 = list(example.features)[0]
            t2 = list(example.features)[1]
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
        

def filter_glue_dataset(dataset_name, aochildes_vocab_path="../data/AOChildes_word_frequency.csv"):
    """Filters GLUE datasets based on AOChildes vocabulary
    Args:
        dataset_name: name of GLUE dataset
        aochildes_vocab_path: the path of AOChildes vocabulary
    """
    
    dataset = load_dataset('glue',dataset_name)
    vocab_freq = pd.read_csv(aochildes_vocab_path)
    vocab_set = set(vocab_freq['word'])
    contractions = set(['nt','s','re','t','d','ll'])
    
    return dataset.filter(lambda example: filter_example(example, vocab_set,contractions))

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
