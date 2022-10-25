from copy import deepcopy
import random
import torch


def postprocess_text(preds, labels, original_code=None):
    """Use this function to postprocess generations and labels before BLEU computation."""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


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
    subset = random.sample(raw_datasets["train"], sample_size)
    raw_datasets["train"] = deepcopy(subset)
    if "validation" in raw_datasets:
        raw_datasets["validation"] = deepcopy(subset)
    if "test" in raw_datasets:
        raw_datasets["test"] = deepcopy(subset)
    return raw_datasets
