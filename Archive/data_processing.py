import pandas
from datasets import Dataset
import json
import utils

with open('data/new_data_raw/processed_data.json', 'r') as f:
    data = json.load(f)


def group_sentences(df, sentence_split_ratio=1.2, max_len=128):
    grouped_lists = []
    cur_sequence = ''
    for example in list(df['TEXT']):
        new_sequence = cur_sequence + example + ' '
        if (len(new_sequence.split(' ')) * sentence_split_ratio) >= max_len:
            grouped_lists.append(cur_sequence)
            cur_sequence = example + ' '
        else:
            cur_sequence = new_sequence
    return grouped_lists


df = pandas.DataFrame.from_dict(data, orient="index")
df = group_sentences(df)
df = pandas.DataFrame({"TEXT": df})
formatted_data = Dataset.from_pandas(df)

formatted_data = formatted_data.train_test_split(test_size=0.04)
formatted_data_test = formatted_data["test"].train_test_split(test_size=0.5)
formatted_data["validation"] = formatted_data_test["train"]
formatted_data["test"] = formatted_data_test["test"]
formatted_data.save_to_disk("formatted_data")
