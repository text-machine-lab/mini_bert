import pandas
from datasets import Dataset
import json
import utils
with open('processed_data.json', 'r') as f:
    data = json.load(f)

df = pandas.DataFrame.from_dict(data, orient="index")
df = utils.group_sentences(df['TEXT'])
df = pandas.DataFrame({"TEXT": df})
formatted_data=Dataset.from_pandas(df)

formatted_data = formatted_data.train_test_split()
formatted_data_test= formatted_data["test"].train_test_split()
formatted_data["validation"]=formatted_data_test["train"]
formatted_data["test"]=formatted_data_test["test"]
formatted_data.save_to_disk("formatted_data")

