import pandas
from datasets import Dataset
import json

with open('processed_data.json', 'r') as f:
    data = json.load(f)

df = pandas.DataFrame.from_dict(data, orient="index")

formatted_data=Dataset.from_pandas(df)
formatted_data.save_to_disk("formatted_data")
