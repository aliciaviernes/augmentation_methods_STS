"""
    NOTE TO SELF special token for detokenization is "▁"
    ''.join(item).replace('▁', ' ')
"""

import pandas as pd
import noising
from transformers import T5Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# LOAD PRETRAINED TOKENIZER
t5_tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')

# Path to our dataset
path = "path/to/lvl2/"
output_name = "input_data.h5"

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

data_clear = list()
limit = 500

# Data Clear is tokenized.
print("Tokenizing data...")
with open(path, 'rt') as f:
    for line in f:
        data_clear.append(t5_tokenizer.encode(line))  
        # use encode instead of tokenize - remove encode function from rachas

data_right_length = list()

# Truncation of length (goes to next list).
for item in tqdm(range(len(data_clear)), desc="Truncating"):
    if len(data_clear[item]) <= limit:
        data_right_length.append(data_clear[item])
    else:
        chnks = list(chunks(data_clear[item], limit))
        data_right_length.extend(chnks)

original_len = len(data_right_length)
print(f"There are {original_len} datapoints!")

print("Doubling datapoints...")  # Doubling
data_triple = data_right_length.copy()
data_triple.extend(data_right_length)
data_triple.extend(data_right_length)

print(f"Now there are more: {len(data_triple)} datapoints!")
print(f"Two times as many? {len(data_triple) / original_len == 2}.")

data_masked = list()
for sample in tqdm(range(len(data_triple)), desc="Text noising progress"):
    i, o = noising.add_noise(data_triple[sample], tokenizer=t5_tokenizer, percent=0.15)
    row = [i, o]
    data_masked.append(row)

print("Create DataFrame...")
vector_df = pd.DataFrame(data_masked, columns=['input', 'output'])

print("Saving Dataframe...")
vector_df.to_hdf(output_name, key='df', mode='w')
