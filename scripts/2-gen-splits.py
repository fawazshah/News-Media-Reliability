import itertools
import json
import numpy as np
import pandas as pd

df = pd.read_csv('../data/emnlp18/corpus-balanced-classes.tsv', sep='\t')
all_urls = df["source_url_normalized"].values

output = {}

splits = np.array_split(all_urls, 5)
unpacked_splits = list(itertools.chain(*splits))
split_size = len(splits[0])

for i in range(5):
    train = unpacked_splits[:i*split_size] + unpacked_splits[(i+1)*split_size:]
    test = unpacked_splits[i*split_size:(i+1)*split_size]
    train_test_split = {"train": train, "test": test}
    output[f"{i}"] = train_test_split

with open("../data/emnlp18/splits-balanced-classes.json", "w") as f:
    json.dump(output, f)
