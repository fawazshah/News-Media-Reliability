import pandas as pd

# Count instances of each bias type in dataset

df = pd.read_csv('../data/emnlp18/corpus.tsv', sep='\t')
print(df['bias'].value_counts())

# Remove left-center and right-center

df = df[(df.bias != 'left-center') & (df.bias != 'right-center')]
print(df['bias'].value_counts())

# Merge extreme-right with right and extreme-left with left

df.loc[(df.bias == 'extreme-right'), 'bias'] = 'right'
df.loc[(df.bias == 'extreme-left'), 'bias'] = 'left'
print(df['bias'].value_counts())

# Write out

df.to_csv('../data/emnlp18/corpus-modified.tsv', sep='\t')
