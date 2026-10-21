# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: lt2326-2026-machine-learning-project (3.12.15)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Imports

# %%
import html
import json
import os
import re

import kagglehub
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

# %%
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:1')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f"Using device: {DEVICE}")

# %% [markdown]
# ### Download the lexicon
#
# The swear word lexicon is downloaded from [noswearing.com](https://www.noswearing.com/fulldict.xml).
#
# ```shell
# curl -s https://www.noswearing.com/fulldict.xml -o data/lexicon/noswearing_raw.xml
# ```

# %% [markdown]
# ### Process and clean the lexicon
#
# The lexicon downloaded is in XML format. The words are extracted and saved in a separate text file.
#
# Note that the lexicon contains some words that aren't swear words. These words are manually removed from the final lexicon.

# %%
lexicon_raw_filepath = './data/lexicon/noswearing_raw.xml'
lexicon_filepath = './data/lexicon/noswearing.txt'

with open(lexicon_raw_filepath, 'r', encoding='utf-8') as f:
    words = re.findall(r'<title>(.*?)</title>', f.read())

unique_words = sorted({word.strip().lower() for word in words})

whitelist = [
    "gay",
    "lesbian",
    "queer",
    "anus",
    "penis",
    "testicle",
    "vagina",
    "cunnilingus",
    "fellatio",
    r"mothafuckin\\\'"
]

clean_lexicon = [word for word in unique_words if word not in whitelist]

with open(lexicon_filepath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(clean_lexicon) + '\n')

print(f"Lexicon saved to {lexicon_filepath} with {len(clean_lexicon)} words")


# %% [markdown]
# ### Download the dataset
#
# Use the `kagglehub` library to download the dataset.
#
# More information about the dataset can be found [here](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/).

# %%
# Download latest version
path = kagglehub.competition_download(
    'jigsaw-toxic-comment-classification-challenge')

print("Path to competition files:", path)

# %% [markdown]
# ### Load the dataset
#
# Use the `pandas` library to load the dataset.
#
# The dataset consists of 3 files:
# - `train.csv.zip` - training data
# - `test.csv.zip` - test data
# - `test_labels.csv.zip` - test labels
#
# The shapes and samples of each file are printed below.

# %%
train_data_path = os.path.join(path, 'train.csv.zip')
train_data = pd.read_csv(train_data_path)

test_data_path = os.path.join(path, 'test.csv.zip')
test_data = pd.read_csv(test_data_path)

test_labels_path = os.path.join(path, 'test_labels.csv.zip')
test_labels = pd.read_csv(test_labels_path)

print(f"Train data shape: {train_data.shape}\n")
print(f"Sample train data:\n{train_data.sample(10)}")
print("\n", '-'*50, "\n")
print(f"Test data shape: {test_data.shape}\n")
print(f"Sample test data:\n{test_data.sample(10)}")
print("\n", '-'*50, "\n")
print(f"Test labels shape: {test_labels.shape}\n")
print(f"Sample test labels:\n{test_labels.sample(10)}")

# %% [markdown]
# ## Preprocessing
#
# ### Merge toxic classes into one
#
# The dataset consists of multiple toxic classes. But for this project, we are only considering if the comment is toxic or not. Thus, a new column `is_toxic` is created by checking if any of the toxic classes is 1.

# %%
# get the toxic classes from the train data
toxic_classes = train_data.columns.tolist()
toxic_classes.remove('id')
toxic_classes.remove('comment_text')
print(f"Toxic classes: {toxic_classes}\n")

# create a new column to indicate if the comment is toxic
train_df = train_data.copy()
train_df['is_toxic'] = (
    train_df[toxic_classes] == 1
).any(axis=1).astype(int)

# drop all the toxic classes columns
train_df = train_df.drop(columns=toxic_classes)

print(f"Sample train data:\n{train_df.head(10)}\n")
print(train_df['is_toxic'].value_counts(), "\n")
print(
    f"Toxic comments percentage: {train_df['is_toxic'].value_counts(normalize=True)[1]:.2%}"
)

# %% [markdown]
# ### Clean up the test labels
#
# In the test labels, there are comments that aren't labelled. These comments have a value of -1 for all the toxic classes. These comments are removed from the test labels.
#
# After which, the same merging of toxic classes is applied. Then, these changes are being reflected on the test dataset.

# %%
# remove the comments that have no labels
test_labels_clean = test_labels[
    (test_labels[toxic_classes] != -1)
    .all(axis=1)
]

# create a new column to indicate if the comment is toxic
test_labels_df = test_labels_clean.copy()
test_labels_df['is_toxic'] = (
    test_labels_df[toxic_classes] == 1
).any(axis=1).astype(int)

# drop all the toxic classes columns
test_labels_df = test_labels_df.drop(columns=toxic_classes)

print(f"Test labels (filtered) shape: {test_labels_df.shape}\n")
print(f"Sample test labels (filtered):\n{test_labels_df.head(10)}\n")

print("\n", '-'*50, "\n")

# merge the test data with the test labels
test_df = pd.merge(test_data, test_labels_df, on='id')

print(f"Test data (filtered) shape: {test_df.shape}\n")
print(f"Sample test data (filtered):\n{test_df.head(10)}\n")
print(test_df['is_toxic'].value_counts(), "\n")
print(
    f"Toxic comments percentage: {test_df['is_toxic'].value_counts(normalize=True)[1]:.2%}"
)

# %% [markdown]
# ### Initial thoughts
#
# - The dataset is extremely imbalanced, with only 10% of the comments being toxic. This will be a challenge when making a fair evaluation.
# - There is no validation dataset, therefore we will have to split the training dataset into training and validation datasets.

# %%
def check_data_quality(df):
    print(f"Total rows: {len(df)}\n")
    print(f"Missing values:\n{df.isna().sum()}\n")
    print(f"Duplicates: {df.duplicated().sum()}\n")
    print(
        f"Empty comments: {(df['comment_text'].fillna("").str.strip() == '').sum()}\n"
    )
    print(
        f"Sample texts:\n{json.dumps(df['comment_text'].sample(10).tolist(), indent=2, ensure_ascii=False)}"
    )


print('#' * 10, "Quality check for train data:", '#' * 10)
check_data_quality(train_df)

print("\n", '-'*50, "\n")

print('#' * 10, "Quality check for test data:", '#' * 10)
check_data_quality(test_df)

# %% [markdown]
# ### Clean the data

# %%
def clean_data(text):
    if pd.isna(text):
        return ""

    # unescape html entities
    text = html.unescape(text)

    # csv quotes
    text = text.replace('""', '"')

    # replace urls with token
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # replace ips with token
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP]', text)

    # remove wiki headers formatting
    text = re.sub(r'={2,}', '', text)

    # remove wiki indentation formatting
    text = re.sub(r'^\s*:+\s*', '', text, flags=re.MULTILINE)

    # collapse elongated words (from 3 to 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # collapse horizontal spacings
    text = re.sub(r'[ \t]+', ' ', text)

    # collapse vertical spacings
    text = re.sub(r'[\n\r]+', '\n', text)

    # remove starting and trailing quotes and spaces
    text = text.strip(' \n\t\r"\'')

    return text


dfs = [train_df, test_df]
for df in dfs:
    df["comment_text"] = df["comment_text"].apply(clean_data)
    print(
        f"Sample texts:\n{json.dumps(df['comment_text'].sample(10).tolist(), indent=2, ensure_ascii=False)}"
    )

# %% [markdown]
# A new column `has_swear` is also created to determine if the comments contain any swear words from the lexicon.

# %%
with open(lexicon_filepath, 'r', encoding='utf-8') as f:
    swear_words = [line.strip() for line in f if line.strip()]

swear_word_pattern = re.compile(
    r'\b(?:' + '|'.join(map(re.escape, swear_words)) + r')\b',
    flags=re.IGNORECASE
)

for df in [train_df, test_df]:
    df['has_swear'] = df['comment_text'].str.contains(
        swear_word_pattern,
        regex=True
    ).astype(int)

# %% [markdown]
# ### Split the data into train and validation datasets

# %%
train_df_backup = train_df.copy()

train_df, valid_df = train_test_split(
    train_df_backup,
    test_size=0.2,
    shuffle=True,
    random_state=13,
    stratify=train_df_backup["is_toxic"],
)

# %% [markdown]
# ### Checkpoint
#
# Saving the dataframes as parquet files to avoid reprocessing the data.

# %%
# train_df.to_parquet("./data/dataframe/train.parquet")
# valid_df.to_parquet("./data/dataframe/valid.parquet")
# test_df.to_parquet("./data/dataframe/test.parquet")

train_df = pd.read_parquet("./data/dataframe/train.parquet")
valid_df = pd.read_parquet("./data/dataframe/valid.parquet")
test_df = pd.read_parquet("./data/dataframe/test.parquet")

# %% [markdown]
# #### Overview

# %%
for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
    print(f"{'#'*10} {name} Dataset (Total: {len(df):,d}) {'#'*10}\n")

    ct = pd.crosstab(
        df["is_toxic"],
        df["has_swear"],
        rownames=["is_toxic"],
        colnames=["has_swear"],
        margins=True
    )
    print(ct, "\n")

    toxic_with_swear = ct.loc[1, 1] / ct.loc[1, "All"]
    nontoxic_with_swear = ct.loc[0, 1] / ct.loc[0, "All"]

    print(f"Toxic comments containing swears: {toxic_with_swear:.2%}")
    print(f"Non-toxic comments containing swears: {nontoxic_with_swear:.2%}")

    print("\n", '-' * 50, "\n")

# %% [markdown]
# ## Baselines
#
# ### Floor baselines

# %%
# majority class baseline
majority_baseline = DummyClassifier(strategy="most_frequent")
majority_baseline.fit(train_df["comment_text"], train_df["is_toxic"])

# random baseline
random_baseline = DummyClassifier(strategy="stratified", random_state=13)
random_baseline.fit(train_df["comment_text"], train_df["is_toxic"])

# %% [markdown]
# ### Rule-based baseline
#
# The rule-based baseline is based on the assumption that toxic comments contain swear words. Thus, if a comment contains any swear words, it is classified as toxic. This can be achieved by referencing the `has_swear` column which was created previously above.
