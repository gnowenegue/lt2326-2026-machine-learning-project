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

# %%
import torch

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
import os
import re

words = re.findall(
    r'<title>(.*?)</title>',
    open('./data/lexicon/noswearing_raw.xml').read()
)

whitelist = [
    "gay",
    "lesbian",
    "queer",
    "anus",
    "penis",
    "testicle",
    "vagina",
    "cunnilingus",
    "fellatio"
]

if not os.path.exists('./data/lexicon/noswearing.txt'):
    with open('./data/lexicon/noswearing.txt', 'w') as f:
        for word in words:
            if word not in whitelist:
                f.write(word + '\n')
    print("Lexicon processed and saved.")
else:
    print("Lexicon already exists.")


# %% [markdown]
# ### Download the dataset
#
# Use the `kagglehub` library to download the dataset.
#
# More information about the dataset can be found [here](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/).

# %%
import kagglehub

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
import os

import pandas as pd

train_data_path = os.path.join(path, 'train.csv.zip')
train_data = pd.read_csv(train_data_path)

test_data_path = os.path.join(path, 'test.csv.zip')
test_data = pd.read_csv(test_data_path)

test_labels_path = os.path.join(path, 'test_labels.csv.zip')
test_labels = pd.read_csv(test_labels_path)

print(f"Train data shape: {train_data.shape}\n")
print(f"Sample train data:\n{train_data.head(10)}")
print("\n", '-'*50, "\n")
print(f"Test data shape: {test_data.shape}\n")
print(f"Sample test data:\n{test_data.head(10)}")
print("\n", '-'*50, "\n")
print(f"Test labels shape: {test_labels.shape}\n")
print(f"Sample test labels:\n{test_labels.head(10)}")

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
