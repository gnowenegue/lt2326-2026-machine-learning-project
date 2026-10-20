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

# %%
import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:1')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f"Using device: {DEVICE}")

# %%
import kagglehub

# Download latest version
path = kagglehub.competition_download(
    'jigsaw-toxic-comment-classification-challenge')

print("Path to competition files:", path)

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

test_df = pd.merge(test_data, test_labels_df, on='id')

print(f"Test data (filtered) shape: {test_df.shape}\n")
print(f"Sample test data (filtered):\n{test_df.head(10)}\n")
print(test_df['is_toxic'].value_counts(), "\n")
print(
    f"Toxic comments percentage: {test_df['is_toxic'].value_counts(normalize=True)[1]:.2%}"
)
