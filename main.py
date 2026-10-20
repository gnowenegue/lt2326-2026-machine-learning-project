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
