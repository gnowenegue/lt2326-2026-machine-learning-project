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
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer,
                          TrainingArguments)

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

for name, df in [("TRAIN DATA", train_data), ("TEST DATA", test_data), ("TEST LABELS", test_labels)]:
    print(f"{'#' * 10} {name} {'#' * 10}")
    print(f"Shape: {df.shape}\n")
    df.info()
    print(f"\nSample data:\n{df.sample(5)}")
    print("\n" + "-" * 50 + "\n")

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

print(f"Sample train data:\n{train_df.sample(5)}\n")
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
print(f"Sample test labels (filtered):\n{test_labels_df.sample(5)}\n")

print("\n", '-'*50, "\n")

# merge the test data with the test labels
test_df = pd.merge(test_data, test_labels_df, on='id')

print(f"Test data (filtered) shape: {test_df.shape}\n")
print(f"Sample test data (filtered):\n{test_df.sample(5)}\n")
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
        f"Empty comments: {(df['comment_text'].fillna('').str.strip() == '').sum()}\n"
    )
    print(
        f"Sample texts:\n{json.dumps(df['comment_text'].sample(5).tolist(), indent=2, ensure_ascii=False)}"
    )


for name, df in [("train", train_df), ("test", test_df)]:
    print('#' * 10, f"Quality check for {name} data:", '#' * 10)
    check_data_quality(df)
    print("\n", '-'*50, "\n")

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

    # remove whitespace
    text = text.strip()

    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    return text


demo_cases = [
    {
        "Rule Tested": "Wiki headers, indentation, URL, IP, HTML entity, word elongation",
        "Before": "== Admin Notice ==\n::: Visit https://en.wikipedia.org or contact 192.168.1.1 immediately! That was sooooo cool &amp; awesome.",
    },
    {
        "Rule Tested": "CSV escaped quotes, www URL, character elongation, trailing spaces",
        "Before": '""Hey you,"" check www.example.com/rules right nowwwwww!!!!!   ',
    }
]

demo_df = pd.DataFrame([
    {
        "Rule Tested": case["Rule Tested"],
        "Before": case["Before"],
        "After": clean_data(case["Before"])
    }
    for case in demo_cases
])

with pd.option_context('display.max_colwidth', None):
    display(demo_df)

dfs = [train_df, test_df]
for df in dfs:
    df["comment_text"] = df["comment_text"].apply(clean_data)
    print(
        f"Sample texts:\n{json.dumps(df['comment_text'].sample(5).tolist(), indent=2, ensure_ascii=False)}"
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

for name, df in [("Train data", train_df), ("Test data", test_df)]:
    df['has_swear'] = df['comment_text'].str.contains(
        swear_word_pattern,
        regex=True
    ).astype(int)
    count_swear = df['has_swear'].sum()
    percentage_swear = count_swear/len(df)
    print(
        f"{name} - Comments with swear words: {count_swear} out of {len(df)} ({percentage_swear:.2%})"
    )

# %% [markdown]
# ### Split the data into train and validation datasets

# %%
train_df_backup = train_df.copy()

train_df, valid_df = train_test_split(
    train_df_backup,
    test_size=0.2,
    shuffle=True,
    random_state=13,
    stratify=train_df_backup[["is_toxic", "has_swear"]],
)

# %% [markdown]
# ### Checkpoint for the dataframes
#
# Saving the dataframes as parquet files to avoid reprocessing the data.

# %%
FORCE_SAVE = False

os.makedirs("./data/dataframe", exist_ok=True)
files_exists = all(
    os.path.exists(f"./data/dataframe/{s}.parquet")
    for s in ["train", "valid", "test"]
)

if not files_exists or FORCE_SAVE:
    train_df.to_parquet("./data/dataframe/train.parquet")
    valid_df.to_parquet("./data/dataframe/valid.parquet")
    test_df.to_parquet("./data/dataframe/test.parquet")
else:
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
        margins=True
    )

    ct.index = ["Non-toxic", "Toxic", "All"]
    ct.columns = ["No swear", "With swear", "All"]

    print(ct, "\n")

    toxic_with_swear = ct.loc["Toxic", "With swear"] / ct.loc["Toxic", "All"]
    nontoxic_with_swear = ct.loc["Non-toxic", "With swear"] \
        / ct.loc["Non-toxic", "All"]

    print(f"Toxic comments with swear word: {toxic_with_swear:.2%}")
    print(f"Non-toxic comments with swear word: {nontoxic_with_swear:.2%}")

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

# %% [markdown]
# #### Traditional ML baseline (TF-IDF + Logistic Regression)

# %%
tfidf_lr_baseline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        min_df=3,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=13))
])

tfidf_lr_baseline.fit(train_df["comment_text"], train_df["is_toxic"])


# %%
def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    pr_auc = average_precision_score(
        y_true, y_proba) if y_proba is not None else None

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "PR-AUC": round(pr_auc, 4) if pr_auc is not None else "N/A",
    }


# %%
y_true = valid_df["is_toxic"]
valid_texts = valid_df["comment_text"]

y_pred_majority = majority_baseline.predict(valid_texts)
y_proba_majority = majority_baseline.predict_proba(valid_texts)[:, 1]
result_majority = evaluate_model(
    y_true, y_pred_majority, y_proba_majority, "Majority Baseline")

y_pred_random = random_baseline.predict(valid_texts)
y_proba_random = random_baseline.predict_proba(valid_texts)[:, 1]
result_random = evaluate_model(
    y_true, y_pred_random, y_proba_random, "Random Baseline")

y_pred_rule = valid_df["has_swear"]
result_rule = evaluate_model(y_true, y_pred_rule, None, "Rule-Based Baseline")

y_pred_tfidf_lr = tfidf_lr_baseline.predict(valid_texts)
y_proba_tfidf_lr = tfidf_lr_baseline.predict_proba(valid_texts)[:, 1]
result_tfidf_lr = evaluate_model(
    y_true, y_pred_tfidf_lr, y_proba_tfidf_lr, "TF-IDF + Logistic Regression")

pd.DataFrame([result_majority, result_random, result_rule, result_tfidf_lr])

# %% [markdown]
# ## Pre-trained model

# %%
MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sample_text = "This is a great project, not toxic at all!"
tokens = tokenizer(sample_text)

print(tokens)

# %%
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
# test_dataset = Dataset.from_pandas(test_df)


def tokenize(batch):
    tokens = tokenizer(batch['comment_text'], truncation=True, max_length=128)
    tokens["labels"] = batch["is_toxic"]
    return tokens


train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)
# test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["id"])
valid_dataset = valid_dataset.remove_columns(["id"])

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(DEVICE)

training_args = TrainingArguments(
    output_dir="./artifact/model",
    # num_train_epochs=3,
    num_train_epochs=1,
    # per_device_train_batch_size=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"every_n_layers": 4},
    # bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # label_names=["is_toxic"],
    # remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    # train_dataset=train_dataset,
    train_dataset=train_dataset.shuffle(seed=13).select(range(1000)),
    # eval_dataset=valid_dataset,
    eval_dataset=valid_dataset.shuffle(seed=13).select(range(300)),
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train()

# %%
