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
#     display_name: lt2326-2026-machine-learning-project (3.12.16)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Import the dependencies

# %%
import html
import json
import os
import re

import kagglehub
import kagglehub.auth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from IPython.display import display
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

# %% [markdown]
# ### Global flags/configs

# %%
# force save the dataframe after cleaning
FORCE_SAVE_DF = True

# force save the dataframe as csv
FORCE_SAVE_DF_CSV = True

# running locally
LOCAL_RUN = True

# force save the fine-tuned model after training
FORCE_SAVE_MODEL = False

# force save the predictions
FORCE_SAVE_PREDICTIONS = False

RANDOM_SEED = 13

# %% [markdown]
# ### Determine the device to run the model on

# %%
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

is_cuda = torch.cuda.is_available()
is_mps = torch.backends.mps.is_available()

if is_cuda:
    DEVICE = torch.device("cuda")
elif is_mps:
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

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
lexicon_directory = "./data/lexicon"
lexicon_raw_filepath = f"{lexicon_directory}/noswearing_raw.xml"
lexicon_filepath = f"{lexicon_directory}/noswearing.txt"

# extract words from the raw lexicon
with open(lexicon_raw_filepath, "r", encoding="utf-8") as f:
    words = re.findall(r"<title>(.*?)</title>", f.read())

# remove duplicates and sort
unique_words = sorted({word.strip().lower() for word in words})

whitelist = [
    # sexual orientation
    "gay",
    "lesbian",
    "queer",

    # sexual parts
    "anus",
    "penis",
    "testicle",
    "vagina",

    # other
    r"mothafuckin\\\'"
]

# remove whitelisted words
lexicon_filtered = [word for word in unique_words if word not in whitelist]

# save the lexicon
os.makedirs(lexicon_directory, exist_ok=True)

with open(lexicon_filepath, "w", encoding="utf-8") as f:
    f.write("\n".join(lexicon_filtered) + "\n")

print(
    f"Lexicon saved to {lexicon_filepath} with {len(lexicon_filtered)} words"
)


# %% [markdown]
# ### Download the dataset
#
# Try to use the `kagglehub` library to download the dataset. If it fails, use the dataset in local directory.
#
# More information about the dataset can be found [here](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/).

# %%
dataset_directory = "./data/dataset_raw"

try:
    if kagglehub.auth.get_kaggle_credentials() is not None:
        dataset_directory = kagglehub.competition_download(
            "jigsaw-toxic-comment-classification-challenge")
except Exception as e:
    print(e)

print("Path to dataset files:", dataset_directory)

# %% [markdown]
# ### Load the dataset
#
# Load the dataset into dataframes.
#
# The dataset consists of 3 files:
# - `train.csv.zip` - training data
# - `test.csv.zip` - test data
# - `test_labels.csv.zip` - test labels
#
# The shapes and samples of each file are printed below.

# %%
train_data_path = os.path.join(dataset_directory, "train.csv.zip")
train_data = pd.read_csv(train_data_path)

test_data_path = os.path.join(dataset_directory, "test.csv.zip")
test_data = pd.read_csv(test_data_path)

test_labels_path = os.path.join(dataset_directory, "test_labels.csv.zip")
test_labels = pd.read_csv(test_labels_path)

for name, df in [("TRAIN DATA", train_data), ("TEST DATA", test_data), ("TEST LABELS", test_labels)]:
    print(f"{"#" * 10} {name} {"#" * 10}")
    print(f"Shape: {df.shape}\n")
    df.info()
    print(f"\nSample data:\n{df.sample(5)}")
    print("\n" + "-" * 50 + "\n")

# %% [markdown]
# ## Preprocessing
#
# ### Merge the toxic labels into one
#
# The dataset consists of multiple toxic labels. But for this project, we are only considering if the comment is toxic or not. Thus, a new class `is_toxic` is created by checking if any of the toxic labels is 1.

# %%
# get the toxic labels from the train data
toxic_labels = train_data.columns.tolist()
toxic_labels.remove("id")
toxic_labels.remove("comment_text")
print(f"Toxic labels: {toxic_labels}\n")


def create_is_toxic_column(df):
    new_df = df.copy()

    new_df["is_toxic"] = (
        new_df[toxic_labels] == 1
    ).any(axis=1).astype(int)

    return new_df


# create a new column to indicate if the comment is toxic
train_df = create_is_toxic_column(train_data)

# drop all the toxic labels columns
train_df = train_df.drop(columns=toxic_labels)

print(f"Sample train data:\n{train_df.sample(5)}\n")

for is_toxic, count in train_df["is_toxic"].value_counts().items():
    print(
        f"Number of {"toxic" if is_toxic == 1 else "non-toxic"} comments: {count:,d} ({count/len(train_df):.2%})"
    )

# %% [markdown]
# ### Clean up the test labels
#
# In the test labels, there are comments that aren't labelled. These comments have a value of -1 for all the toxic labels. These comments are removed from the test labels.
#
# After which, the same merging of toxic labels is applied. Then, these changes are being reflected on the test dataset.

# %%
# remove the comments that have no labels
test_labels_clean = test_labels[
    (test_labels[toxic_labels] != -1)
    .all(axis=1)
]

# create a new column to indicate if the comment is toxic
test_labels_df = create_is_toxic_column(test_labels_clean)

# drop all the toxic classes columns
test_labels_df = test_labels_df.drop(columns=toxic_labels)

print(f"Test labels (filtered) shape: {test_labels_df.shape}\n")
print(f"Sample test labels (filtered):\n{test_labels_df.sample(5)}")

print("\n", "-"*50, "\n")

# merge the test data with the test labels
test_df = pd.merge(test_data, test_labels_df, on="id")

print(f"Test data (filtered) shape: {test_df.shape}\n")
print(f"Sample test data (filtered):\n{test_df.sample(5)}\n")

for is_toxic, count in test_df["is_toxic"].value_counts().items():
    print(
        f"Number of {"toxic" if is_toxic == 1 else "non-toxic"} comments: {count:,d} ({count/len(test_df):.2%})"
    )

# print(test_df['is_toxic'].value_counts(), "\n")
# print(
    # f"Toxic comments percentage: {test_df['is_toxic'].value_counts(normalize=True)[1]:.2%}"
# )

# %% [markdown]
# ### Initial thoughts
#
# - The dataset is extremely imbalanced, with only 10% of the comments being toxic. This will be a challenge when making a fair evaluation.
# - There is no validation dataset, therefore we will have to split the training dataset into training and validation datasets.

# %% [markdown]
# ### Basic check on data quality

# %%
def check_data_quality(df):
    print(f"Total rows: {len(df)}\n")
    print(f"Missing values:\n{df.isna().sum()}\n")
    print(f"Duplicates: {df.duplicated().sum()}")
    print(
        f"Empty comments: {(df['comment_text'].fillna("").str.strip() == "").sum()}\n"
    )
    print(
        f"Sample texts:\n{json.dumps(df['comment_text'].sample(5).tolist(), indent=2, ensure_ascii=False)}"
    )


for name, df in [("train", train_df), ("test", test_df)]:
    print("#" * 10, f"Quality check for {name} data:", "#" * 10)
    check_data_quality(df)
    print("\n", "-"*50, "\n")


# %% [markdown]
# #### Check for elongated words

# %%
def has_repeated_chars(text):
    return bool(re.search(r"(.)\1{2,}", text))


for name, df in [("TRAIN DATA", train_df), ("TEST DATA", test_df)]:
    repeated_count = df["comment_text"].apply(has_repeated_chars).sum()
    print(
        f"{name} comments with repeated characters (>=3): {repeated_count:,d} ({repeated_count / len(df):.2%})"
    )

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
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

    # replace ips with token
    text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "[IP]", text)

    # remove wiki headers formatting
    text = re.sub(r"={2,}", "", text)

    # remove wiki indentation formatting
    text = re.sub(r"^\s*:+\s*", "", text, flags=re.MULTILINE)

    # collapse elongated words (from 3 to 2)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # collapse horizontal spacings
    text = re.sub(r"[ \t]+", " ", text)

    # collapse vertical spacings
    text = re.sub(r"[\n\r]+", "\n", text)

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

with pd.option_context("display.max_colwidth", None):
    display(demo_df)

for name, df in [("Train data", train_df), ("Test data", test_df)]:
    df["comment_text"] = df["comment_text"].apply(clean_data)
    print(
        f"{name} sample texts:\n{json.dumps(df['comment_text'].sample(5).tolist(), indent=2, ensure_ascii=False)}"
    )
    print("\n", "-"*50, "\n")

# %% [markdown]
# A new class `has_swear` is also created to determine if the comments contain any swear words from the lexicon.

# %%
with open(lexicon_filepath, "r", encoding="utf-8") as f:
    swear_words = [line.strip() for line in f if line.strip()]

swear_words = "|".join(map(re.escape, swear_words))
swear_word_pattern = re.compile(rf'\b(?:{swear_words})\b', flags=re.IGNORECASE)

for name, df in [("Train data", train_df), ("Test data", test_df)]:
    df["has_swear"] = df["comment_text"].str.contains(
        swear_word_pattern,
        regex=True
    ).astype(int)

    count_swear = df["has_swear"].sum()
    percentage_swear = count_swear/len(df)
    print(
        f"{name} - Comments with swear words: {count_swear:,d} out of {len(df):,d} ({percentage_swear:.2%})"
    )

# %% [markdown]
# ### Split the training data into train and validation datasets

# %%
train_df_backup = train_df.copy()

train_df, valid_df = train_test_split(
    train_df_backup,
    test_size=0.2,
    shuffle=True,
    random_state=RANDOM_SEED,
    stratify=train_df_backup[["is_toxic", "has_swear"]],
)

# %% [markdown]
# ### Split the test dataset into 4 slices

# %%
# create word count column
test_df["word_count"] = test_df["comment_text"].apply(lambda x: len(x.split()))

# remap binary values to labels
is_toxic = test_df["is_toxic"].map({1: "toxic", 0: "non_toxic"})
has_swear = test_df["has_swear"].map({1: "with_swear", 0: "without_swear"})

# create slice class
test_df["slice"] = is_toxic + "_" + has_swear

for name, count in test_df["slice"].value_counts().items():
    print(f"{name}: {count:,d} ({count / len(test_df):.2%})")

# %%
os.makedirs("./data/dataset", exist_ok=True)
csv_files_exists = all(
    os.path.exists(f"./data/dataset/{s}_clean.csv")
    for s in ["train", "valid", "test"]
)

if not csv_files_exists or FORCE_SAVE_DF_CSV:
    train_df.to_csv("./data/dataset/train_clean.csv", index=False)
    valid_df.to_csv("./data/dataset/valid_clean.csv", index=False)
    test_df.to_csv("./data/dataset/test_clean.csv", index=False)
    print("Dataframes saved as csv")
else:
    print("Dataframes already saved as csv")

# %% [markdown]
# ### Checkpoint for the dataframes
#
# Saving the dataframes as parquet files to avoid reprocessing the data.

# %%
os.makedirs("./data/dataframe", exist_ok=True)
df_files_exists = all(
    os.path.exists(f"./data/dataframe/{s}.parquet")
    for s in ["train", "valid", "test"]
)

if not df_files_exists or FORCE_SAVE_DF:
    train_df.to_parquet("./data/dataframe/train.parquet")
    valid_df.to_parquet("./data/dataframe/valid.parquet")
    test_df.to_parquet("./data/dataframe/test.parquet")
    print("Dataframes saved")
else:
    train_df = pd.read_parquet("./data/dataframe/train.parquet")
    valid_df = pd.read_parquet("./data/dataframe/valid.parquet")
    test_df = pd.read_parquet("./data/dataframe/test.parquet")
    print("Dataframes loaded")

# %% [markdown]
# ### Overview
#
# Big picture metrics for all 3 datasets.

# %%
for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
    print(f"{"#"*10} {name} Dataset (Total: {len(df):,d}) {"#"*10}\n")

    ct = pd.crosstab(
        df["is_toxic"],
        df["has_swear"],
        margins=True
    )

    ct.index = ["Non-toxic", "Toxic", "All"]
    ct.columns = ["No swear", "With swear", "All"]

    def format_ct(count):
        percentage = count/len(df)
        return f"{count:,d} ({percentage:.2%})"

    print(ct.map(format_ct), "\n")

    toxic_with_swear = ct.loc["Toxic", "With swear"] / ct.loc["Toxic", "All"]
    nontoxic_with_swear = ct.loc["Non-toxic", "With swear"] \
        / ct.loc["Non-toxic", "All"]

    print(f"{toxic_with_swear:.2%} of toxic comments contain swear words")
    print(f"{nontoxic_with_swear:.2%} of non-toxic comments contain swear words")

    print("\n", "-" * 50, "\n")

# %% [markdown]
# ## Baselines
#
# ### Floor baselines

# %%
# majority class baseline
majority_baseline = DummyClassifier(strategy="most_frequent")
majority_baseline.fit(train_df["comment_text"], train_df["is_toxic"])

# random baseline
random_baseline = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
_ = random_baseline.fit(train_df["comment_text"], train_df["is_toxic"])

# %% [markdown]
# ### Rule-based baseline
#
# The rule-based baseline is based on the assumption that toxic comments contain swear words. Thus, if a comment contains any swear words, it is classified as toxic. This can be achieved by referencing the `has_swear` column which was created previously above.

# %% [markdown]
# ### Traditional ML baseline (TF-IDF + Logistic Regression)

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
        random_state=RANDOM_SEED))
])

_ = tfidf_lr_baseline.fit(train_df["comment_text"], train_df["is_toxic"])


# %% [markdown]
# ### Evaluation helper method

# %%
def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    # overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(
        y_true,
        y_proba
    ) if y_proba is not None else None

    def calculation_slice_metrics(slice_name, calculation_method):
        slice = test_df["slice"] == slice_name
        slice_pred = y_pred[slice]
        slice_true = y_true[slice]
        return calculation_method(
            slice_true,
            slice_pred,
            zero_division=0
        )

    def false_positive_rate(y_true, y_pred, zero_division=0):
        false_positive = ((y_true == 0) & (y_pred == 1)).sum()
        true_negative = ((y_true == 0) & (y_pred == 0)).sum()
        total_negatives = false_positive + true_negative

        if total_negatives == 0:
            return zero_division

        return false_positive / total_negatives

    # recall for toxic slices
    recall_toxic_with_swear = calculation_slice_metrics(
        "toxic_with_swear",
        recall_score
    )

    recall_toxic_without_swear = calculation_slice_metrics(
        "toxic_without_swear",
        recall_score
    )

    delta_recall = recall_toxic_with_swear - recall_toxic_without_swear

    # fpr for non-toxic slices
    fpr_non_toxic_with_swear = calculation_slice_metrics(
        "non_toxic_with_swear",
        false_positive_rate
    )
    fpr_non_toxic_without_swear = calculation_slice_metrics(
        "non_toxic_without_swear",
        false_positive_rate
    )

    delta_fpr = fpr_non_toxic_with_swear - fpr_non_toxic_without_swear

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "PR-AUC": round(pr_auc, 4) if pr_auc is not None else "N/A",
        "Recall (toxic with swear)": round(recall_toxic_with_swear, 4),
        "Recall (toxic without swear)": round(recall_toxic_without_swear, 4),
        "Delta Recall": round(delta_recall, 4),
        "FPR (non-toxic with swear)": round(fpr_non_toxic_with_swear, 4),
        "FPR (non-toxic without swear)": round(fpr_non_toxic_without_swear, 4),
        "Delta FPR": round(delta_fpr, 4)
    }


# %% [markdown]
# ### Run evaluation on baselines

# %%
y_true = test_df["is_toxic"]
test_texts = test_df["comment_text"]

# majority baseline
y_pred_majority = majority_baseline.predict(test_texts)
y_proba_majority = majority_baseline.predict_proba(test_texts)[:, 1]
result_majority = evaluate_model(
    y_true, y_pred_majority, y_proba_majority, "Majority Baseline")

# random baseline
y_pred_random = random_baseline.predict(test_texts)
y_proba_random = random_baseline.predict_proba(test_texts)[:, 1]
result_random = evaluate_model(
    y_true, y_pred_random, y_proba_random, "Random Baseline")

# rule-based baseline
y_pred_rule = test_df["has_swear"]
result_rule = evaluate_model(y_true, y_pred_rule, None, "Rule-Based Baseline")

# TF-IDF + Logistic Regression
y_pred_tfidf_lr = tfidf_lr_baseline.predict(test_texts)
y_proba_tfidf_lr = tfidf_lr_baseline.predict_proba(test_texts)[:, 1]
result_tfidf_lr = evaluate_model(
    y_true, y_pred_tfidf_lr, y_proba_tfidf_lr, "TF-IDF + Logistic Regression")

# %% [markdown]
# ## Pre-trained model
#
# ### Define the model

# %%
MODEL_NAME = "microsoft/deberta-v3-small" if LOCAL_RUN else "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sample_text = "This is a great project, not toxic at all!"
tokens = tokenizer(sample_text)
decoded_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"])

token_preview = pd.DataFrame({
    "Token": decoded_tokens,
    "Token ID": tokens["input_ids"]
})

display(token_preview)

# %% [markdown]
# ### Find the token size

# %%
sample_comments = train_df["comment_text"].sample(
    10000,
    random_state=RANDOM_SEED
).tolist()
token_output = tokenizer(sample_comments, truncation=False)

# count the number of tokens in each comment
token_length = pd.Series([len(tokens) for tokens in token_output["input_ids"]])

# calculate percentiles and the percentage that fit under 128
print("Percentiles:")
print(token_length.quantile([0.50, 0.75, 0.82, 0.95, 1.0]))
percentage_under_128 = (token_length <= 128).mean()
print(f"\nExact percentage <= 128 tokens: {percentage_under_128:.2%}")

# %% [markdown]
# ### Tokenize the dataset

# %%
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)


def tokenize(batch):
    tokens = tokenizer(batch["comment_text"], truncation=True, max_length=128)
    tokens["labels"] = batch["is_toxic"]
    return tokens


train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["id"])
valid_dataset = valid_dataset.remove_columns(["id"])
test_dataset = test_dataset.remove_columns(["id"])

# %%
if LOCAL_RUN:
    train_dataset = train_dataset.shuffle(seed=RANDOM_SEED).select(range(3000))
    valid_dataset = valid_dataset.shuffle(seed=RANDOM_SEED).select(range(800))
    test_dataset = test_dataset.shuffle(seed=RANDOM_SEED).select(range(1000))

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.float32
)
model.to(DEVICE)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "precision": round(precision_score(labels, predictions, zero_division=0), 4),
        "recall": round(recall_score(labels, predictions, zero_division=0), 4),
        "f1": round(f1_score(labels, predictions, zero_division=0), 4),
    }


batch_size = 16 if is_cuda else 8
target_batch_size = 32

training_args = TrainingArguments(
    output_dir="./artifact/model",
    num_train_epochs=1 if LOCAL_RUN else 3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    gradient_accumulation_steps=target_batch_size // batch_size,
    bf16=is_cuda,
    learning_rate=2e-5,
    logging_steps=50,
    disable_tqdm=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2
    # label_names=["is_toxic"],
    # remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

model_directory = "./artifact/model/best"
os.makedirs(model_directory, exist_ok=True)

if FORCE_SAVE_MODEL:
    trainer.train()
    trainer.save_model(model_directory)
    print(f"Model saved to {model_directory}")

else:
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    model.to(DEVICE)
    model.eval()
    trainer.model = model
    print(f"Model loaded from {model_directory}")

# %% [markdown]
# ### Run the prediction

# %%
os.makedirs("./artifact/logits", exist_ok=True)

logits_path = "./artifact/logits/deberta_test_logits.npy"

if FORCE_SAVE_PREDICTIONS:
    test_pred = trainer.predict(test_dataset)
    logits = test_pred.predictions

    np.save(logits_path, logits)
    print(f"saved predictions to {logits_path}")
else:
    print("loading saved predictions")
    logits = np.load(logits_path)


y_pred_deberta = logits.argmax(axis=-1)
y_proba_deberta = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

result_deberta = evaluate_model(
    y_true,
    y_pred_deberta,
    y_proba_deberta,
    model_name="DeBERTa"
)

# %% [markdown]
# ## Evaluation results

# %%
test_results = pd.DataFrame([
    result_majority, result_random, result_rule, result_tfidf_lr, result_deberta
])

display(test_results)

# %%
overall_columns = [
    "Model", "Accuracy", "Precision", "Recall", "F1", "PR-AUC"
]

print(
    test_results[overall_columns].to_latex(
        index=False,
        float_format="%.4f",
        caption="Overall test set performance across baselines and DeBERTa.",
        label="tab:overall_results"
    )
)

# %%
slice_cols = [
    "Model",
    "Recall (toxic with swear)",
    "Recall (toxic without swear)",
    "Delta Recall",
    "FPR (non-toxic with swear)",
    "FPR (non-toxic without swear)",
    "Delta FPR"
]
print(
    test_results[slice_cols].to_latex(
        index=False,
        float_format="%.4f",
        caption="Slice-based recall and false positive rates.",
        label="tab:slice_results"
    )
)

# %% [markdown]
# ### Precision-Recall Curve

# %%
# calculate precision and recall points for DeBERTa
deberta_precision, deberta_recall, _ = precision_recall_curve(
    y_true, y_proba_deberta
)

# calculate precision and recall points for TF-IDF + Logistic Regression
tfidf_lr_precision, tfidf_lr_recall, _ = precision_recall_curve(
    y_true, y_proba_tfidf_lr
)

# create the plot canvas
plt.figure(figsize=(7, 5))

# plot the DeBERTa curve
plt.plot(
    deberta_recall,
    deberta_precision,
    label=f"DeBERTa (PR-AUC = {result_deberta['PR-AUC']:.4f})",
    color="royalblue",
    linewidth=2
)

# plot the TF-IDF + Logistic Regression curve
plt.plot(
    tfidf_lr_recall,
    tfidf_lr_precision,
    label=f"TF-IDF + LR (PR-AUC = {result_tfidf_lr['PR-AUC']:.4f})",
    color="darkorange",
    linewidth=2
)

# plot the baseline reference (the overall percentage of toxic comments)
toxic_prevalence = y_true.mean()
plt.axhline(
    y=toxic_prevalence,
    color="gray",
    linestyle="--",
    label=f"No-skill baseline ({toxic_prevalence:.2%})"
)

# labels, titles, and legend
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve on Test Set")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower left")

os.makedirs("./artifact/figures", exist_ok=True)
plt.savefig("./artifact/figures/pr_auc_curve.pdf")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Peek into TFIDF + Logistic Regression

# %%
tfidf = tfidf_lr_baseline.named_steps["tfidf"]
logistic_regression = tfidf_lr_baseline.named_steps["clf"]

# get features and their coefficients
feature_names = tfidf.get_feature_names_out()
coefficients = logistic_regression.coef_[0]

features_coefficients_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

top_toxic_words = features_coefficients_df.sort_values(
    by="coefficient",
    ascending=False
).head(10)
print("Top 10 words predicting TOXIC:")
display(top_toxic_words)

top_non_toxic_words = features_coefficients_df.sort_values(
    by="coefficient",
    ascending=True
).head(10)
print("Top 10 words predicting NON-TOXIC:")
display(top_non_toxic_words)

def print_samples(df, n=3):
    for _, row in df.sample(n).iterrows():
        print(f"\n[Comment]:\n{row['comment_text'][:200]}...")
        print(
            f"[TF-IDF+LR]: {row['pred_tfidf_lr']} (Prob: {row['prob_tfidf_lr']})"
        )
        print(
            f"[DeBERTa]: {row['pred_deberta']} (Prob: {row['prob_deberta']})"
        )


non_toxic_with_swear_slice = analysis_df[
    analysis_df["slice"] == "non_toxic_with_swear"
]
tfidf_lr_incorrect = (
    non_toxic_with_swear_slice["pred_tfidf_lr"] == 1
).sum()
deberta_incorrect = (non_toxic_with_swear_slice["pred_deberta"] == 1).sum()

print(
    f"Total non-toxic comments with swear words: {len(non_toxic_with_swear_slice)}"
)
print(f"\tFalse alarms by TF-IDF+LR: {tfidf_lr_incorrect}")
print(f"\tFalse alarms by DeBERTa: {deberta_incorrect}")

print('\n', '-'*100, '\n')

deberta_correct_tfidf_lr_incorrect = non_toxic_with_swear_slice[
    (non_toxic_with_swear_slice["pred_tfidf_lr"] == 1) &
    (non_toxic_with_swear_slice["pred_deberta"] == 0)
]

print(
    f"TF-IDF wrong, DeBERTa correct: {len(deberta_correct_tfidf_lr_incorrect)}"
)
print_samples(deberta_correct_tfidf_lr_incorrect)

print('\n', '-'*100, '\n')

deberta_incorrect_tfidf_lr_incorrect = non_toxic_with_swear_slice[
    (non_toxic_with_swear_slice["pred_tfidf_lr"] == 1) &
    (non_toxic_with_swear_slice["pred_deberta"] == 1)
]

print(
    f"Both TF-IDF and DeBERTa wrong: {len(deberta_incorrect_tfidf_lr_incorrect)}"
)
print_samples(deberta_incorrect_tfidf_lr_incorrect)

# %%
toxic_without_swear_slice = analysis_df[
    analysis_df["slice"] == "toxic_without_swear"
]
tfidf_lr_misses = (toxic_without_swear_slice["pred_tfidf_lr"] == 0).sum()
deberta_misses = (toxic_without_swear_slice["pred_deberta"] == 0).sum()

print(
    f"Total toxic comments without swear words: {len(toxic_without_swear_slice)}"
)
print(f"\tMissed by TF-IDF+LR: {tfidf_lr_misses}")
print(f"\tMissed by DeBERTa: {deberta_misses}")

print('\n', '-'*100, '\n')

deberta_catches_tfidf_lr_misses = toxic_without_swear_slice[
    (toxic_without_swear_slice["pred_tfidf_lr"] == 0) &
    (toxic_without_swear_slice["pred_deberta"] == 1)
]

print(
    f"TF-IDF misses, DeBERTa catches: {len(deberta_catches_tfidf_lr_misses)}"
)
print_samples(deberta_catches_tfidf_lr_misses)

print('\n', '-'*100, '\n')

deberta_misses_tfidf_lr_misses = toxic_without_swear_slice[
    (toxic_without_swear_slice["pred_tfidf_lr"] == 0) &
    (toxic_without_swear_slice["pred_deberta"] == 0)
]

print(
    f"Both TF-IDF and DeBERTa misses: {len(deberta_misses_tfidf_lr_misses)}"
)
print_samples(deberta_misses_tfidf_lr_misses)
