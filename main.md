## Setup

### Imports


```python
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
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
```

### Determine the device


```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

is_cuda = torch.cuda.is_available()
is_mps = torch.backends.mps.is_available()

if is_cuda:
    DEVICE = torch.device('cuda')
elif is_mps:
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f"Using device: {DEVICE}")
```

    Using device: cuda


### Download the lexicon

The swear word lexicon is downloaded from [noswearing.com](https://www.noswearing.com/fulldict.xml).

```shell
curl -s https://www.noswearing.com/fulldict.xml -o data/lexicon/noswearing_raw.xml
```

### Process and clean the lexicon

The lexicon downloaded is in XML format. The words are extracted and saved in a separate text file.

Note that the lexicon contains some words that aren't swear words. These words are manually removed from the final lexicon.


```python
lexicon_directory = './data/lexicon'
lexicon_raw_filepath = f"{lexicon_directory}/noswearing_raw.xml"
lexicon_filepath = f"{lexicon_directory}/noswearing.txt"

# extract words from the raw lexicon
with open(lexicon_raw_filepath, 'r', encoding='utf-8') as f:
    words = re.findall(r'<title>(.*?)</title>', f.read())

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

with open(lexicon_filepath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lexicon_filtered) + '\n')

print(
    f"Lexicon saved to {lexicon_filepath} with {len(lexicon_filtered)} words"
)
```

    Lexicon saved to ./data/lexicon/noswearing.txt with 341 words


### Download the dataset

Use the `kagglehub` library to download the dataset.

More information about the dataset can be found [here](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/).


```python
dataset_directory = "./data/dataset_raw"

try:
    if kagglehub.auth.get_kaggle_credentials() is not None:
        dataset_directory = kagglehub.competition_download(
            'jigsaw-toxic-comment-classification-challenge')
except Exception as e:
    print(e)

print("Path to dataset files:", dataset_directory)
```

    Path to dataset files: ./data/dataset_raw


### Load the dataset

Use the `pandas` library to load the dataset.

The dataset consists of 3 files:
- `train.csv.zip` - training data
- `test.csv.zip` - test data
- `test_labels.csv.zip` - test labels

The shapes and samples of each file are printed below.


```python
train_data_path = os.path.join(dataset_directory, 'train.csv.zip')
train_data = pd.read_csv(train_data_path)

test_data_path = os.path.join(dataset_directory, 'test.csv.zip')
test_data = pd.read_csv(test_data_path)

test_labels_path = os.path.join(dataset_directory, 'test_labels.csv.zip')
test_labels = pd.read_csv(test_labels_path)

for name, df in [("TRAIN DATA", train_data), ("TEST DATA", test_data), ("TEST LABELS", test_labels)]:
    print(f"{'#' * 10} {name} {'#' * 10}")
    print(f"Shape: {df.shape}\n")
    df.info()
    print(f"\nSample data:\n{df.sample(5)}")
    print("\n" + "-" * 50 + "\n")
```

    ########## TRAIN DATA ##########
    Shape: (159571, 8)
    
    <class 'pandas.DataFrame'>
    RangeIndex: 159571 entries, 0 to 159570
    Data columns (total 8 columns):
     #   Column         Non-Null Count   Dtype
    ---  ------         --------------   -----
     0   id             159571 non-null  str  
     1   comment_text   159571 non-null  str  
     2   toxic          159571 non-null  int64
     3   severe_toxic   159571 non-null  int64
     4   obscene        159571 non-null  int64
     5   threat         159571 non-null  int64
     6   insult         159571 non-null  int64
     7   identity_hate  159571 non-null  int64
    dtypes: int64(6), str(2)
    memory usage: 72.2 MB
    
    Sample data:
                          id                                       comment_text  \
    147016  357dd88ffcbf9278  "\nFurther, you are however, allowed to cite t...   
    148973  54e280d05b2710d7  , whom you have blocked, is requesting to be u...   
    63360   a986410358edb01b  Sock Puppet\nHello. I have filed this in conne...   
    87651   ea7008c7181193f9  "As Drew often points out on the show....""you...   
    115734  6aeec383131e30ce      Quit bending over before the Nazi Christians.   
    
            toxic  severe_toxic  obscene  threat  insult  identity_hate  
    147016      0             0        0       0       0              0  
    148973      0             0        0       0       0              0  
    63360       0             0        0       0       0              0  
    87651       0             0        0       0       0              0  
    115734      0             0        0       0       0              0  
    
    --------------------------------------------------
    
    ########## TEST DATA ##########
    Shape: (153164, 2)
    
    <class 'pandas.DataFrame'>
    RangeIndex: 153164 entries, 0 to 153163
    Data columns (total 2 columns):
     #   Column        Non-Null Count   Dtype
    ---  ------        --------------   -----
     0   id            153164 non-null  str  
     1   comment_text  153164 non-null  str  
    dtypes: str(2)
    memory usage: 58.6 MB
    
    Sample data:
                          id                                       comment_text
    101448  a94eb70410b41127  ::::The compromise is to include both template...
    61832   66e2d070c983cfd1  ::I've removed the links as they easily fail W...
    130989  daf0423276a14ad4  ::Thank you. Just for an outside perspective, ...
    92052   99745071f9300e5c  == List of ships captured by Somali pirates ==...
    103565  ace44ee3baffa89a  Someone's fixed your captialization. Troll pos...
    
    --------------------------------------------------
    
    ########## TEST LABELS ##########
    Shape: (153164, 7)
    
    <class 'pandas.DataFrame'>
    RangeIndex: 153164 entries, 0 to 153163
    Data columns (total 7 columns):
     #   Column         Non-Null Count   Dtype
    ---  ------         --------------   -----
     0   id             153164 non-null  str  
     1   toxic          153164 non-null  int64
     2   severe_toxic   153164 non-null  int64
     3   obscene        153164 non-null  int64
     4   threat         153164 non-null  int64
     5   insult         153164 non-null  int64
     6   identity_hate  153164 non-null  int64
    dtypes: int64(6), str(1)
    memory usage: 10.5 MB
    
    Sample data:
                          id  toxic  severe_toxic  obscene  threat  insult  \
    100487  a7aed5a086f51263     -1            -1       -1      -1      -1   
    114503  bf172b0314f9cd11      0             0        0       0       0   
    106708  b21145e7025ea830     -1            -1       -1      -1      -1   
    145759  f3aae4c6c6a31bab     -1            -1       -1      -1      -1   
    3358    05c51423ecf689c6     -1            -1       -1      -1      -1   
    
            identity_hate  
    100487             -1  
    114503              0  
    106708             -1  
    145759             -1  
    3358               -1  
    
    --------------------------------------------------
    


## Preprocessing

### Merge toxic classes into one

The dataset consists of multiple toxic classes. But for this project, we are only considering if the comment is toxic or not. Thus, a new column `is_toxic` is created by checking if any of the toxic classes is 1.


```python
# get the toxic classes from the train data
toxic_classes = train_data.columns.tolist()
toxic_classes.remove('id')
toxic_classes.remove('comment_text')
print(f"Toxic classes: {toxic_classes}\n")


def create_is_toxic_column(df):
    new_df = df.copy()

    new_df['is_toxic'] = (
        new_df[toxic_classes] == 1
    ).any(axis=1).astype(int)

    return new_df


# create a new column to indicate if the comment is toxic
train_df = create_is_toxic_column(train_data)

# drop all the toxic classes columns
train_df = train_df.drop(columns=toxic_classes)

print(f"Sample train data:\n{train_df.sample(5)}\n")
print(train_df['is_toxic'].value_counts(), "\n")
print(
    f"Toxic comments percentage: {train_df['is_toxic'].value_counts(normalize=True)[1]:.2%}"
)
```

    Toxic classes: ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    Sample train data:
                          id                                       comment_text  \
    93621   fa574f7daf2ff8ad  "\n\n You refused My Block Application On The ...   
    67803   b567a2568c3879e1  "\n\n Fan club on MySpace \n\nI have a fanclub...   
    16504   2b80c89d871e68de  "\n\nThanks George! Signing off just now, but ...   
    133977  cca342dbebbf9679  Done! \n\nhttp://en.wikipedia.org/w/index.php?...   
    31839   549562f82c4bb970  Let me set the record straight. I am not Delic...   
    
            is_toxic  
    93621          1  
    67803          0  
    16504          0  
    133977         0  
    31839          0  
    
    is_toxic
    0    143346
    1     16225
    Name: count, dtype: int64 
    
    Toxic comments percentage: 10.17%


### Clean up the test labels

In the test labels, there are comments that aren't labelled. These comments have a value of -1 for all the toxic classes. These comments are removed from the test labels.

After which, the same merging of toxic classes is applied. Then, these changes are being reflected on the test dataset.


```python
# remove the comments that have no labels
test_labels_clean = test_labels[
    (test_labels[toxic_classes] != -1)
    .all(axis=1)
]

# create a new column to indicate if the comment is toxic
test_labels_df = create_is_toxic_column(test_labels_clean)

# drop all the toxic classes columns
test_labels_df = test_labels_df.drop(columns=toxic_classes)

print(f"Test labels (filtered) shape: {test_labels_df.shape}\n")
print(f"Sample test labels (filtered):\n{test_labels_df.sample(5)}")

print("\n", '-'*50, "\n")

# merge the test data with the test labels
test_df = pd.merge(test_data, test_labels_df, on='id')

print(f"Test data (filtered) shape: {test_df.shape}\n")
print(f"Sample test data (filtered):\n{test_df.sample(5)}\n")
print(test_df['is_toxic'].value_counts(), "\n")
print(
    f"Toxic comments percentage: {test_df['is_toxic'].value_counts(normalize=True)[1]:.2%}"
)
```

    Test labels (filtered) shape: (63978, 2)
    
    Sample test labels (filtered):
                          id  is_toxic
    104578  ae94f4cf380196b9         0
    88113   92fb29295a55782e         0
    103445  acacd7ceaaa49146         0
    72305   787aca8f8a948521         1
    55763   5cbfbed603c34628         0
    
     -------------------------------------------------- 
    
    Test data (filtered) shape: (63978, 3)
    
    Sample test data (filtered):
                         id                                       comment_text  \
    43436  ad1eadaae6b43e22  "Separate opinions from the definition == \n\n...   
    52535  d199e8471604a638  , 7 August 2006 (UTC) \n #Strong oppose. Per P...   
    52218  d05644a3a464baca   stupid. So shut up or i will delete your account   
    21678  561b4e7f1394f6d9  ==Low German is not a dialect of German== \n\n...   
    35453  8d9f0480c51ebb75  " \n :Chastain, any chef will tell you that yo...   
    
           is_toxic  
    43436         0  
    52535         0  
    52218         1  
    21678         0  
    35453         0  
    
    is_toxic
    0    57735
    1     6243
    Name: count, dtype: int64 
    
    Toxic comments percentage: 9.76%


### Initial thoughts

- The dataset is extremely imbalanced, with only 10% of the comments being toxic. This will be a challenge when making a fair evaluation.
- There is no validation dataset, therefore we will have to split the training dataset into training and validation datasets.


```python
def check_data_quality(df):
    print(f"Total rows: {len(df)}\n")
    print(f"Missing values:\n{df.isna().sum()}\n")
    print(f"Duplicates: {df.duplicated().sum()}")
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
```

    ########## Quality check for train data: ##########
    Total rows: 159571
    
    Missing values:
    id              0
    comment_text    0
    is_toxic        0
    dtype: int64
    
    Duplicates: 0
    Empty comments: 0
    
    Sample texts:
    [
      "Linear B writing = Pelasgians language\n\nIf linear B writing is consider as first form of Greek language then we must assume that this language is originated from Pelasgian language. We must accept that Pelasgian language is a indo –european language and since the oldest recognized such language is Albanian the oldest greek language is originated from oldest Albanian language “The law formulated in 1892 by J. Wackernagel, according to which unstressed parts of the sentence tend to occupy a position after the first stressed word normally situated at the beginning of a sentence qualifies Albanian as the oldest living Indo European language.” Dodona",
      "\"\nFadix, you are in no position to speculate about \"\"good faith\"\" here. Editors who exercise bad faith are those who remove sourced and appropriate, verifiable info such as the Hetq article and other information to balance this article and make it NPOV.  \"",
      "Your category tags on British Commonwealth Honours Systems is deeply offensive!\n\nHu, your tags on British Commonwealth Honours System-related articles\nare deeply offensive. Here's an example of how politically offensive your tag is; Rhodesian Honours System. I want an apology please!-\n( )",
      "Okay, done. What do you think?  hab",
      "Nandesuka's cyberstalking \n\ncyberstalking"
    ]
    
     -------------------------------------------------- 
    
    ########## Quality check for test data: ##########
    Total rows: 63978
    
    Missing values:
    id              0
    comment_text    0
    is_toxic        0
    dtype: int64
    
    Duplicates: 0
    Empty comments: 0
    
    Sample texts:
    [
      "\" \n\n ::I rewrote the section to show both points of view. Fans of Stern seeing it as a joke while saying what Opie & Anthony say.   ' '  \"",
      "If you continue to try to save your hoax article on Nivlek Disease, you will be blocked from editing.",
      "::::I just would like to point out that the ICAO specification 9303 part 5 which is about TD1 size (54x85 mm) does not contain the word passport. This document call them TD1 Size Travel documents. It is the TD3 size specification (88x125 mm) which uses the word passport. Either way if the IPC is called passort or identity card, they will probably be accepted by the EU, since passports or identity cards conforming to some standards will be accepted, but there is a little confusion and some countries define them as identity cards and Ireland and UK seem to define them as passports.",
      "== Advertisement? == \n\n I don't think it reads like an advertisement.",
      "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$mitty supppppppppppppppppppp its me james!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! sup smit the one and olny taylor  \n this is from all ur friends u rule!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ::::::::)"
    ]
    
     -------------------------------------------------- 
    


### Clean the data


```python
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

for name, df in [("Train data", train_df), ("Test data", test_df)]:
    df["comment_text"] = df["comment_text"].apply(clean_data)
    print(
        f"{name} sample texts:\n{json.dumps(df['comment_text'].sample(5).tolist(), indent=2, ensure_ascii=False)}"
    )
    print("\n", '-'*50, "\n")
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rule Tested</th>
      <th>Before</th>
      <th>After</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wiki headers, indentation, URL, IP, HTML entity, word elongation</td>
      <td>== Admin Notice ==\n::: Visit https://en.wikipedia.org or contact 192.168.1.1 immediately! That was sooooo cool &amp;amp; awesome.</td>
      <td>Admin Notice \nVisit [URL] or contact [IP] immediately! That was soo cool &amp; awesome.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CSV escaped quotes, www URL, character elongation, trailing spaces</td>
      <td>""Hey you,"" check www.example.com/rules right nowwwwww!!!!!</td>
      <td>"Hey you," check [URL] right noww!!</td>
    </tr>
  </tbody>
</table>
</div>


    Train data sample texts:
    [
      "I disagree. I think the iconic 1960s Dylan is the most important and recognizable photo to have heading the article. This is a matter that has been discussed here before, and I think the last discussion settled on the 1960s Dylan to head the article. (Which is not to say that, given the right photo(s), the discussion might not go the other way this time.) There are more recent photos further down in the article; perhaps they could be given more prominence. And yes, you are correct, it looks like the lead photo is just a cropped version of the photo with Joan Baez. This isn't ideal, perhaps it could be fixed by changing one photo or the other.",
      "I suppose it is just a guideline, but it's what most teachers adhere to unless other concerns arise. Personally, I never head about the campaign against grade inflation except from here, so I don't even know if that's actually happening.",
      "Hi, I wrote in my edit summary WP:BRD-alert, meaning it was a drastic edit, and you have every right to revert it. When I have time I will come and discuss and perhaps defend it in part at least. In any case, it is mostly up to you to decide, but clearly one has to sometimes make drastic edits otherwise there is nothing to compare to.",
      "Maybe he needs to go fly a kite in a storm.",
      "Code128 Auto \nBesides A, B, C there is Code128 Auto. The explaination is \"Automatically switch between the different code sets and performs characters optimization \n\". But how to do characters optimization?"
    ]
    
     -------------------------------------------------- 
    
    Test data sample texts:
    [
      "Killer Tornado Map \n The map for killer tornadoes needs to be updated. I would do it myself, but I don't know how. New map",
      "Is gaelic not an offical language?",
      "Just to rehash this.. The article has 30k of readable prose which is well within WP:SIZE's guidelines. There are also 38k of references and that is just the references themselves, not the text from the templates cite templates, so it's probably safe to double that 38k to account for the ref tags and use of cite templates, the remainder of the size of the article is the multitude of infoboxes, templates, categories, other languages, etc, etc that exist at the bottom of the article. However, all of that is just a pittance compared to the almost 400k of images that are used in this article. We could trim this article down to just the images and have it be less than 1k of wikitext, but it would still load as a 400k article and still take ages to load via modem because of the images alone.",
      "How can you say that Users who are looking for \"politics of China\" are not looking for the RoC? I am such a user.",
      "my friend says he does this all the time and he never ever got blocked i think you are kidding"
    ]
    
     -------------------------------------------------- 
    


A new column `has_swear` is also created to determine if the comments contain any swear words from the lexicon.


```python
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
```

    Train data - Comments with swear words: 10971 out of 159571 (6.88%)
    Test data - Comments with swear words: 6108 out of 63978 (9.55%)


### Split the data into train and validation datasets


```python
train_df_backup = train_df.copy()

train_df, valid_df = train_test_split(
    train_df_backup,
    test_size=0.2,
    shuffle=True,
    random_state=13,
    stratify=train_df_backup[["is_toxic", "has_swear"]],
)
```

### Split the test dataset into 4 slices


```python
# create word count column
test_df["word_count"] = test_df["comment_text"].apply(lambda x: len(x.split()))

# remap binary values to labels
is_toxic = test_df["is_toxic"].map({1: "toxic", 0: "non_toxic"})
has_swear = test_df["has_swear"].map({1: "with_swear", 0: "without_swear"})

# create slice column
test_df["slice"] = is_toxic + "_" + has_swear

# test_slices = {
#     "toxic_with_swear": test_df[test_df["slice"] == "toxic_with_swear"],
#     "toxic_no_swear": test_df[test_df["slice"] == "toxic_no_swear"],
#     "non_toxic_with_swear": test_df[test_df["slice"] == "non_toxic_with_swear"],
#     "non_toxic_no_swear": test_df[test_df["slice"] == "non_toxic_no_swear"],
# }

test_df["slice"].value_counts()
```




    slice
    non_toxic_without_swear    54861
    toxic_with_swear            3234
    toxic_without_swear         3009
    non_toxic_with_swear        2874
    Name: count, dtype: int64




```python
os.makedirs("./data/dataset", exist_ok=True)

train_df.to_csv("./data/dataset/train_clean.csv", index=False)
valid_df.to_csv("./data/dataset/valid_clean.csv", index=False)
test_df.to_csv("./data/dataset/test_clean.csv", index=False)
```

### Checkpoint for the dataframes

Saving the dataframes as parquet files to avoid reprocessing the data.


```python
FORCE_SAVE_DF = False

os.makedirs("./data/dataframe", exist_ok=True)
files_exists = all(
    os.path.exists(f"./data/dataframe/{s}.parquet")
    for s in ["train", "valid", "test"]
)

if not files_exists or FORCE_SAVE_DF:
    train_df.to_parquet("./data/dataframe/train.parquet")
    valid_df.to_parquet("./data/dataframe/valid.parquet")
    test_df.to_parquet("./data/dataframe/test.parquet")
    print("Dataframes saved")
else:
    train_df = pd.read_parquet("./data/dataframe/train.parquet")
    valid_df = pd.read_parquet("./data/dataframe/valid.parquet")
    test_df = pd.read_parquet("./data/dataframe/test.parquet")
    print("Dataframes loaded")
```

    Dataframes saved


#### Overview


```python
for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
    print(f"{'#'*10} {name} Dataset (Total: {len(df):,d}) {'#'*10}\n")

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

    print("\n", '-' * 50, "\n")
```

    ########## Train Dataset (Total: 127,656) ##########
    
                       No swear     With swear                All
    Non-toxic  112,764 (88.33%)  1,912 (1.50%)   114,676 (89.83%)
    Toxic         6,115 (4.79%)  6,865 (5.38%)    12,980 (10.17%)
    All        118,879 (93.12%)  8,777 (6.88%)  127,656 (100.00%) 
    
    52.89% of toxic comments contain swear words
    1.67% of non-toxic comments contain swear words
    
     -------------------------------------------------- 
    
    ########## Valid Dataset (Total: 31,915) ##########
    
                      No swear     With swear               All
    Non-toxic  28,192 (88.33%)    478 (1.50%)   28,670 (89.83%)
    Toxic        1,529 (4.79%)  1,716 (5.38%)    3,245 (10.17%)
    All        29,721 (93.13%)  2,194 (6.87%)  31,915 (100.00%) 
    
    52.88% of toxic comments contain swear words
    1.67% of non-toxic comments contain swear words
    
     -------------------------------------------------- 
    
    ########## Test Dataset (Total: 63,978) ##########
    
                      No swear     With swear               All
    Non-toxic  54,861 (85.75%)  2,874 (4.49%)   57,735 (90.24%)
    Toxic        3,009 (4.70%)  3,234 (5.05%)     6,243 (9.76%)
    All        57,870 (90.45%)  6,108 (9.55%)  63,978 (100.00%) 
    
    51.80% of toxic comments contain swear words
    4.98% of non-toxic comments contain swear words
    
     -------------------------------------------------- 
    


## Baselines

### Floor baselines


```python
# majority class baseline
majority_baseline = DummyClassifier(strategy="most_frequent")
majority_baseline.fit(train_df["comment_text"], train_df["is_toxic"])

# random baseline
random_baseline = DummyClassifier(strategy="stratified", random_state=13)
_ = random_baseline.fit(train_df["comment_text"], train_df["is_toxic"])
```

### Rule-based baseline

The rule-based baseline is based on the assumption that toxic comments contain swear words. Thus, if a comment contains any swear words, it is classified as toxic. This can be achieved by referencing the `has_swear` column which was created previously above.

#### Traditional ML baseline (TF-IDF + Logistic Regression)


```python
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

_ = tfidf_lr_baseline.fit(train_df["comment_text"], train_df["is_toxic"])
```


```python
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
```


```python
y_true = test_df["is_toxic"]
test_texts = test_df["comment_text"]

y_pred_majority = majority_baseline.predict(test_texts)
y_proba_majority = majority_baseline.predict_proba(test_texts)[:, 1]
result_majority = evaluate_model(
    y_true, y_pred_majority, y_proba_majority, "Majority Baseline")

y_pred_random = random_baseline.predict(test_texts)
y_proba_random = random_baseline.predict_proba(test_texts)[:, 1]
result_random = evaluate_model(
    y_true, y_pred_random, y_proba_random, "Random Baseline")

y_pred_rule = test_df["has_swear"]
result_rule = evaluate_model(y_true, y_pred_rule, None, "Rule-Based Baseline")

y_pred_tfidf_lr = tfidf_lr_baseline.predict(test_texts)
y_proba_tfidf_lr = tfidf_lr_baseline.predict_proba(test_texts)[:, 1]
result_tfidf_lr = evaluate_model(
    y_true, y_pred_tfidf_lr, y_proba_tfidf_lr, "TF-IDF + Logistic Regression")

test_results = pd.DataFrame(
    [result_majority, result_random, result_rule, result_tfidf_lr]
)
```

## Pre-trained model


```python
MODEL_NAME = "microsoft/deberta-v3-base" if is_cuda else "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sample_text = "This is a great project, not toxic at all!"
tokens = tokenizer(sample_text)
decoded_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"])

token_preview = pd.DataFrame({
    "Token": decoded_tokens,
    "Token ID": tokens["input_ids"]
})

display(token_preview)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Token</th>
      <th>Token ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[CLS]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>▁This</td>
      <td>329</td>
    </tr>
    <tr>
      <th>2</th>
      <td>▁is</td>
      <td>269</td>
    </tr>
    <tr>
      <th>3</th>
      <td>▁a</td>
      <td>266</td>
    </tr>
    <tr>
      <th>4</th>
      <td>▁great</td>
      <td>426</td>
    </tr>
    <tr>
      <th>5</th>
      <td>▁project</td>
      <td>663</td>
    </tr>
    <tr>
      <th>6</th>
      <td>,</td>
      <td>261</td>
    </tr>
    <tr>
      <th>7</th>
      <td>▁not</td>
      <td>298</td>
    </tr>
    <tr>
      <th>8</th>
      <td>▁toxic</td>
      <td>7290</td>
    </tr>
    <tr>
      <th>9</th>
      <td>▁at</td>
      <td>288</td>
    </tr>
    <tr>
      <th>10</th>
      <td>▁all</td>
      <td>305</td>
    </tr>
    <tr>
      <th>11</th>
      <td>!</td>
      <td>300</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[SEP]</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)


def tokenize(batch):
    tokens = tokenizer(batch['comment_text'], truncation=True, max_length=128)
    tokens["labels"] = batch["is_toxic"]
    return tokens


train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["id"])
valid_dataset = valid_dataset.remove_columns(["id"])
test_dataset = test_dataset.remove_columns(["id"])
```


    Map:   0%|          | 0/127656 [00:00<?, ? examples/s]



    Map:   0%|          | 0/31915 [00:00<?, ? examples/s]



    Map:   0%|          | 0/63978 [00:00<?, ? examples/s]



```python
LOCAL_RUN = False

if LOCAL_RUN:
    train_dataset = train_dataset.shuffle(seed=13).select(range(3000))
    valid_dataset = valid_dataset.shuffle(seed=13).select(range(800))
    # test_dataset = test_dataset.shuffle(seed=13).select(range(1000))
```


```python
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
    num_train_epochs=3 if LOCAL_RUN else 3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    gradient_accumulation_steps=target_batch_size // batch_size,
    gradient_checkpointing=False,
    # gradient_checkpointing_kwargs={"every_n_layers": 4},
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

trainer.train()
```

    [transformers] `torch_dtype` is deprecated! Use `dtype` instead!
    /srv/data/guswonju/lt2326-2026-machine-learning-project/.venv/lib64/python3.12/site-packages/torch/jit/_script.py:1491: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
      warnings.warn(
    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    Loading weights:   0%|          | 0/198 [00:00<?, ?it/s]


    [transformers] [1mDebertaV2ForSequenceClassification LOAD REPORT[0m from: microsoft/deberta-v3-base
    Key                                     | Status     | 
    ----------------------------------------+------------+-
    lm_predictions.lm_head.dense.bias       | UNEXPECTED | 
    lm_predictions.lm_head.bias             | UNEXPECTED | 
    mask_predictions.dense.weight           | UNEXPECTED | 
    lm_predictions.lm_head.LayerNorm.weight | UNEXPECTED | 
    mask_predictions.classifier.weight      | UNEXPECTED | 
    mask_predictions.LayerNorm.bias         | UNEXPECTED | 
    mask_predictions.dense.bias             | UNEXPECTED | 
    mask_predictions.classifier.bias        | UNEXPECTED | 
    lm_predictions.lm_head.dense.weight     | UNEXPECTED | 
    lm_predictions.lm_head.LayerNorm.bias   | UNEXPECTED | 
    mask_predictions.LayerNorm.weight       | UNEXPECTED | 
    pooler.dense.bias                       | MISSING    | 
    pooler.dense.weight                     | MISSING    | 
    classifier.bias                         | MISSING    | 
    classifier.weight                       | MISSING    | 
    
    Notes:
    - UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
    [transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 2, 'bos_token_id': 1}.




    <div>

      <progress value='11970' max='11970' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [11970/11970 27:16, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.184748</td>
      <td>0.085860</td>
      <td>0.966500</td>
      <td>0.806300</td>
      <td>0.882600</td>
      <td>0.842700</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.175539</td>
      <td>0.083455</td>
      <td>0.970100</td>
      <td>0.843700</td>
      <td>0.866600</td>
      <td>0.855000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.068746</td>
      <td>0.109804</td>
      <td>0.969200</td>
      <td>0.835200</td>
      <td>0.868100</td>
      <td>0.851300</td>
    </tr>
  </tbody>
</table><p>



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]





    TrainOutput(global_step=11970, training_loss=0.14812782030854507, metrics={'train_runtime': 1637.2437, 'train_samples_per_second': 233.91, 'train_steps_per_second': 7.311, 'total_flos': 2.5015493759816256e+16, 'train_loss': 0.14812782030854507, 'epoch': 3.0})




```python
model_directory = "./artifact/model/best"

trainer.save_model(model_directory)

print(f"Model saved to {model_directory}")
```


    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]


    Model saved to ./artifact/model/best



```python
model = AutoModelForSequenceClassification.from_pretrained(model_directory)
model.to(DEVICE)
model.eval()
```


```python
os.makedirs("./artifact/logits", exist_ok=True)

logits_path = "./artifact/logits/deberta_test_logits.npy"

if os.path.exists(logits_path):
    print("loading saved predictions")
    logits = np.load(logits_path)
else:
    test_pred = trainer.predict(test_dataset)
    logits = test_pred.predictions

    np.save(logits_path, logits)
    print(f"saved predictions to {logits_path}")


y_pred_deberta = logits.argmax(axis=1)
y_proba_deberta = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

deberta_name = "DeBERTa (base)" if is_cuda else "DeBERTa (small)"

result_deberta = evaluate_model(
    y_true,
    y_pred_deberta,
    y_proba_deberta,
    model_name=deberta_name
)

test_results = pd.concat([test_results, pd.DataFrame([result_deberta])])

display(test_results)
```





    saved predictions to ./artifact/logits/deberta_test_logits.npy



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
      <th>PR-AUC</th>
      <th>Recall (toxic with swear)</th>
      <th>Recall (toxic without swear)</th>
      <th>Delta Recall</th>
      <th>FPR (non-toxic with swear)</th>
      <th>FPR (non-toxic without swear)</th>
      <th>Delta FPR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Majority Baseline</td>
      <td>0.9024</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0976</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Baseline</td>
      <td>0.8219</td>
      <td>0.1000</td>
      <td>0.1032</td>
      <td>0.1016</td>
      <td>0.0978</td>
      <td>0.1030</td>
      <td>0.1034</td>
      <td>-0.0004</td>
      <td>0.1023</td>
      <td>0.1003</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rule-Based Baseline</td>
      <td>0.9080</td>
      <td>0.5295</td>
      <td>0.5180</td>
      <td>0.5237</td>
      <td>N/A</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TF-IDF + Logistic Regression</td>
      <td>0.8722</td>
      <td>0.4267</td>
      <td>0.9020</td>
      <td>0.5793</td>
      <td>0.7474</td>
      <td>0.9536</td>
      <td>0.8465</td>
      <td>0.1072</td>
      <td>0.6500</td>
      <td>0.1039</td>
      <td>0.5461</td>
    </tr>
    <tr>
      <th>0</th>
      <td>DeBERTa (small)</td>
      <td>0.9074</td>
      <td>0.5141</td>
      <td>0.9286</td>
      <td>0.6618</td>
      <td>0.8099</td>
      <td>0.9682</td>
      <td>0.8860</td>
      <td>0.0821</td>
      <td>0.5772</td>
      <td>0.0696</td>
      <td>0.5076</td>
    </tr>
  </tbody>
</table>
</div>



```python

```
