# The Role of Swear Words in Toxicity Detection

Author: Eugene Wong

Research Question:\
**How does the presence or absence of swear words affect the performance of toxicity detection systems?**

## Why

Swear words are often associated with toxicity, but the two are not the same thing. People swear for humour, excitement, emphasis or frustration without meaning any harm, and a comment can be hostile without containing a single swear word. This project investigates whether toxicity detection systems struggle in these two situations. I expect more mistakes when swear words and toxicity do not match, such as non-toxic comments containing swear words and toxic comments containing none.

## Who

This research may benefit social media platforms, online forums, and content moderators who use automated tools to help flag harmful comments. A system that leans too heavily on swear words may wrongly flag harmless content, or miss toxic content that contains none.

## Related Work

Previous work has shown that swear words appear in both abusive and non-abusive contexts, and that toxicity classifiers may rely too strongly on particular word cues ([Pamungkas et al., 2020](https://aclanthology.org/2020.lrec-1.765/); [Zhou et al., 2021](https://aclanthology.org/2021.eacl-main.274/)). [HateCheck (Röttger et al., 2021)](https://aclanthology.org/2021.acl-long.4/) also showed that overall scores can hide specific weaknesses in hate-speech detection models. This project follows that idea by testing toxicity detection separately on comments with and without swear words.

## What

This project will use the [Jigsaw Toxic Comment Classification Challenge, 2018](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset, which contains Wikipedia comments labelled by human raters for toxic behaviour. I chose it because it focuses directly on toxicity classification. Following Pamungkas et al. (2020), I will use the [NoSwearing](https://www.noswearing.com) dictionary to identify comments containing swear words. One limitation is that a dictionary lookup may miss obfuscated spellings such as "f*ck" or "sh1t", so some comments I count as swear-free could still contain swearing.

For the baseline, I will use **TF-IDF** with **Logistic Regression** as a traditional machine-learning approach. I will compare it with a small pretrained encoder-only transformer, which can use the surrounding context of each word. Possible models include **DistilBERT**, **ELECTRA-small**, or **DeBERTa-v3-small**.

## Evaluation

I will first evaluate both models on the full test set using **precision**, **recall**, and **F1 score**. The test set will then be divided into four subsets: **toxic comments with swear words**, **toxic comments without swear words**, **non-toxic comments with swear words**, and **non-toxic comments without swear words**.

For the two toxic subsets, I will use recall to measure how often each model correctly detects toxic comments. For the two non-toxic subsets, I will use false positive rate to measure how often harmless comments are wrongly classified as toxic. I will compare the recall difference between toxic comments with and without swear words, and separately compare the false positive rate difference between non-toxic comments with and without swear words. These differences will then be compared between the two models to see whether one is less affected by the presence or absence of swear words.
