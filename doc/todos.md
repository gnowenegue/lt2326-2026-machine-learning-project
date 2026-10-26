# The Role of Swear Words in Toxicity Detection

**Research Question:** *How does the presence or absence of swear words affect the performance of toxicity detection systems?*
**Author:** Eugene Wong
**Dataset:** Jigsaw Toxic Comment Classification Challenge (Kaggle 2018)
**Lexicon:** NoSwearing Dictionary

---

## 1. Setup & Data Preparation

- [x] Set up project environment
- [x] Download and clean the swear word list
- [x] Filter out words from the list that aren't actually swearing
- [x] Load the Jigsaw dataset
- [x] Filter out unlabelled test comments
- [x] Combine toxic categories into a simple toxic vs. non-toxic label (`is_toxic`)
- [x] Clean the comment text
- [x] Split data into training, validation, and test sets

## 2. Test Groups (Slices)

Split the test comments into 4 groups to test swear word bias:

- [x] **Toxic + Swear:** Obvious attacks and toxic comments with swear words
- [x] **Toxic + No Swear:** Subtle attacks and toxic comments without swear words
- [x] **Non-Toxic + Swear:** Harmless comments with casual swearing (false alarms)
- [x] **Non-Toxic + No Swear:** Normal comments without any swearing (clean baseline)
- [x] Measure the performance gap between swear and non-swear comments

## 3. Baseline Models

Test simple models on the test set and all 4 groups:

- [x] Simple baselines (always guess non-toxic & random guessing)
- [x] Dictionary baseline (flag comment as toxic if it has a swear word)
- [x] Classic machine learning baseline (TF-IDF + Logistic Regression)

## 4. Deep Learning Model (DeBERTa)

Train and test a transformer model (DeBERTa):

- [x] Prepare and tokenize text for DeBERTa
- [x] Test training locally with a smaller model (`deberta-v3-small`)
- [x] Train `deberta-v3-base` on the full training set
- [x] Run predictions on the test set
- [x] Compare all models in a single results table

## 5. Looking at Mistakes (Error Analysis)

Look at real misclassified comments to understand *why* models fail:

- [x] Check which words Logistic Regression relies on the most (coefficients)
- [x] Analyze false alarms: Compare how TF-IDF and DeBERTa handle harmless swearing
- [x] Analyze missed attacks: Compare how TF-IDF and DeBERTa handle toxic comments without swear words

## 6. Final Report & Submission

- [ ] Summarize the main takeaways from the results table
- [ ] Write up the discussion and findings
- [ ] Final check to make sure code runs cleanly from scratch
