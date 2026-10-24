# Project Roadmap: The Role of Swear Words in Toxicity Detection

**Research Question:** *How does the presence or absence of swear words affect the performance of toxicity detection systems?*
**Author:** Eugene Wong
**Dataset:** Jigsaw Toxic Comment Classification Challenge (Kaggle 2018)
**Lexicon:** NoSwearing Dictionary

---

## High-Level Checklist

- [x] **Phase 1: Environment & Project Setup**
  - [x] Configure dependency management with `uv` and pin Python 3.12.
  - [x] Verify hardware acceleration (Apple Silicon MPS / CUDA).
  - [x] Finalize `.gitignore` and project structure.
  - [x] Establish initial git tracking and commit history.

- [x] **Phase 2: Data Acquisition & Preprocessing**
  - [x] Download Jigsaw Toxic Comment dataset via `kagglehub`.
  - [x] Filter out unlabelled test rows (`toxic == -1`).
  - [x] Define binary ground truth target (`is_toxic`).
  - [x] Implement text normalization and cleaning.
  - [x] Create reproducible train / validation / test splits.

- [x] **Phase 3: Lexicon Processing & Comment Tagging**
  - [x] Acquire raw NoSwearing lexicon (`data/lexicon/noswearing_raw.xml`).
  - [x] Parse, normalize, and deduplicate lexicon word list.
  - [x] Implement robust token/word-boundary matching to avoid substring false positives.
  - [x] Tag dataset splits with swear presence (`has_swear`) and inspect distributions.

- [x] **Phase 4: Test Set Slicing (2x2 Evaluation Matrix)**
  - [x] Construct the four evaluation slices from the test set:
    - **`toxic_with_swear`** ($T_{sw}$): Toxic with swear words *(Overt toxicity)*
    - **`toxic_no_swear`** ($T_{nosw}$): Toxic without swear words *(Covert toxicity)*
    - **`non-toxic_with_swear`** ($NT_{sw}$): Non-toxic with swear words *(Benign profanity / false alarm risk)*
    - **`non-toxic_no_swear`** ($NT_{nosw}$): Non-toxic without swear words *(Clean negative baseline)*
  - [x] Generate summary statistics for each slice (sample count, proportion, length distribution).

- [x] **Phase 5: Baseline Models**
  - [x] **Floor baselines**: Majority class classifier and random guessing baseline.
  - [x] **Rule-based baseline**: Direct lexicon lookup (predict toxic if swear present).
  - [x] **Traditional ML baseline**: TF-IDF feature extraction + Logistic Regression.

- [ ] **Phase 6: Transformer Fine-Tuning**
  - [x] Select pre-trained encoder architecture (e.g., DistilBERT or DeBERTa-v3).
  - [x] Build tokenization and batching pipeline.
  - [ ] Train binary classifier with validation tracking and save best checkpoint.

- [ ] **Phase 7: Sliced Evaluation & Comparative Analysis**
  - [ ] Evaluate all models on the full test set (Precision, Recall, F1, PR-AUC, ROC-AUC).
  - [ ] Compute core sliced comparative metrics:
    - **Sensitivity drop on toxic comments:** $\Delta \text{Recall} = \text{Recall}(T_{sw}) - \text{Recall}(T_{nosw})$
    - **Benign profanity penalty:** $\Delta \text{FPR} = \text{FPR}(NT_{sw}) - \text{FPR}(NT_{nosw})$
  - [ ] Build unified comparative results table and visualization plots.

- [ ] **Phase 8: Qualitative Error Analysis**
  - [ ] Analyze false positives in $NT_{sw}$ (harmless swearing flagged as toxic).
  - [ ] Analyze false negatives in $T_{nosw}$ (covert toxicity missed without swear words).
  - [ ] Inspect feature weights / model confidence distributions across slices.

- [ ] **Phase 9: Stretch Goals (Optional)**
  - [ ] Counterfactual perturbation tests (swapping or removing swear words).
  - [ ] Breakdown by toxicity sub-dimensions (`threat`, `insult`, `identity_hate`).

- [ ] **Phase 10: Final Deliverables & Report**
  - [ ] Clean and document pipeline for reproducibility.
  - [ ] Write final report following the proposal outline and research findings.
