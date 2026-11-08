# The Role of Swear Words in Toxicity Detection

Do toxicity classifiers actually understand swear words, or do they just freak out whenever they see one?

This research project looks at how the presence and absence of swear words affect toxic comment detection models. In practice, models tend to make two big mistakes:

1. **False alarms:** Flagging harmless comments just because of casual swearing (like *"this is fucking awesome"*).
2. **Blind spots:** Completely missing toxic comments and harassment when written without obvious bad words.

We test basic rule filters, TF-IDF with Logistic Regression, and a fine-tuned DeBERTa model on the Wikipedia Jigsaw dataset to see where models fail and why.

## Project Documentation

* **Project Proposal:** [doc/proposal/proposal.pdf](doc/proposal/proposal.pdf)
* **Presentation Slides:** [doc/presentation/slides.pdf](doc/presentation/slides.pdf)
* **Final Report:** [doc/report/report.pdf](doc/report/report.pdf)
* **AI Assistance Logs:** [doc/ai_logs.pdf](doc/ai_logs.pdf) ([Markdown source](doc/ai_logs.md))

## Datasets & Lexicon

* **Toxicity Dataset:** [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) (Wikipedia talk page comments labelled for toxicity). The notebook downloads it automatically via `kagglehub` if Kaggle credentials are configured, or reads local copies from `data/dataset_raw/`.
* **Profanity Lexicon:** [NoSwearing.com](https://www.noswearing.com) dictionary, used to deterministically identify swear words and build evaluation test slices.

## Fine-Tuned Model

The fine-tuned model is hosted on the Hugging Face Hub:
* **Model:** [`gnowenegue/deberta-v3-base-toxic-classification`](https://huggingface.co/gnowenegue/deberta-v3-base-toxic-classification)

You can use it directly in Python with the `transformers` pipeline:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="gnowenegue/deberta-v3-base-toxic-classification"
)

# Test with a harmless comment containing casual slang
print(classifier("This is freaking awesome!"))
```

## How to Run

1. **Install dependencies:**
   Make sure you have [`uv`](https://github.com/astral-sh/uv) installed, then run:

   ```bash
   uv sync
   ```

2. **Run the notebook or view the walkthrough:**
   - Run interactively: Open [`main.ipynb`](main.ipynb) and run all cells. It walks through data cleaning, baseline models, DeBERTa evaluation, and real-world misclassification examples.
   - Read offline / in browser: Check [`main.md`](main.md) for the rendered Markdown version of the notebook with outputs and analysis.
