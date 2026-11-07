# The Role of Swear Words in Toxicity Detection

Do toxicity classifiers actually understand swear words, or do they just freak out whenever they see one?

This research project looks at how the presence and absence of swear words affect toxic comment detection models. In practice, models tend to make two big mistakes:

1. **False alarms:** Flagging harmless comments just because of casual swearing (like *"this is fucking awesome"*).
2. **Blind spots:** Completely missing toxic comments and harassment when written without obvious bad words.

We test basic rule filters, TF-IDF with Logistic Regression, and a fine-tuned DeBERTa model on the Wikipedia Jigsaw dataset to see where models fail and why.

## Project Documentation

* **Project Proposal:** [doc/proposal/proposal.pdf](doc/proposal/proposal.pdf)
* **Presentation Slides:** [doc/presentation/slides.pdf](doc/presentation/slides.pdf)
* **Final Report (WIP):** [doc/report/report.pdf](doc/report/report.pdf)

## How to Run

1. **Install dependencies:**
   Make sure you have [`uv`](https://github.com/astral-sh/uv) installed, then run:

   ```bash
   uv sync
   ```

2. **Run the notebook:**
   Open [`main.ipynb`](main.ipynb) and run all cells. It walks through data cleaning, the baseline models, the DeBERTa evaluation, and real-world misclassification examples.
