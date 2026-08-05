# Fake News Detection NLP Classifier

Binary text classifier that distinguishes fake from real news articles with **97% macro F1** on a held-out test set of 14,419 articles. Combines a spaCy preprocessing pipeline (lemmatisation + stopword removal) with TF-IDF feature extraction and an XGBoost classifier, served via a live Streamlit app.

**Stack:** Python · scikit-learn · XGBoost · spaCy · NLTK · Streamlit

---

## Key Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Fake (0) | 0.98 | 0.96 | 0.97 |
| Real (1) | 0.96 | 0.98 | 0.97 |
| **Macro avg** | **0.97** | **0.97** | **0.97** |

Trained on 57,715 articles, evaluated on 14,419 — stratified split, class-balanced dataset.

---

## Dataset

**WELFake** — 72,134 news articles aggregated from four public sources (Kaggle, McIntire, Reuters, BuzzFeed Political News).

- Articles per class: ~37,000 fake / ~35,000 real (near-balanced)
- Features used: `title` + `text` concatenated into a single `combined` field
- Labels: `0` = Fake, `1` = Real

---

## Model Architecture

```
Input text
    │
    ▼
TextPreprocessor  (spaCy: lowercase → URL/HTML strip → lemmatise → remove stopwords)
    │
    ▼
TfidfVectorizer   (50,000 features · unigrams + bigrams · sublinear TF · min_df=3)
    │
    ▼
XGBClassifier     (300 estimators · max_depth=6 · learning_rate=0.1)
    │
    ▼
Prediction: FAKE / REAL + probability
```

The full pipeline is serialised to `models/tfidf_xgboost_pipeline.pkl` and loaded by the Streamlit app at runtime.

---

## Project Structure

```
Fake News Detection NLP Classifier/
├── data/
│   ├── WELFake_Dataset.csv         # 72,134 labelled articles (source dataset)
│   ├── train_distilbert.csv        # Train split exported for future transformer work
│   └── test_distilbert.csv         # Test split exported for future transformer work
├── notebooks/
│   ├── EDA.ipynb                   # Class balance, article length, text inspection
│   ├── preprocessing.ipynb         # TextPreprocessor class definition and testing
│   └── tfidf_baseline.ipynb        # TF-IDF + XGBoost training, evaluation, model save
├── models/
│   └── tfidf_xgboost_pipeline.pkl  # Serialised sklearn Pipeline (ready for inference)
├── app/
│   └── dashboard.py                # Streamlit app — paste an article, get a verdict
├── requirements.txt                # Dashboard-only deps (Streamlit Cloud)
└── requirements-dev.txt            # Full pipeline deps (EDA + training notebooks)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spaCy language model (one-time — the app also self-downloads
#    it on first launch if this step is skipped, e.g. on a fresh deploy)
python -m spacy download en_core_web_sm

# 3. Run the Streamlit app (model is pre-trained and committed to the repo —
#    no training required)
streamlit run app/dashboard.py
```

To retrain the model from scratch, install the full pipeline deps and run the notebooks in order:

```bash
pip install -r requirements-dev.txt
```

```
notebooks/EDA.ipynb → preprocessing.ipynb → tfidf_baseline.ipynb
```

`tfidf_baseline.ipynb` will overwrite `models/tfidf_xgboost_pipeline.pkl` with the retrained pipeline.

---

## Deploying to Streamlit Cloud

1. Point the app at `app/dashboard.py` as the main file path.
2. Streamlit Cloud installs `requirements.txt` only — it does not run the `python -m spacy download` step from Quick Start. `app/dashboard.py`'s `load_nlp()` handles this: it tries `spacy.load("en_core_web_sm")` first and falls back to downloading the model on first boot if it's missing, so a fresh deploy works without a manual step.
3. `models/tfidf_xgboost_pipeline.pkl` is committed to the repo (2.7 MB, well under GitHub's 100 MB limit — no Git LFS needed), so the deployed app has a trained model to load without running any notebook first.

---

## Preprocessing Pipeline

The `TextPreprocessor` class (sklearn-compatible transformer) applies five steps in sequence:

1. Lowercase
2. Remove URLs and HTML tags
3. Strip non-alphabetic characters
4. spaCy lemmatisation
5. Remove English stopwords and short tokens (≤ 2 characters)

Example:

```
Input:  "BREAKING: The President said vaccines cause autism!!! https://news.com"
Output: "break president say vaccine cause autism"
```

---

## Limitations & Generalization

The 97% macro F1 above is measured on a stratified held-out split of WELFake itself, not on independent, out-of-distribution news — it reflects how well the model fits *this* four-source aggregation, not how well it detects fabricated news in general. WELFake and its component sources carry well-documented dataset artifacts: differences in formatting, punctuation, wire-service boilerplate, and article length correlate with the fake/real label as an accident of how the sources were assembled, not because they are genuine signals of misinformation. A TF-IDF + XGBoost model is well-positioned to exploit exactly these superficial regularities, so a meaningful share of the reported 97% likely reflects the model learning source and formatting fingerprints rather than deceptive content itself. Expect materially lower accuracy on genuinely out-of-distribution text: articles from outlets not represented in WELFake, news published after the dataset's collection window, differently formatted input (social posts, press releases, non-wire-service prose), or fake articles deliberately written to avoid the dataset's stylistic tells. Treat this project as a demonstration of the TF-IDF + XGBoost pipeline on a benchmark dataset, not as a production-ready misinformation detector — a real deployment would need evaluation on held-out sources and time periods the model has never seen, and should pair any prediction with human review rather than trusting the verdict standalone.

---

## Tech Stack

- **Python** 3.10+
- **pandas**, **NumPy** — data loading and manipulation
- **scikit-learn** — pipeline, TF-IDF, train/test split, metrics
- **XGBoost** — gradient boosted classifier
- **spaCy** (`en_core_web_sm`) — lemmatisation
- **NLTK** — stopwords corpus
- **Streamlit** — interactive demo app
- **Matplotlib**, **seaborn** — EDA visualisation
- **Jupyter** — notebook environment
