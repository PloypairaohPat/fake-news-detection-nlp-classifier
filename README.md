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
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spaCy language model (one-time)
python -m spacy download en_core_web_sm

# 3. Run the Streamlit app (model is pre-trained — no training required)
streamlit run app/dashboard.py
```

To retrain the model from scratch, run the notebooks in order:

```
notebooks/EDA.ipynb → preprocessing.ipynb → tfidf_baseline.ipynb
```

`tfidf_baseline.ipynb` will overwrite `models/tfidf_xgboost_pipeline.pkl` with the retrained pipeline.

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
