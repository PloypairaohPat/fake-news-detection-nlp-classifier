import pickle
import re
from pathlib import Path

import nltk
import spacy
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# ── must be defined before loading the pkl so unpickling can find the class ──
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])

nlp = load_nlp()

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._clean(text) for text in X]

    def _clean(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        doc = nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if token.text not in STOPWORDS
            and len(token.text) > 2
            and not token.is_space
        ]
        return " ".join(tokens)


MODEL_PATH = Path(__file__).parent.parent / "models" / "tfidf_xgboost_pipeline.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
)

st.title("Fake News Detector")
st.caption(
    "TF-IDF + XGBoost classifier trained on 72,134 articles from the WELFake dataset · 97% macro F1"
)

st.divider()

# ── input ────────────────────────────────────────────────────────────────────
headline = st.text_input("Headline (optional)", placeholder="e.g. Scientists discover cure for cancer")

body = st.text_area(
    "Article body",
    height=200,
    placeholder="Paste the article text here…",
)

# ── predict ───────────────────────────────────────────────────────────────────
if st.button("Analyse", type="primary", use_container_width=True):
    combined = f"{headline} {body}".strip()

    if not combined:
        st.warning("Please enter some text before analysing.")
    else:
        with st.spinner("Analysing…"):
            pipeline = load_model()
            proba = pipeline.predict_proba([combined])[0]
            pred = int(proba[1] > proba[0])
            confidence = float(proba[pred])

        st.divider()

        if pred == 1:
            st.success(f"REAL  —  {confidence:.1%} confidence")
        else:
            st.error(f"FAKE  —  {confidence:.1%} confidence")

        col1, col2 = st.columns(2)
        col1.metric("Real probability", f"{proba[1]:.1%}")
        col2.metric("Fake probability", f"{proba[0]:.1%}")

        st.progress(float(proba[1]), text="Real ← confidence → Fake")

# ── sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        """
**Model:** TF-IDF (50k features, bigrams) + XGBoost (300 estimators)

**Dataset:** WELFake — 72,134 articles aggregated from four sources
(Kaggle, McIntire, Reuters, BuzzFeed)

**Performance on held-out test set (14,419 articles):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Fake | 0.98 | 0.96 | 0.97 |
| Real | 0.96 | 0.98 | 0.97 |
| **Macro avg** | **0.97** | **0.97** | **0.97** |

**Preprocessing:** lowercase → URL/HTML removal → regex clean
→ spaCy lemmatisation → stopword removal
        """
    )
    st.divider()
    st.caption("Labels: 0 = Fake · 1 = Real")
