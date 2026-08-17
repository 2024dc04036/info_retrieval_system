"""
preprocess.py
-------------
Turns raw article text into clean tokens.

This is the "text mining" foundation of the assignment. Every later step
(indexing, search, recommendation) works on the cleaned output produced here.

Steps performed:
  1. Lower-casing.
  2. Removing punctuation / numbers (keep letters only).
  3. Splitting into tokens (words).
  4. Removing very common "stop words" (the, and, of, ...).
  5. Removing very short tokens (length < 3).

We deliberately use a small, hand-written stop-word list and simple rules so the
code stays readable and easy to explain, rather than pulling in heavy NLP tools.
"""

import re
from collections import Counter

# A compact English stop-word list. Enough to clean up results without needing
# an external download.
STOPWORDS = set("""
a an the and or but if while of to in on at by for with about against between
into through during before after above below from up down out off over under
is are was were be been being have has had do does did doing this that these
those i you he she it we they them his her its their our your my me us as not
no nor only own same so than too very can will just should now also which who
whom what when where why how all any both each few more most other some such
""".split())


def clean_text(text):
    """Return a single lowercase string with only letters and spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)     # drop anything that is not a-z or space
    text = re.sub(r"\s+", " ", text)          # collapse repeated whitespace
    return text.strip()


def tokenize(text):
    """
    Full pipeline: clean -> split -> remove stop words & short tokens.
    Returns a list of tokens.
    """
    cleaned = clean_text(text)
    tokens = cleaned.split(" ")
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    return tokens


def token_string(text):
    """Convenience: return the cleaned tokens joined back into one string.

    scikit-learn's vectorizers want a string per document, so we feed them this.
    """
    return " ".join(tokenize(text))


def top_keywords(text, n=10):
    """Return the n most frequent tokens in a document (simple keyword extraction)."""
    tokens = tokenize(text)
    return Counter(tokens).most_common(n)


def document_profile(doc):
    """
    Build a small 'profile' of one document for the UI.

    Returns raw-vs-clean counts plus the top keywords — handy to SHOW how
    preprocessing shrinks and focuses the text.
    """
    raw_words = len(doc["content"].split())
    tokens = tokenize(doc["content"])
    return {
        "title": doc["title"],
        "topic": doc["seed_topic"],
        "raw_word_count": raw_words,
        "clean_token_count": len(tokens),
        "unique_tokens": len(set(tokens)),
        "top_keywords": top_keywords(doc["content"], 8),
    }
