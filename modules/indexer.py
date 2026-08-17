"""
indexer.py
----------
Builds the two data structures a search engine needs:

  1. An INVERTED INDEX: term -> list of document ids that contain the term.
     This is the classic IR structure that lets us find candidate documents fast.

  2. A TF-IDF MATRIX (via scikit-learn): a numeric vector per document.
     This lets us RANK documents by cosine similarity to a query.

We also expose helpers for "index management" in the UI: index statistics and a
peek at the postings list for any term.
"""

from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

from modules.preprocess import tokenize, token_string


def build_inverted_index(docs):
    """
    Build a dictionary: term -> sorted list of document ids containing it.

    Example: {"kafka": [3, 7, 12], "index": [0, 3, 4, ...]}
    """
    index = defaultdict(set)
    for doc in docs:
        for term in set(tokenize(doc["content"])):     # set() -> one entry per doc
            index[term].add(doc["id"])
    # convert sets to sorted lists so the output is stable and printable
    return {term: sorted(ids) for term, ids in index.items()}


def build_tfidf(docs):
    """
    Build a TF-IDF matrix over all documents.

    Returns
    -------
    vectorizer : the fitted TfidfVectorizer (needed to transform queries later)
    matrix     : sparse matrix of shape (n_docs, n_terms)
    """
    # We pre-clean each document with our own tokenizer, then let scikit-learn
    # compute TF-IDF weights. This keeps preprocessing consistent everywhere.
    corpus = [token_string(d["content"]) for d in docs]
    vectorizer = TfidfVectorizer(
        min_df=1,            # keep a term if it appears in at least 1 document
        max_df=0.9,          # drop terms that appear in >90% of docs (too common)
        sublinear_tf=True,   # use 1 + log(tf), the standard "l" weighting
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def index_stats(docs, inverted_index):
    """Return a dictionary of headline numbers for the 'Index Management' page."""
    total_tokens = sum(len(tokenize(d["content"])) for d in docs)
    return {
        "documents": len(docs),
        "unique_terms": len(inverted_index),
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": round(total_tokens / max(len(docs), 1), 1),
    }


def postings(inverted_index, term):
    """Return the list of document ids for a term (its 'postings list')."""
    return inverted_index.get(term.lower().strip(), [])
