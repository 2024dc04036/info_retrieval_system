"""
recommender.py
--------------
Three flavours of recommender, as required by the assignment:

  1. CONTENT-BASED
     "Show me documents whose TEXT is similar to this one."
     Uses TF-IDF cosine similarity between documents.

  2. COLLABORATIVE
     "People who liked this document also liked ..."
     Uses a user-item ratings matrix. Because we have no real user logs, we
     generate a small SYNTHETIC ratings matrix (clearly labelled as synthetic in
     the report). This is standard practice for demonstrating the technique.

  3. HYBRID
     A simple weighted blend of the two scores above.

Each function returns Top-K (document, similarity_score) pairs.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. Content-based
# ---------------------------------------------------------------------------
def content_based(doc_id, docs, matrix, top_k=5):
    """Recommend documents most textually similar to `doc_id`."""
    sims = cosine_similarity(matrix[doc_id], matrix).flatten()
    order = np.argsort(sims)[::-1]                 # highest similarity first
    recs = []
    for j in order:
        if j == doc_id:                            # skip the document itself
            continue
        recs.append({
            "id": int(j),
            "title": docs[j]["title"],
            "topic": docs[j]["seed_topic"],
            "score": round(float(sims[j]), 4),
        })
        if len(recs) >= top_k:
            break
    return recs


# ---------------------------------------------------------------------------
# 2. Collaborative (synthetic user-item matrix)
# ---------------------------------------------------------------------------
def make_synthetic_ratings(docs, n_users=15, seed=42):
    """
    Create a fake but realistic user-item ratings matrix.

    Logic: each user has a couple of favourite TOPICS and rates documents from
    those topics highly (4-5), other documents low or not at all (0). This gives
    collaborative filtering a real signal to find.

    Returns a (n_users x n_docs) NumPy array of ratings (0 = not rated).
    """
    rng = np.random.default_rng(seed)
    topics = sorted({d["seed_topic"] for d in docs})
    ratings = np.zeros((n_users, len(docs)))
    for u in range(n_users):
        # each user likes 1-2 topics
        liked = rng.choice(topics, size=min(2, len(topics)), replace=False)
        for d in docs:
            if d["seed_topic"] in liked:
                if rng.random() < 0.7:                 # rated 70% of liked-topic docs
                    ratings[u, d["id"]] = rng.integers(4, 6)   # 4 or 5
            else:
                if rng.random() < 0.2:                 # occasionally rate others low
                    ratings[u, d["id"]] = rng.integers(1, 3)   # 1 or 2
    return ratings


def collaborative(doc_id, docs, ratings, top_k=5):
    """
    Item-based collaborative filtering.

    We treat each COLUMN of the ratings matrix as a document's "rating vector"
    across users, then recommend documents whose rating vectors are most similar
    to the target document's vector (i.e. liked by the same users).
    """
    item_matrix = ratings.T                            # shape: (n_docs x n_users)
    sims = cosine_similarity(item_matrix[doc_id].reshape(1, -1), item_matrix).flatten()
    order = np.argsort(sims)[::-1]
    recs = []
    for j in order:
        if j == doc_id:
            continue
        recs.append({
            "id": int(j),
            "title": docs[j]["title"],
            "topic": docs[j]["seed_topic"],
            "score": round(float(sims[j]), 4),
        })
        if len(recs) >= top_k:
            break
    return recs


# ---------------------------------------------------------------------------
# 3. Hybrid
# ---------------------------------------------------------------------------
def hybrid(doc_id, docs, matrix, ratings, beta=0.5, top_k=5):
    """
    Blend content and collaborative scores: beta * content + (1-beta) * collab.

    beta = 1.0 -> pure content-based
    beta = 0.0 -> pure collaborative
    """
    content_sims = cosine_similarity(matrix[doc_id], matrix).flatten()
    item_matrix = ratings.T
    collab_sims = cosine_similarity(
        item_matrix[doc_id].reshape(1, -1), item_matrix
    ).flatten()

    combined = beta * content_sims + (1 - beta) * collab_sims
    order = np.argsort(combined)[::-1]
    recs = []
    for j in order:
        if j == doc_id:
            continue
        recs.append({
            "id": int(j),
            "title": docs[j]["title"],
            "topic": docs[j]["seed_topic"],
            "score": round(float(combined[j]), 4),
            "content_part": round(float(content_sims[j]), 4),
            "collab_part": round(float(collab_sims[j]), 4),
        })
        if len(recs) >= top_k:
            break
    return recs
