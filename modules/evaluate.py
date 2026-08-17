"""
evaluate.py
-----------
Measures how good the retrieval system is, using standard IR metrics:

  Precision, Recall, F1, Precision@K, Recall@K, MAP, MRR, NDCG.

Where does the "ground truth" come from?
  Every crawled document carries the SEED TOPIC it was reached from
  (e.g. "Machine learning"). We build one test query per topic, and declare a
  document RELEVANT to that query if its seed topic matches. This gives us a
  free, automatic set of relevance judgements to score against — no manual
  labelling needed, which keeps the assignment self-contained.
"""

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metrics for a single query
# ---------------------------------------------------------------------------
def precision_recall_f1(retrieved_ids, relevant_ids):
    """Set-based precision, recall and F1 over the whole retrieved list."""
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)
    if not retrieved:
        return 0.0, 0.0, 0.0
    hits = len(retrieved & relevant)
    precision = hits / len(retrieved)
    recall = hits / len(relevant) if relevant else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of the top-k results that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of all relevant documents that appear in the top-k."""
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant)
    return hits / len(relevant)


def average_precision(retrieved_ids, relevant_ids):
    """
    Average Precision for one query (the 'AP' that MAP averages).

    We walk down the ranked list; every time we hit a relevant document we
    record the precision at that point, then average those values.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, d in enumerate(retrieved_ids, start=1):
        if d in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """1 / (rank of the first relevant document). 0 if none are retrieved."""
    relevant = set(relevant_ids)
    for i, d in enumerate(retrieved_ids, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """
    Normalised Discounted Cumulative Gain at rank k (binary relevance).

    DCG rewards putting relevant documents near the top (gains are discounted
    by log2 of their position). IDCG is the best possible DCG. NDCG = DCG / IDCG.
    """
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]

    dcg = 0.0
    for i, d in enumerate(top_k, start=1):
        rel = 1 if d in relevant else 0
        dcg += rel / math.log2(i + 1)

    # Ideal DCG: as many relevant docs as possible, all at the top.
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Full-system evaluation over several test queries
# ---------------------------------------------------------------------------
def build_test_queries(docs):
    """
    One query per seed topic. Relevant docs = docs sharing that topic.

    Returns a list of (query_text, relevant_id_set) pairs.
    """
    topics = sorted({d["seed_topic"] for d in docs})
    queries = []
    for t in topics:
        relevant = {d["id"] for d in docs if d["seed_topic"] == t}
        queries.append((t, relevant))
    return queries


def evaluate_system(docs, run_search_fn, k=5):
    """
    Run every test query through the provided search function and score it.

    Parameters
    ----------
    docs : the corpus
    run_search_fn : function(query_text) -> ordered list of doc ids
        The Streamlit app passes a small wrapper around search.search() here.
    k : cut-off used for the @K metrics.

    Returns
    -------
    per_query_df : pandas DataFrame, one row per query
    summary : dict of averaged metrics (this is where MAP and MRR live)
    """
    rows = []
    ap_list, rr_list = [], []

    for query_text, relevant in build_test_queries(docs):
        retrieved = run_search_fn(query_text)          # ordered doc ids

        p, r, f1 = precision_recall_f1(retrieved, relevant)
        pk = precision_at_k(retrieved, relevant, k)
        rk = recall_at_k(retrieved, relevant, k)
        ap = average_precision(retrieved, relevant)
        rr = reciprocal_rank(retrieved, relevant)
        ndcg = ndcg_at_k(retrieved, relevant, k)

        ap_list.append(ap)
        rr_list.append(rr)

        rows.append({
            "Query": query_text,
            "#Relevant": len(relevant),
            "#Retrieved": len(retrieved),
            "Precision": round(p, 3),
            "Recall": round(r, 3),
            "F1": round(f1, 3),
            "P@K": round(pk, 3),
            "R@K": round(rk, 3),
            "AP": round(ap, 3),
            "RR": round(rr, 3),
            "NDCG@K": round(ndcg, 3),
        })

    per_query_df = pd.DataFrame(rows)
    summary = {
        "MAP": round(float(np.mean(ap_list)) if ap_list else 0.0, 3),
        "MRR": round(float(np.mean(rr_list)) if rr_list else 0.0, 3),
        "Mean Precision": round(float(per_query_df["Precision"].mean()), 3),
        "Mean Recall": round(float(per_query_df["Recall"].mean()), 3),
        "Mean F1": round(float(per_query_df["F1"].mean()), 3),
        "Mean NDCG@K": round(float(per_query_df["NDCG@K"].mean()), 3),
        "K": k,
    }
    return per_query_df, summary
