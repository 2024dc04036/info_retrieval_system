"""
search.py
---------
The retrieval + ranking brain of the system.

Two ideas are combined here:

  1. CONTENT relevance (TF-IDF + cosine similarity)
     How well does the document's text match the query words?

  2. AUTHORITY / importance (PageRank or HITS on the link graph)
     Independently of the query, how "central" is this document in the
     crawled link network? Well-connected pages are usually more authoritative.

Final score = a * content_score + (1 - a) * link_score
The slider for 'a' (alpha) in the UI lets us SHOW why ranking matters: the same
result set can be re-ordered dramatically by changing how much authority counts.
"""

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from modules.preprocess import token_string


# ---------------------------------------------------------------------------
# 1. Link-analysis scores (PageRank & HITS)
# ---------------------------------------------------------------------------
def build_graph(docs, edges):
    """Build a directed graph of documents from the crawled internal edges."""
    g = nx.DiGraph()
    for d in docs:
        g.add_node(d["title"])
    for a, b in edges:
        g.add_edge(a, b)
    return g


def pagerank_scores(docs, edges):
    """
    Return {doc_id: pagerank_score}. Scores are normalised to 0..1 for display.

    PageRank models a 'random surfer' clicking links; pages that many links
    point to accumulate a higher score.
    """
    g = build_graph(docs, edges)
    if g.number_of_edges() == 0:
        # No links -> everyone is equally (un)important.
        return {d["id"]: 0.0 for d in docs}
    pr = nx.pagerank(g)
    title_to_id = {d["title"]: d["id"] for d in docs}
    raw = {title_to_id[t]: pr.get(t, 0.0) for t in title_to_id}
    return _normalise(raw)


def hits_scores(docs, edges):
    """
    Return {doc_id: authority_score} using the HITS algorithm.

    HITS produces two numbers per page: 'hub' (points to good pages) and
    'authority' (pointed to by good hubs). We use the authority score for ranking.
    """
    g = build_graph(docs, edges)
    if g.number_of_edges() == 0:
        return {d["id"]: 0.0 for d in docs}
    try:
        hubs, authorities = nx.hits(g, max_iter=500)
    except Exception:
        return {d["id"]: 0.0 for d in docs}
    title_to_id = {d["title"]: d["id"] for d in docs}
    raw = {title_to_id[t]: authorities.get(t, 0.0) for t in title_to_id}
    return _normalise(raw)


def _normalise(score_dict):
    """Scale a dict of scores into the 0..1 range (so we can blend them fairly)."""
    if not score_dict:
        return {}
    values = list(score_dict.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


# ---------------------------------------------------------------------------
# 2. The main search function
# ---------------------------------------------------------------------------
def search(query, docs, vectorizer, matrix, link_scores, alpha=0.7, top_k=10):
    """
    Run a query and return a ranked list of results.

    Parameters
    ----------
    query : str            -> the user's search text
    docs : list[dict]      -> the corpus
    vectorizer, matrix     -> the fitted TF-IDF model + document matrix
    link_scores : dict     -> {doc_id: score} from PageRank or HITS
    alpha : float (0..1)   -> weight on CONTENT vs LINK authority
    top_k : int            -> how many results to return

    Returns a list of result dicts, already sorted best-first.
    """
    # Transform the query the SAME way documents were transformed.
    q_clean = token_string(query)
    q_vec = vectorizer.transform([q_clean])

    # Cosine similarity between the query and every document.
    content_sims = cosine_similarity(q_vec, matrix).flatten()   # array of length n_docs

    results = []
    for doc in docs:
        i = doc["id"]
        c = float(content_sims[i])
        l = float(link_scores.get(i, 0.0))
        final = alpha * c + (1 - alpha) * l
        results.append({
            "id": i,
            "title": doc["title"],
            "url": doc["url"],
            "topic": doc["seed_topic"],
            "content_score": round(c, 4),
            "link_score": round(l, 4),
            "final_score": round(final, 4),
            "snippet": doc["content"][:200].replace("\n", " ") + "...",
        })

    # Keep only documents that match the query text at all (content_score > 0),
    # then sort by the blended final score.
    results = [r for r in results if r["content_score"] > 0]
    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:top_k]


def expand_query(query, synonyms=None):
    """
    A very small 'query optimisation' helper: expand the query with synonyms.

    This demonstrates query expansion without needing a big thesaurus. The UI
    lets the user toggle it on/off to see the effect on recall.
    """
    if synonyms is None:
        synonyms = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "ir": "information retrieval",
            "nlp": "natural language processing",
            "db": "database",
        }
    words = query.lower().split()
    extra = [synonyms[w] for w in words if w in synonyms]
    return query + " " + " ".join(extra) if extra else query
