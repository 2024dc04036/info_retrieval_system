"""
crawler.py
----------
A small, beginner-friendly web crawler for Wikipedia.

What this module does (mapping to the assignment requirements):
  * Acquires information from a heterogeneous public source (Wikipedia REST API).
  * Supports CONFIGURABLE crawling DEPTH and MULTIPLE SEED sources.
  * Handles DUPLICATE URLs (a page is never visited twice).
  * Handles DUPLICATE DOCUMENTS (identical text is detected using an MD5 hash).
  * Records the link graph between pages so we can run PageRank / HITS later.

How the crawl works (Breadth-First Search / BFS):
  1. Start from the "seed" articles the user provides.
  2. For each page, download its clean text and the titles it links to.
  3. Add a limited number of those linked titles to the queue (next level).
  4. Repeat until we reach the requested depth or the max number of documents.

Note on offline use:
  The Anthropic sandbox and some exam machines have no internet access.
  If a network call fails, the caller (the Streamlit app) can instead load the
  bundled sample corpus in data/sample_corpus.json, so the app always runs.
"""

import hashlib
import time
from collections import deque

import requests

# Wikipedia "action" API endpoint (public, no API key required).
WIKI_API = "https://en.wikipedia.org/w/api.php"

# A descriptive User-Agent is good manners and is requested by Wikipedia.
HEADERS = {"User-Agent": "BITS-IR-Assignment/1.0 (educational use)"}


def _content_hash(text):
    """Return an MD5 fingerprint of the cleaned text (used for duplicate docs)."""
    normalized = " ".join(text.lower().split())          # collapse whitespace
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def fetch_page(title):
    """
    Download one Wikipedia page.

    Returns a dictionary with the plain-text extract, the outgoing link titles,
    and the page categories. Returns None if the page cannot be fetched.
    """
    # --- 1. Get the plain-text extract of the page ---------------------------
    params_text = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,        # plain text, not HTML
        "titles": title,
        "format": "json",
        "redirects": 1,          # follow redirects (helps avoid duplicates)
    }
    r = requests.get(WIKI_API, params=params_text, headers=HEADERS, timeout=20)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page or "extract" not in page:
        return None
    text = page["extract"].strip()
    real_title = page["title"]                       # canonical title after redirects
    if len(text) < 200:                              # skip near-empty stubs
        return None

    # --- 2. Get the outgoing links (only to real articles, namespace 0) ------
    params_links = {
        "action": "query",
        "prop": "links",
        "titles": real_title,
        "plnamespace": 0,        # 0 = normal articles only (skip File:, Help:, etc.)
        "pllimit": "max",
        "format": "json",
    }
    r2 = requests.get(WIKI_API, params=params_links, headers=HEADERS, timeout=20)
    r2.raise_for_status()
    page2 = next(iter(r2.json()["query"]["pages"].values()))
    links = [l["title"] for l in page2.get("links", [])]

    # --- 3. Get a few categories (used as extra metadata) --------------------
    params_cat = {
        "action": "query",
        "prop": "categories",
        "titles": real_title,
        "cllimit": 10,
        "format": "json",
    }
    r3 = requests.get(WIKI_API, params=params_cat, headers=HEADERS, timeout=20)
    r3.raise_for_status()
    page3 = next(iter(r3.json()["query"]["pages"].values()))
    cats = [c["title"].replace("Category:", "") for c in page3.get("categories", [])]

    return {
        "title": real_title,
        "url": "https://en.wikipedia.org/wiki/" + real_title.replace(" ", "_"),
        "content": text,
        "links": links,
        "categories": cats,
    }


def crawl(seeds, depth=1, max_links_per_page=6, max_docs=40, polite_delay=0.2):
    """
    Breadth-first crawl starting from the given seed titles.

    Parameters
    ----------
    seeds : list[str]
        Seed article titles, e.g. ["Information retrieval", "Machine learning"].
        The seed a document was reached from becomes its "topic" label, which we
        later use as ground truth for evaluation.
    depth : int
        How many link-hops away from the seeds we are allowed to travel.
        depth=0 -> only the seeds themselves.
        depth=1 -> seeds + their direct neighbours, and so on.
    max_links_per_page : int
        Branching factor: how many links we follow from each page (keeps the
        corpus a sensible size instead of exploding into thousands of pages).
    max_docs : int
        Hard upper limit on the number of documents collected.
    polite_delay : float
        Seconds to wait between requests, to be gentle on the Wikipedia servers.

    Returns
    -------
    docs  : list[dict]  -> one dict per unique document
    edges : list[tuple] -> (source_title, target_title) links between crawled docs
    log   : list[str]   -> human-readable crawl log for the Streamlit UI
    """
    visited_titles = set()      # duplicate-URL guard
    seen_hashes = set()         # duplicate-document guard
    docs = []
    edges = []
    log = []

    # The queue holds (title, current_depth, seed_topic) triples.
    queue = deque()
    for s in seeds:
        queue.append((s, 0, s))

    doc_id = 0
    while queue and len(docs) < max_docs:
        title, d, topic = queue.popleft()

        # --- duplicate-URL check --------------------------------------------
        key = title.lower().strip()
        if key in visited_titles:
            continue
        visited_titles.add(key)

        # --- download the page ----------------------------------------------
        try:
            page = fetch_page(title)
        except Exception as e:                       # network / API error
            log.append("SKIP  (error) " + title + " -> " + str(e))
            continue
        if page is None:
            log.append("SKIP  (empty/missing) " + title)
            continue

        # --- duplicate-document check ---------------------------------------
        h = _content_hash(page["content"])
        if h in seen_hashes:
            log.append("SKIP  (duplicate content) " + page["title"])
            continue
        seen_hashes.add(h)

        # --- store the document ---------------------------------------------
        docs.append({
            "id": doc_id,
            "title": page["title"],
            "url": page["url"],
            "content": page["content"],
            "seed_topic": topic,
            "categories": page["categories"],
            "content_hash": h,
            "depth": d,
        })
        log.append("KEEP  depth=" + str(d) + " topic=" + topic + " :: " + page["title"])
        doc_id += 1

        # --- record edges + enqueue neighbours (if we can go deeper) --------
        if d < depth:
            for neighbour in page["links"][:max_links_per_page]:
                edges.append((page["title"], neighbour))
                if neighbour.lower().strip() not in visited_titles:
                    queue.append((neighbour, d + 1, topic))

        time.sleep(polite_delay)

    return docs, edges, log


# Titles that link between already-crawled documents are the only useful edges
# for PageRank/HITS. This helper keeps only those edges.
def filter_internal_edges(docs, edges):
    """Keep only edges whose BOTH endpoints are documents we actually crawled."""
    titles = {d["title"] for d in docs}
    return [(a, b) for (a, b) in edges if a in titles and b in titles]
