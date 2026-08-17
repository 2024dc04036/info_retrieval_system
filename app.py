"""
app.py  -  End-to-End Information Retrieval System (BITS WILP - DSECLZG537, Assignment 2)
========================================================================================

A single, cohesive Streamlit application that demonstrates the full IR lifecycle:

    Crawl  ->  Preprocess/Mine  ->  Index  ->  Search + Rank  ->  Recommend  ->  Evaluate

Use case: a technical knowledge-base search engine for a data / analytics engineering
team, built over Wikipedia articles on IR, machine learning, NLP and big data.

Run it with:
    streamlit run app.py

Everything below is driven from the sidebar and the buttons on each page, so the
complete workflow is executable through the Streamlit front end only.
"""

import json
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from modules import preprocess, indexer, search, recommender, evaluate
from modules.crawler import crawl, filter_internal_edges

# ---------------------------------------------------------------------------
# Paths & default configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data"
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_corpus.json")     # offline fallback
CRAWL_PATH = os.path.join(DATA_DIR, "crawled_corpus.json")     # saved live crawl
META_PATH = os.path.join(DATA_DIR, "metadata.csv")            # metadata (kept SEPARATE)
DOCS_PATH = os.path.join(DATA_DIR, "documents.json")          # contents (kept SEPARATE)

# The SEED articles the crawler starts from. Each seed is also a "topic" label
# used as ground truth during evaluation. Edit these in the Crawling page.
DEFAULT_SEEDS = [
    "Information retrieval",
    "Machine learning",
    "Natural language processing",
    "Big data",
]

st.set_page_config(page_title="IR System - BITS Assignment 2", layout="wide")


# ---------------------------------------------------------------------------
# Core helpers: (re)build all derived structures from a corpus
# ---------------------------------------------------------------------------
def rebuild_all(docs, edges):
    """Given docs + edges, build index, TF-IDF, link scores, ratings; store in session."""
    edges = filter_internal_edges(docs, edges)

    t0 = time.time()
    inv = indexer.build_inverted_index(docs)
    vec, mat = indexer.build_tfidf(docs)
    index_time = time.time() - t0

    pr = search.pagerank_scores(docs, edges)
    hits = search.hits_scores(docs, edges)
    ratings = recommender.make_synthetic_ratings(docs)

    st.session_state.docs = docs
    st.session_state.edges = edges
    st.session_state.inv_index = inv
    st.session_state.vectorizer = vec
    st.session_state.matrix = mat
    st.session_state.pagerank = pr
    st.session_state.hits = hits
    st.session_state.ratings = ratings
    st.session_state.timings["index_build_s"] = round(index_time, 3)


def save_corpus_split(docs, edges):
    """
    Persist the corpus, keeping METADATA SEPARATE FROM CONTENT (a requirement).

      * documents.json -> {id: full_text}      (the raw content)
      * metadata.csv   -> id,title,url,topic,length,hash,depth  (the metadata)
      * crawled_corpus.json -> everything, so the app can reload in one shot.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    contents = {str(d["id"]): d["content"] for d in docs}
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(contents, f)

    meta_rows = [{
        "id": d["id"], "title": d["title"], "url": d["url"],
        "seed_topic": d["seed_topic"], "length_chars": len(d["content"]),
        "content_hash": d["content_hash"], "depth": d["depth"],
    } for d in docs]
    pd.DataFrame(meta_rows).to_csv(META_PATH, index=False)

    with open(CRAWL_PATH, "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "edges": edges}, f)


def load_corpus(path):
    data = json.load(open(path, encoding="utf-8"))
    return data["docs"], data["edges"]


def corpus_ready():
    return "docs" in st.session_state and st.session_state.docs


# Initialise session containers once.
if "timings" not in st.session_state:
    st.session_state.timings = {}
if "crawl_log" not in st.session_state:
    st.session_state.crawl_log = []


# ---------------------------------------------------------------------------
# Sidebar: data source + navigation
# ---------------------------------------------------------------------------
st.sidebar.title("IR System")
st.sidebar.caption("BITS WILP - DSECLZG537 - Assignment 2")

st.sidebar.markdown("### 1. Load data")
if st.sidebar.button("Load bundled sample corpus (offline)"):
    docs, edges = load_corpus(SAMPLE_PATH)
    rebuild_all(docs, edges)
    save_corpus_split(docs, edges)
    st.sidebar.success(f"Loaded sample: {len(docs)} documents.")

if os.path.exists(CRAWL_PATH) and st.sidebar.button("Reload last crawl"):
    docs, edges = load_corpus(CRAWL_PATH)
    rebuild_all(docs, edges)
    st.sidebar.success(f"Reloaded crawl: {len(docs)} documents.")

st.sidebar.markdown("### 2. Go to")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Crawling", "Text Mining", "Index Management",
     "Search", "Ranking Visualization", "Recommendations",
     "Evaluation", "Performance Analytics"],
    label_visibility="collapsed",
)

if corpus_ready():
    st.sidebar.info(f"Corpus in memory: {len(st.session_state.docs)} docs")
else:
    st.sidebar.warning("No corpus loaded yet. Load the sample or run a crawl.")


# ===========================================================================
# PAGE: Dashboard
# ===========================================================================
if page == "Dashboard":
    st.title("End-to-End Information Retrieval System")
    st.markdown(
        "This application walks through the **complete IR lifecycle** on a "
        "knowledge base of technical articles. Use the sidebar to move between "
        "stages. A typical demo order is shown below."
    )

    steps = [
        ("1. Crawling", "Collect articles from Wikipedia seeds (or load the sample)."),
        ("2. Text Mining", "Clean text, extract keywords, profile & classify documents."),
        ("3. Index Management", "Build the inverted index and TF-IDF matrix."),
        ("4. Search", "Run queries with ranked retrieval and query expansion."),
        ("5. Ranking Visualization", "See how PageRank / HITS re-order results."),
        ("6. Recommendations", "Content-based, collaborative and hybrid Top-K."),
        ("7. Evaluation", "Precision, Recall, F1, P@K, R@K, MAP, MRR, NDCG."),
        ("8. Performance Analytics", "Corpus stats and operation timings."),
    ]
    c1, c2 = st.columns(2)
    for i, (title, desc) in enumerate(steps):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f"**{title}**  \n{desc}")

    st.divider()
    if corpus_ready():
        docs = st.session_state.docs
        stats = indexer.index_stats(docs, st.session_state.inv_index)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Documents", stats["documents"])
        m2.metric("Unique terms", stats["unique_terms"])
        m3.metric("Total tokens", stats["total_tokens"])
        m4.metric("Topics (seeds)", len({d["seed_topic"] for d in docs}))
    else:
        st.info("Load the bundled sample corpus from the sidebar to begin.")


# ===========================================================================
# PAGE: Crawling
# ===========================================================================
elif page == "Crawling":
    st.title("Crawling Interface")
    st.markdown(
        "The crawler starts from one or more **seed articles** and follows their "
        "links using breadth-first search. Each seed also acts as a **topic label** "
        "used later for evaluation."
    )

    st.subheader("Seed configuration")
    st.caption(
        "SEEDS are the starting points of the crawl. Put one article title per line. "
        "The defaults below build a data / analytics engineering knowledge base."
    )
    seed_text = st.text_area("Seed article titles (one per line)",
                             value="\n".join(DEFAULT_SEEDS), height=140)
    seeds = [s.strip() for s in seed_text.splitlines() if s.strip()]

    col1, col2, col3 = st.columns(3)
    depth = col1.slider("Crawl depth (link hops)", 0, 2, 1,
                        help="0 = only seeds, 1 = seeds + neighbours, 2 = two hops out")
    branch = col2.slider("Links followed per page", 2, 12, 5,
                         help="Branching factor - keeps the corpus a sensible size")
    max_docs = col3.slider("Maximum documents", 10, 80, 40)

    st.write("**Seeds to be used:**", ", ".join(seeds) if seeds else "(none)")

    if st.button("Start crawl", type="primary"):
        if not seeds:
            st.error("Please provide at least one seed article title.")
        else:
            with st.spinner("Crawling Wikipedia... (needs internet access)"):
                t0 = time.time()
                try:
                    docs, edges, log = crawl(seeds, depth=depth,
                                             max_links_per_page=branch,
                                             max_docs=max_docs)
                    st.session_state.timings["crawl_s"] = round(time.time() - t0, 2)
                    if not docs:
                        st.error("Crawl returned no documents. Try different seeds.")
                    else:
                        rebuild_all(docs, edges)
                        save_corpus_split(docs, edges)
                        st.session_state.crawl_log = log
                        st.success(
                            f"Crawled {len(docs)} unique documents in "
                            f"{st.session_state.timings['crawl_s']}s. Metadata and "
                            "contents saved separately in the data/ folder."
                        )
                except Exception as e:
                    st.error(f"Crawl failed (likely no internet): {e}")
                    st.info("Tip: use 'Load bundled sample corpus' in the sidebar "
                            "to run the rest of the app offline.")

    if st.session_state.crawl_log:
        with st.expander("Crawl log (duplicate handling shown here)"):
            st.text("\n".join(st.session_state.crawl_log))

    if corpus_ready():
        st.subheader("Collected documents")
        df = pd.DataFrame([{
            "id": d["id"], "title": d["title"], "topic": d["seed_topic"],
            "depth": d["depth"], "chars": len(d["content"]),
        } for d in st.session_state.docs])
        st.dataframe(df, height=300)


# ===========================================================================
# PAGE: Text Mining
# ===========================================================================
elif page == "Text Mining":
    st.title("Text Preprocessing & Mining")
    if not corpus_ready():
        st.warning("Load a corpus first (sidebar).")
    else:
        docs = st.session_state.docs
        st.markdown(
            "Raw text is cleaned, tokenised and stop-word filtered. Below you can "
            "inspect any document's profile and see corpus-wide feature statistics."
        )

        # --- single document profile ---
        st.subheader("Document profiling & keyword extraction")
        titles = [f"{d['id']} - {d['title']}" for d in docs]
        pick = st.selectbox("Choose a document", titles)
        chosen = docs[int(pick.split(" - ")[0])]
        prof = preprocess.document_profile(chosen)

        c1, c2, c3 = st.columns(3)
        c1.metric("Raw words", prof["raw_word_count"])
        c2.metric("Clean tokens", prof["clean_token_count"])
        c3.metric("Unique tokens", prof["unique_tokens"])

        kw = pd.DataFrame(prof["top_keywords"], columns=["keyword", "count"])
        st.write("**Top keywords (simple frequency-based extraction):**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(kw["keyword"][::-1], kw["count"][::-1])
        ax.set_xlabel("frequency")
        st.pyplot(fig)

        # --- corpus-level comparison of preprocessing effect ---
        st.subheader("Comparative analysis: raw vs cleaned")
        rows = []
        for d in docs:
            p = preprocess.document_profile(d)
            rows.append({"title": d["title"], "topic": d["seed_topic"],
                         "raw_words": p["raw_word_count"],
                         "clean_tokens": p["clean_token_count"]})
        comp = pd.DataFrame(rows)
        st.dataframe(comp, height=260)
        st.caption(
            f"Preprocessing removed on average "
            f"{round((1 - comp['clean_tokens'].sum() / comp['raw_words'].sum()) * 100)}% "
            "of tokens (stop words, punctuation, short words)."
        )

        # --- simple document classification by topic ---
        st.subheader("Document classification (by seed topic)")
        st.caption(
            "A Naive-Bayes-free demonstration: we group documents by their seed "
            "topic and show the class distribution. This is the label set a "
            "classifier would learn to predict."
        )
        dist = comp["topic"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(dist.index, dist.values)
        ax2.set_ylabel("documents")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig2)


# ===========================================================================
# PAGE: Index Management
# ===========================================================================
elif page == "Index Management":
    st.title("Index Management")
    if not corpus_ready():
        st.warning("Load a corpus first (sidebar).")
    else:
        docs = st.session_state.docs
        inv = st.session_state.inv_index
        stats = indexer.index_stats(docs, inv)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", stats["documents"])
        c2.metric("Unique terms", stats["unique_terms"])
        c3.metric("Total tokens", stats["total_tokens"])
        c4.metric("Avg tokens/doc", stats["avg_tokens_per_doc"])

        st.subheader("Inspect the inverted index")
        st.caption("Type a term to see its postings list (the ids of documents "
                   "that contain it).")
        term = st.text_input("Term", value="index")
        if term:
            ids = indexer.postings(inv, term)
            if ids:
                st.write(f"**'{term.lower()}'** appears in {len(ids)} document(s):")
                st.write(pd.DataFrame([{"id": i, "title": docs[i]["title"]} for i in ids]))
            else:
                st.info(f"'{term}' is not in the index.")

        st.subheader("Most frequent index terms")
        freq = sorted(inv.items(), key=lambda kv: len(kv[1]), reverse=True)[:15]
        fdf = pd.DataFrame([{"term": t, "in_documents": len(ids)} for t, ids in freq])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(fdf["term"][::-1], fdf["in_documents"][::-1])
        ax.set_xlabel("number of documents")
        st.pyplot(fig)

        st.caption("Metadata and document contents are stored separately on disk: "
                   "`data/metadata.csv` and `data/documents.json`.")


# ===========================================================================
# PAGE: Search
# ===========================================================================
elif page == "Search":
    st.title("Search Interface")
    if not corpus_ready():
        st.warning("Load a corpus first (sidebar).")
    else:
        docs = st.session_state.docs
        st.markdown("Enter a query. Results are ranked by a blend of **text "
                    "relevance (TF-IDF)** and **link authority (PageRank/HITS)**.")

        query = st.text_input("Search query", value="how does document ranking work")

        c1, c2, c3 = st.columns(3)
        method = c1.selectbox("Authority signal", ["PageRank", "HITS", "None (text only)"])
        alpha = c2.slider("Text vs authority (alpha)", 0.0, 1.0, 0.7, 0.05,
                          help="1.0 = pure text match, 0.0 = pure link authority")
        top_k = c3.slider("Results to show", 3, 15, 8)
        expand = st.checkbox("Expand query with synonyms (query optimisation)", value=False)

        if st.button("Search", type="primary") and query:
            q = search.expand_query(query) if expand else query
            if expand and q != query:
                st.caption(f"Expanded query: *{q}*")

            if method == "PageRank":
                link = st.session_state.pagerank
            elif method == "HITS":
                link = st.session_state.hits
            else:
                link = {d["id"]: 0.0 for d in docs}

            t0 = time.time()
            results = search.search(q, docs, st.session_state.vectorizer,
                                    st.session_state.matrix, link,
                                    alpha=alpha, top_k=top_k)
            st.session_state.timings["last_search_ms"] = round((time.time() - t0) * 1000, 1)

            if not results:
                st.info("No documents matched. Try different words.")
            else:
                st.caption(f"Retrieved in {st.session_state.timings['last_search_ms']} ms")
                for rank, r in enumerate(results, 1):
                    st.markdown(f"**{rank}. [{r['title']}]({r['url']})**  ·  "
                                f"topic: *{r['topic']}*")
                    st.caption(
                        f"final {r['final_score']:.3f}  =  "
                        f"{alpha:.2f}×text({r['content_score']:.3f}) + "
                        f"{1-alpha:.2f}×link({r['link_score']:.3f})")
                    st.write(r["snippet"])
                    st.divider()


# ===========================================================================
# PAGE: Ranking Visualization
# ===========================================================================
elif page == "Ranking Visualization":
    st.title("Ranking Visualization")
    if not corpus_ready():
        st.warning("Load a corpus first (sidebar).")
    else:
        docs = st.session_state.docs
        st.markdown("This page shows **why ranking matters**: the same query is "
                    "re-ranked as we shift weight from text relevance to link "
                    "authority.")

        query = st.text_input("Query", value="information retrieval and ranking")
        method = st.radio("Authority signal", ["PageRank", "HITS"], horizontal=True)
        link = st.session_state.pagerank if method == "PageRank" else st.session_state.hits

        # Compare alpha = 1.0 (pure text) vs alpha = 0.4 (blended)
        pure = search.search(query, docs, st.session_state.vectorizer,
                             st.session_state.matrix, link, alpha=1.0, top_k=6)
        blended = search.search(query, docs, st.session_state.vectorizer,
                                st.session_state.matrix, link, alpha=0.4, top_k=6)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Pure text ranking (alpha=1.0)")
            st.table(pd.DataFrame([{"rank": i + 1, "title": r["title"],
                                    "score": r["final_score"]}
                                   for i, r in enumerate(pure)]))
        with c2:
            st.subheader(f"Blended with {method} (alpha=0.4)")
            st.table(pd.DataFrame([{"rank": i + 1, "title": r["title"],
                                    "score": r["final_score"]}
                                   for i, r in enumerate(blended)]))

        st.subheader(f"{method} authority scores (top documents)")
        top = sorted(link.items(), key=lambda kv: kv[1], reverse=True)[:10]
        tdf = pd.DataFrame([{"title": docs[i]["title"], "score": round(s, 3)}
                            for i, s in top])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(tdf["title"][::-1], tdf["score"][::-1])
        ax.set_xlabel(f"{method} score (normalised)")
        st.pyplot(fig)

        # Optional: draw the link graph if it is small enough
        if len(docs) <= 30 and st.session_state.edges:
            st.subheader("Link graph")
            g = search.build_graph(docs, st.session_state.edges)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            pos = nx.spring_layout(g, seed=1, k=0.6)
            nx.draw_networkx(g, pos, ax=ax2, node_size=500, font_size=6,
                             node_color="#cfe8ff", edge_color="#999999",
                             arrowsize=8)
            ax2.axis("off")
            st.pyplot(fig2)


# ===========================================================================
# PAGE: Recommendations
# ===========================================================================
elif page == "Recommendations":
    st.title("Recommendation Panel")
    if not corpus_ready():
        st.warning("Load a corpus first (sidebar).")
    else:
        docs = st.session_state.docs
        st.markdown("Pick a document to get **Top-K similar documents** using three "
                    "strategies. Similarity scores are shown for each.")

        titles = [f"{d['id']} - {d['title']}" for d in docs]
        pick = st.selectbox("Base document", titles)
        base_id = int(pick.split(" - ")[0])
        top_k = st.slider("Top-K", 3, 8, 5)

        method = st.radio("Strategy",
                          ["Content-based", "Collaborative", "Hybrid"],
                          horizontal=True)

        if method == "Hybrid":
            beta = st.slider("Content vs collaborative (beta)", 0.0, 1.0, 0.5, 0.1)
            recs = recommender.hybrid(base_id, docs, st.session_state.matrix,
                                      st.session_state.ratings, beta=beta, top_k=top_k)
            st.caption("Collaborative part uses a clearly-labelled SYNTHETIC "
                       "user-item ratings matrix.")
            df = pd.DataFrame([{"title": r["title"], "topic": r["topic"],
                                "score": r["score"], "content_part": r["content_part"],
                                "collab_part": r["collab_part"]} for r in recs])
        elif method == "Collaborative":
            recs = recommender.collaborative(base_id, docs,
                                             st.session_state.ratings, top_k=top_k)
            st.caption("Uses a SYNTHETIC user-item ratings matrix (no real user "
                       "logs are available for a class assignment).")
            df = pd.DataFrame([{"title": r["title"], "topic": r["topic"],
                                "score": r["score"]} for r in recs])
        else:
            recs = recommender.content_based(base_id, docs,
                                             st.session_state.matrix, top_k=top_k)
            df = pd.DataFrame([{"title": r["title"], "topic": r["topic"],
                                "score": r["score"]} for r in recs])

        st.subheader(f"Top-{top_k} recommendations ({method})")
        st.dataframe(df)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(df["title"][::-1], df["score"][::-1])
        ax.set_xlabel("similarity score")
        st.pyplot(fig)


# ===========================================================================
# PAGE: Evaluation
# ===========================================================================
elif page == "Evaluation":
    st.title("Evaluation Dashboard")
    if not corpus_ready():
        st.warning("Load a corpus first (sidebar).")
    else:
        docs = st.session_state.docs
        st.markdown(
            "Each **seed topic** becomes a test query; documents of that topic are "
            "treated as relevant. We then score the ranked results with standard "
            "IR metrics."
        )

        c1, c2 = st.columns(2)
        k = c1.slider("K for @K metrics", 3, 10, 5)
        method = c2.selectbox("Ranking used", ["PageRank", "HITS", "Text only"])
        alpha = st.slider("alpha (text vs authority)", 0.0, 1.0, 0.7, 0.05)

        if method == "PageRank":
            link = st.session_state.pagerank
        elif method == "HITS":
            link = st.session_state.hits
        else:
            link = {d["id"]: 0.0 for d in docs}

        def run_search_fn(q):
            res = search.search(q, docs, st.session_state.vectorizer,
                                st.session_state.matrix, link,
                                alpha=alpha, top_k=len(docs))
            return [r["id"] for r in res]

        per_query, summary = evaluate.evaluate_system(docs, run_search_fn, k=k)

        st.subheader("Per-query metrics")
        st.dataframe(per_query)

        st.subheader("System summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("MAP", summary["MAP"])
        s2.metric("MRR", summary["MRR"])
        s3.metric("Mean F1", summary["Mean F1"])
        s4.metric(f"Mean NDCG@{k}", summary["Mean NDCG@K"])

        st.subheader("Metric comparison across queries")
        plot_df = per_query.set_index("Query")[["Precision", "Recall", "F1", "NDCG@K"]]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        plot_df.plot(kind="bar", ax=ax)
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)

        # Let the user compare text-only vs PageRank vs HITS in one table.
        st.subheader("Ranking-strategy comparison (MAP / MRR / NDCG)")
        comp_rows = []
        for name, ln in [("Text only", {d["id"]: 0.0 for d in docs}),
                         ("PageRank", st.session_state.pagerank),
                         ("HITS", st.session_state.hits)]:
            def fn(q, ln=ln):
                res = search.search(q, docs, st.session_state.vectorizer,
                                    st.session_state.matrix, ln, alpha=alpha,
                                    top_k=len(docs))
                return [r["id"] for r in res]
            _, s = evaluate.evaluate_system(docs, fn, k=k)
            comp_rows.append({"strategy": name, "MAP": s["MAP"], "MRR": s["MRR"],
                              "Mean F1": s["Mean F1"], f"NDCG@{k}": s["Mean NDCG@K"]})
        st.table(pd.DataFrame(comp_rows))


# ===========================================================================
# PAGE: Performance Analytics
# ===========================================================================
elif page == "Performance Analytics":
    st.title("Performance Analytics")
    if not corpus_ready():
        st.warning("Load a corpus first (sidebar).")
    else:
        docs = st.session_state.docs
        st.subheader("Corpus composition")
        topic_counts = pd.Series([d["seed_topic"] for d in docs]).value_counts()
        c1, c2 = st.columns([1, 1])
        with c1:
            st.table(topic_counts.rename("documents"))
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.pie(topic_counts.values, labels=topic_counts.index, autopct="%1.0f%%")
            st.pyplot(fig)

        st.subheader("Document length distribution")
        lengths = [len(preprocess.tokenize(d["content"])) for d in docs]
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.hist(lengths, bins=10)
        ax2.set_xlabel("tokens per document")
        ax2.set_ylabel("count")
        st.pyplot(fig2)

        st.subheader("Operation timings")
        t = st.session_state.timings
        rows = [
            {"operation": "Crawl (last run)", "value": f"{t.get('crawl_s', '—')} s"},
            {"operation": "Index + TF-IDF build", "value": f"{t.get('index_build_s', '—')} s"},
            {"operation": "Last search latency", "value": f"{t.get('last_search_ms', '—')} ms"},
        ]
        st.table(pd.DataFrame(rows))
        st.caption("Timings are captured live as you use the app.")
