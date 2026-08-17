# End-to-End Information Retrieval System
### BITS Pilani WILP — DSECLZG537 (Information Retrieval), Assignment 2

A single Streamlit application that demonstrates the complete IR lifecycle:

```
Crawl  ->  Preprocess / Text-mine  ->  Index  ->  Search + Rank  ->  Recommend  ->  Evaluate
```

**Use case:** a technical knowledge-base search engine for a data / analytics
engineering team, built over Wikipedia articles about Information Retrieval,
Machine Learning, Natural Language Processing and Big Data.

---

## 1. Install

You need Python 3.9 or newer.

```bash
# (recommended) create a clean virtual environment
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 2. Build the offline sample dataset (once)

This creates `data/sample_corpus.json`, which lets the app run **without internet**
(useful on exam machines). It is already included, but you can regenerate it:

```bash
python build_sample_corpus.py
```

## 3. Run the app

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Everything is driven from the
sidebar and the buttons on each page.

---

## 4. How to demo it (suggested order)

1. **Sidebar → "Load bundled sample corpus (offline)"** — instant data, no internet.
   *(Or go to the **Crawling** page and run a live Wikipedia crawl — needs internet.)*
2. **Crawling** — see the seeds, depth and the collected-documents table.
3. **Text Mining** — pick a document, view keywords, raw-vs-clean stats, classes.
4. **Index Management** — inverted-index lookup and index statistics.
5. **Search** — run a query; adjust the *alpha* slider (text vs authority).
6. **Ranking Visualization** — watch results re-order as authority weight changes.
7. **Recommendations** — Content-based / Collaborative / Hybrid Top-K.
8. **Evaluation** — Precision, Recall, F1, P@K, R@K, MAP, MRR, NDCG + strategy table.
9. **Performance Analytics** — corpus composition and operation timings.

---

## 5. Seeds

The crawl starts from **seed articles** (editable on the Crawling page). Defaults:

- Information retrieval
- Machine learning
- Natural language processing
- Big data

Each seed also acts as a **topic label**, used as ground truth for evaluation.

---

## 6. Project structure

```
ir_assignment/
├── app.py                     # Main Streamlit app (all interfaces)
├── build_sample_corpus.py     # Generates the offline sample dataset
├── requirements.txt
├── README.md
├── modules/
│   ├── crawler.py             # Wikipedia crawl: depth, seeds, duplicate handling
│   ├── preprocess.py          # Cleaning, tokenising, keyword extraction
│   ├── indexer.py             # Inverted index + TF-IDF
│   ├── search.py              # TF-IDF ranking + PageRank / HITS
│   ├── recommender.py         # Content / collaborative / hybrid
│   └── evaluate.py            # Precision, Recall, F1, P@K, R@K, MAP, MRR, NDCG
└── data/
    ├── sample_corpus.json     # Offline fallback corpus
    ├── documents.json         # (generated) contents, stored SEPARATELY
    ├── metadata.csv           # (generated) metadata, stored SEPARATELY
    └── crawled_corpus.json    # (generated) last live crawl
```

## 7. Notes

- **Internet:** only the live crawl needs it. The bundled sample corpus makes the
  entire app runnable offline.
- **Metadata is stored separately from content** (`metadata.csv` vs
  `documents.json`), as required.
- **Duplicate URLs and duplicate documents** are both handled by the crawler
  (a visited-set for URLs, an MD5 content hash for documents).
