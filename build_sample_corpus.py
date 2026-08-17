"""
build_sample_corpus.py
----------------------
Generates data/sample_corpus.json: a small, hand-written corpus used when there
is no internet connection (exam machines, sandboxes). Run once:

    python build_sample_corpus.py

The live app prefers real Wikipedia crawling; this file is only the fallback.
"""

import json
import hashlib
import os

# (title, seed_topic, text) ---------------------------------------------------
DOCS = [
    # ---- Topic: Information retrieval ----
    ("Information retrieval", "Information retrieval",
     "Information retrieval is the science of searching for relevant documents "
     "within large collections of text. A search engine takes a user query, "
     "matches it against an index, and returns a ranked list of documents. "
     "Core components include crawling, indexing, ranking and evaluation. "
     "Retrieval quality is measured with precision, recall and ranking metrics "
     "such as mean average precision and normalised discounted cumulative gain."),
    ("Inverted index", "Information retrieval",
     "An inverted index is the central data structure of a search engine. It "
     "maps every term to the list of documents that contain that term, called a "
     "postings list. Instead of scanning every document, the engine looks up the "
     "query terms and intersects their postings lists. Inverted indexes support "
     "fast Boolean and ranked retrieval over millions of documents and are built "
     "using algorithms such as blocked sort-based indexing and SPIMI."),
    ("Tf idf", "Information retrieval",
     "Term frequency inverse document frequency, or tf-idf, is a weighting scheme "
     "that scores how important a term is to a document within a collection. Term "
     "frequency rewards words that appear often in a document, while inverse "
     "document frequency penalises words that appear in many documents. The vector "
     "space model represents documents as tf-idf vectors and ranks them by cosine "
     "similarity to the query vector."),
    ("PageRank", "Information retrieval",
     "PageRank is a link-analysis algorithm that measures the importance of web "
     "pages based on the structure of hyperlinks. It models a random surfer who "
     "clicks links at random; pages that receive many links from other important "
     "pages accumulate a higher score. PageRank complements text relevance by "
     "adding an authority signal, so highly connected documents rank higher even "
     "when their text match is similar to others."),
    ("Vector space model", "Information retrieval",
     "The vector space model represents each document and query as a vector in a "
     "high dimensional term space. Each dimension corresponds to a term, weighted "
     "by tf-idf. The similarity between a query and a document is computed as the "
     "cosine of the angle between their vectors. Documents are then ranked in "
     "decreasing order of cosine similarity, giving a ranked retrieval result."),

    # ---- Topic: Machine learning ----
    ("Machine learning", "Machine learning",
     "Machine learning is a field of artificial intelligence in which computer "
     "systems learn patterns from data instead of being explicitly programmed. "
     "Learning tasks are commonly grouped into supervised learning, unsupervised "
     "learning and reinforcement learning. A model is trained on example data, "
     "evaluated on unseen data, and tuned to generalise well. Machine learning "
     "powers recommendation systems, ranking, classification and prediction."),
    ("Supervised learning", "Machine learning",
     "Supervised learning trains a model on labelled examples, where each input is "
     "paired with a known correct output. The model learns a mapping from inputs "
     "to outputs and is then used to predict labels for new, unseen inputs. Common "
     "supervised tasks are classification and regression. Popular algorithms "
     "include logistic regression, decision trees, support vector machines and "
     "neural networks."),
    ("Neural network", "Machine learning",
     "A neural network is a machine learning model loosely inspired by the brain. "
     "It is built from layers of connected units called neurons, each applying a "
     "weighted sum followed by a nonlinear activation function. Training adjusts "
     "the weights using backpropagation and gradient descent to minimise a loss "
     "function. Deep neural networks with many layers drive modern advances in "
     "vision and natural language processing."),
    ("Decision tree", "Machine learning",
     "A decision tree is a supervised learning model that predicts an output by "
     "asking a sequence of yes or no questions about the input features. Each "
     "internal node splits the data on a feature, and each leaf assigns a label. "
     "Decision trees are easy to interpret and form the basis of powerful ensemble "
     "methods such as random forests and gradient boosted trees."),
    ("Overfitting", "Machine learning",
     "Overfitting occurs when a machine learning model learns the training data "
     "too closely, including its noise, and therefore performs poorly on new data. "
     "Signs of overfitting include high accuracy on training data but low accuracy "
     "on test data. Techniques to reduce overfitting include using more data, "
     "regularisation, simpler models, cross validation and early stopping."),

    # ---- Topic: Natural language processing ----
    ("Natural language processing", "Natural language processing",
     "Natural language processing is the branch of artificial intelligence that "
     "enables computers to understand, interpret and generate human language. "
     "Typical tasks include tokenisation, part of speech tagging, named entity "
     "recognition, machine translation and question answering. Modern natural "
     "language processing relies heavily on word embeddings and neural networks "
     "trained on very large text collections."),
    ("Word embedding", "Natural language processing",
     "A word embedding is a dense numeric vector that represents the meaning of a "
     "word. Words used in similar contexts end up with similar vectors, so simple "
     "arithmetic on embeddings can capture analogies. Popular methods to learn "
     "embeddings include word2vec, GloVe and the skip gram model. Embeddings are a "
     "core input to most modern natural language processing systems."),
    ("Tokenization", "Natural language processing",
     "Tokenisation is the first step in most natural language processing pipelines. "
     "It splits raw text into smaller units called tokens, usually words or "
     "subwords. Good tokenisation handles punctuation, contractions and special "
     "characters consistently. The resulting tokens are then cleaned, for example "
     "by lowercasing and removing stop words, before being turned into features."),
    ("Named entity recognition", "Natural language processing",
     "Named entity recognition is a natural language processing task that locates "
     "and classifies named entities in text, such as people, organisations, "
     "locations and dates. It is often framed as a sequence labelling problem and "
     "solved with models such as conditional random fields or neural networks. "
     "Named entity recognition supports search, question answering and information "
     "extraction."),

    # ---- Topic: Big data ----
    ("Big data", "Big data",
     "Big data refers to datasets so large or complex that traditional tools "
     "cannot process them efficiently. It is often described by the three Vs: "
     "volume, velocity and variety. Big data systems distribute storage and "
     "computation across clusters of machines. Frameworks such as Hadoop and Spark "
     "let engineers store and analyse petabytes of data reliably and in parallel."),
    ("Apache Kafka", "Big data",
     "Apache Kafka is a distributed streaming platform used to publish and "
     "subscribe to streams of records in real time. Producers write messages to "
     "topics, which are split into partitions and replicated across brokers for "
     "fault tolerance. Consumers read messages in order within each partition. "
     "Kafka is widely used to build real time data pipelines and event driven "
     "applications at large scale."),
    ("Apache Hadoop", "Big data",
     "Apache Hadoop is an open source framework for distributed storage and "
     "processing of very large datasets across clusters of commodity hardware. Its "
     "core components are the Hadoop distributed file system for storage and "
     "MapReduce or YARN for computation. Hadoop made it practical for many "
     "organisations to run big data analytics affordably and at scale."),
    ("MapReduce", "Big data",
     "MapReduce is a programming model for processing large datasets in parallel "
     "across a cluster. The map step transforms input records into intermediate "
     "key value pairs, and the reduce step aggregates all values sharing a key. "
     "The framework handles data distribution, scheduling and fault tolerance, so "
     "developers focus only on the map and reduce logic."),
    ("Data lake", "Big data",
     "A data lake is a centralised repository that stores raw structured and "
     "unstructured data at any scale. Unlike a traditional data warehouse, a data "
     "lake keeps data in its native format until it is needed, following a schema "
     "on read approach. Data lakes support analytics, machine learning and big "
     "data processing over diverse data sources."),
]

# Cross-links between documents (source_title -> target_title). These give the
# link graph structure so PageRank and HITS produce meaningful scores.
EDGES = [
    ("Information retrieval", "Inverted index"),
    ("Information retrieval", "Tf idf"),
    ("Information retrieval", "PageRank"),
    ("Information retrieval", "Vector space model"),
    ("Information retrieval", "Machine learning"),
    ("Inverted index", "Tf idf"),
    ("Tf idf", "Vector space model"),
    ("Vector space model", "Tf idf"),
    ("PageRank", "Information retrieval"),
    ("Vector space model", "Information retrieval"),
    ("Machine learning", "Supervised learning"),
    ("Machine learning", "Neural network"),
    ("Supervised learning", "Decision tree"),
    ("Supervised learning", "Overfitting"),
    ("Neural network", "Machine learning"),
    ("Neural network", "Natural language processing"),
    ("Decision tree", "Supervised learning"),
    ("Natural language processing", "Word embedding"),
    ("Natural language processing", "Tokenization"),
    ("Natural language processing", "Named entity recognition"),
    ("Word embedding", "Neural network"),
    ("Named entity recognition", "Natural language processing"),
    ("Big data", "Apache Kafka"),
    ("Big data", "Apache Hadoop"),
    ("Big data", "MapReduce"),
    ("Big data", "Data lake"),
    ("Apache Hadoop", "MapReduce"),
    ("MapReduce", "Apache Hadoop"),
    ("Apache Kafka", "Big data"),
    ("Data lake", "Big data"),
    ("Machine learning", "Big data"),
]


def content_hash(text):
    return hashlib.md5(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def main():
    docs = []
    for i, (title, topic, text) in enumerate(DOCS):
        docs.append({
            "id": i,
            "title": title,
            "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
            "content": text,
            "seed_topic": topic,
            "categories": [topic],
            "content_hash": content_hash(text),
            "depth": 0,
        })

    os.makedirs("data", exist_ok=True)
    with open("data/sample_corpus.json", "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "edges": EDGES}, f, indent=2)
    print("Wrote data/sample_corpus.json with", len(docs), "documents and",
          len(EDGES), "edges.")


if __name__ == "__main__":
    main()
