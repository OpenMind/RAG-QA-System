"""
Search the FAISS index for the most relevant QA pairs.

Usage:
    python search.py "your question here"
    python search.py --top 5 "your question here"

Prerequisites:
    1. Embedding server running:  python embedding_server.py
    2. Index already built:       python build_index.py example_qa.json
"""

import argparse
import base64
import logging
import pickle
import time

import faiss
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("search")

EMBED_URL = "http://localhost:8100/embed"
INDEX_FILE = "qa_index.faiss"
DATA_FILE = "qa_data.pkl"


def load_index():
    """Load FAISS index and QA data from disk."""
    index = faiss.read_index(INDEX_FILE)
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded index with %d vectors", index.ntotal)
    return index, data["questions"], data["answers"]


def embed_query(query: str) -> np.ndarray:
    """Get embedding for a single query from the embedding server."""
    resp = requests.post(EMBED_URL, json={"query": query}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return np.frombuffer(base64.b64decode(data["embedding_b64"]), dtype="float32")


def search(query: str, top_k: int = 3):
    """Search the index and print results."""
    index, questions, answers = load_index()

    start = time.perf_counter()
    query_vec = embed_query(query).reshape(1, -1)
    scores, indices = index.search(query_vec, top_k)
    elapsed = (time.perf_counter() - start) * 1000

    logger.info("Search completed in %.1fms", elapsed)
    print(f"\nQuery: {query}")
    print(f"{'=' * 60}")

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx == -1:
            break
        print(f"\n--- Result {rank} (score: {score:.4f}) ---")
        print(f"Matched Q: {questions[idx]}")
        print(f"Answer:    {answers[idx]}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Search QA index")
    parser.add_argument("query", help="Question to search for")
    parser.add_argument("--top", type=int, default=3, help="Number of results (default: 3)")
    args = parser.parse_args()
    search(args.query, args.top)


if __name__ == "__main__":
    main()
