"""
Build a FAISS index from QA JSON files for semantic search.

Supported JSON formats:

    Format A (grouped questions):
        [
            {
                "questions": ["How do I reset?", "Where is the reset button?"],
                "a": "Press the red button on the back panel."
            }
        ]

    Format B (single QA pairs):
        [
            {
                "q": "How do I reset?",
                "a": "Press the red button on the back panel."
            }
        ]

Both formats can be mixed across files. The script will auto-detect
which format each entry uses.

Prerequisites:
    1. Start the embedding server:  python embedding_server.py
    2. Run this script:             python build_index.py

Output:
    qa_index.faiss   - FAISS inner-product index
    qa_data.pkl      - Pickled questions and answers list
"""

import json
import logging
import time
import base64
import argparse

import numpy as np
import faiss
import pickle
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build-index")

# ── Config ──
EMBED_BATCH_URL = "http://localhost:8100/embed_batch"
INDEX_FILE = "qa_index.faiss"
DATA_FILE = "qa_data.pkl"
BATCH_SIZE = 64


def load_qa_files(filepaths: list[str]) -> tuple[list[str], list[str]]:
    """Load and merge QA pairs from multiple JSON files.

    Supports both grouped format ({"questions": [...], "a": "..."})
    and flat format ({"q": "...", "a": "..."}).

    Returns:
        Tuple of (questions, answers) lists.
    """
    questions = []
    answers = []

    for filepath in filepaths:
        with open(filepath) as f:
            raw_data = json.load(f)

        count = 0
        for item in raw_data:
            a = item.get("a") or item.get("answer") or ""

            if "questions" in item:
                for q in item["questions"]:
                    questions.append(q)
                    answers.append(a)
                    count += 1
            else:
                q = item.get("q") or item.get("question") or ""
                questions.append(q)
                answers.append(a)
                count += 1

        logger.info("Loaded %s: %d question-answer pairs", filepath, count)

    logger.info("Total: %d pairs from %d files", len(questions), len(filepaths))
    return questions, answers


def embed_questions(questions: list[str]) -> np.ndarray:
    """Send questions to embedding server in batches. Returns (N, dim) float32 array."""
    logger.info("Embedding %d questions via %s", len(questions), EMBED_BATCH_URL)
    session = requests.Session()
    all_embeddings = []
    start = time.perf_counter()

    for i in range(0, len(questions), BATCH_SIZE):
        batch = questions[i : i + BATCH_SIZE]
        resp = session.post(EMBED_BATCH_URL, json={"queries": batch}, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        batch_embs = [
            np.frombuffer(base64.b64decode(b), dtype="float32")
            for b in data["embeddings_b64"]
        ]
        all_embeddings.extend(batch_embs)
        logger.info(
            "  [%d/%d] batch embedded (%.1fms)",
            i + len(batch),
            len(questions),
            data["latency_ms"],
        )

    elapsed = time.perf_counter() - start
    embeddings = np.array(all_embeddings, dtype="float32")
    logger.info(
        "Embedding complete: %.2fs total (%.1fms per query), shape=%s",
        elapsed,
        elapsed / len(questions) * 1000,
        embeddings.shape,
    )
    return embeddings


def build_and_save(embeddings: np.ndarray, questions: list[str], answers: list[str]):
    """Build FAISS index and save to disk."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dimension)

    faiss.write_index(index, INDEX_FILE)
    with open(DATA_FILE, "wb") as f:
        pickle.dump({"questions": questions, "answers": answers}, f)

    logger.info("Saved %s and %s", INDEX_FILE, DATA_FILE)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from QA JSON files")
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more QA JSON files to index",
    )
    args = parser.parse_args()

    questions, answers = load_qa_files(args.files)
    if not questions:
        logger.error("No QA pairs found. Check your input files.")
        return

    embeddings = embed_questions(questions)
    build_and_save(embeddings, questions, answers)
    logger.info("Done. Ready for search.")


if __name__ == "__main__":
    main()
