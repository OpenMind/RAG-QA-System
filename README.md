# RAG QA System

A lightweight retrieval-augmented generation (RAG) pipeline for question answering. It embeds QA pairs into a FAISS vector index using a local embedding server, then retrieves the most relevant answers via semantic similarity search.

## Architecture

```
                  +-----------------------+
                  |   QA JSON Files       |
                  |  (your knowledge base)|
                  +-----------+-----------+
                              |
                    build_index.py
                              |
               +--------------+--------------+
               |                             |
    +----------v----------+     +------------v-----------+
    |  embedding_server.py |     |     FAISS Index        |
    |  (e5-small-v2 model) |     |  qa_index.faiss        |
    |  localhost:8100      |     |  qa_data.pkl           |
    +----------+-----------+     +------------+-----------+
               |                              |
               +---------- search.py ---------+
                              |
                        Top-K answers
```

**How it works:**

1. `embedding_server.py` loads the `intfloat/e5-small-v2` sentence transformer model and serves embeddings over HTTP on port 8100. Runs on GPU if available, falls back to CPU.

2. `build_index.py` reads your QA JSON files, sends all questions to the embedding server, and builds a FAISS inner-product index for fast similarity search.

3. `search.py` takes a user query, embeds it via the same server, and finds the closest matching questions in the index. Returns the associated answers ranked by similarity score.

4. `generate_q_variants.py` (optional) uses the Claude API to generate alternative phrasings of your questions, improving search recall.


## QA Data Format

Your JSON files can use either format (or mix both):

**Format A -- Grouped questions (recommended for better search recall):**

```json
[
  {
    "questions": [
      "What is the WiFi password?",
      "How do I connect to WiFi?",
      "WiFi password"
    ],
    "a": "The WiFi network is 'ConferenceNet' and the password is 'welcome2024'."
  }
]
```

**Format B -- Single QA pairs (simplest):**

```json
[
  {
    "q": "What is the WiFi password?",
    "a": "The WiFi network is 'ConferenceNet' and the password is 'welcome2024'."
  }
]
```

Format A maps multiple question phrasings to the same answer, which improves search accuracy. Format B is simpler if you just want to get started quickly. Both formats can coexist in the same file.


## Setup

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for faster embeddings

### macOS

```bash
# Install Python if needed
brew install python@3.12

# Clone or copy this project
cd rag_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Note: On macOS, `faiss-cpu` is used. The embedding server will run on CPU, which is fine for building indexes and testing. Typical latency is ~20ms per query on Apple Silicon.

### Linux (Ubuntu/Debian)

```bash
# Install Python if needed
sudo apt update
sudo apt install python3 python3-venv python3-pip

# Clone or copy this project
cd rag_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

For GPU acceleration on Linux with NVIDIA GPU:

```bash
# Install PyTorch with CUDA (check https://pytorch.org for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install the rest
pip install -r requirements.txt
```

The embedding server auto-detects CUDA and uses GPU when available.


## Usage

All commands assume the virtual environment is activated (`source venv/bin/activate`).

### Step 1: Start the embedding server

```bash
python embedding_server.py
```

You should see:

```
2026-04-09 10:00:00 [INFO] embedding-server - Loading intfloat/e5-small-v2 on cpu ...
2026-04-09 10:00:05 [INFO] embedding-server - Model loaded and warmed up
```

Keep this running in a separate terminal.

### Step 2: Build the index

```bash
python build_index.py example_qa.json
```

You can pass multiple files:

```bash
python build_index.py faq.json sessions.json restaurants.json
```

Output:

```
2026-04-09 10:01:00 [INFO] build-index - Loaded example_qa.json: 5 question-answer pairs
2026-04-09 10:01:00 [INFO] build-index - Total: 5 pairs from 1 files
2026-04-09 10:01:00 [INFO] build-index - Embedding 5 questions via http://localhost:8100/embed_batch
2026-04-09 10:01:01 [INFO] build-index - FAISS index built: 5 vectors, dim=384
2026-04-09 10:01:01 [INFO] build-index - Saved qa_index.faiss and qa_data.pkl
```

### Step 3: Search

```bash
python search.py "where can I eat nearby"
```

Output:

```
Query: where can I eat nearby
============================================================

--- Result 1 (score: 0.8921) ---
Matched Q: Where should I eat in Downtown San Jose?
Answer:    Recommended restaurants in Downtown San Jose include ...

--- Result 2 (score: 0.8534) ---
Matched Q: best restaurants Downtown San Jose area
Answer:    Recommended restaurants in Downtown San Jose include ...
```

### (Optional) Step 4: Generate question variants

If you want to expand your QA data with alternative phrasings:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
python generate_q_variants.py input_qa.json expanded_qa.json
```

This creates an expanded file in Format A (grouped questions) that you can feed into `build_index.py`. If interrupted, re-run the same command and it resumes from where it stopped.


## File Overview

| File                     | Purpose                                          | Required |
|--------------------------|--------------------------------------------------|----------|
| `embedding_server.py`    | Local HTTP embedding service (e5-small-v2)       | Yes      |
| `build_index.py`         | Build FAISS index from QA JSON files             | Yes      |
| `search.py`              | Query the index from command line                | Yes      |
| `generate_q_variants.py` | Expand questions via Claude API                  | No       |
| `requirements.txt`       | Python dependencies                              | Yes      |
| `example_qa.json`        | Sample QA data to test with                      | No       |


## Integrating Into Your Application

The core search logic is straightforward to embed in any Python application:

```python
import faiss
import pickle
import numpy as np
import base64
import requests

# Load once at startup
index = faiss.read_index("qa_index.faiss")
with open("qa_data.pkl", "rb") as f:
    data = pickle.load(f)

# Search function
def search_qa(query: str, top_k: int = 3) -> list[dict]:
    resp = requests.post(
        "http://localhost:8100/embed",
        json={"query": query},
    )
    vec = np.frombuffer(
        base64.b64decode(resp.json()["embedding_b64"]),
        dtype="float32",
    ).reshape(1, -1)

    scores, indices = index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            break
        results.append({
            "question": data["questions"][idx],
            "answer": data["answers"][idx],
            "score": float(score),
        })
    return results
```
