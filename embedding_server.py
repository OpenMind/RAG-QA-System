"""
Embedding microservice using e5-small-v2 sentence transformer.

Provides REST endpoints for generating text embeddings via FastAPI.
Runs locally -- no Docker required.

Endpoints:
    POST /embed         - Single query embedding
    POST /embed_batch   - Batch query embedding
    GET  /health        - Health check

Start:
    python embedding_server.py

Or with uvicorn directly:
    uvicorn embedding_server:app --host 0.0.0.0 --port 8100
"""

import logging
import time
import base64

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("embedding-server")

app = FastAPI(title="Embedding Service")
model = None


# ── Request / Response schemas ──


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    embedding_b64: str
    dimension: int
    latency_ms: float


class BatchRequest(BaseModel):
    queries: list[str]


class BatchResponse(BaseModel):
    embeddings_b64: list[str]
    dimension: int
    count: int
    latency_ms: float


# ── Startup ──


@app.on_event("startup")
def load_model():
    """Load sentence transformer model and run warmup inferences."""
    global model
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            logger.info("CUDA detected, using GPU")
    except ImportError:
        pass

    logger.info("Loading intfloat/e5-small-v2 on %s ...", device)
    model = SentenceTransformer("intfloat/e5-small-v2", device=device)

    for _ in range(3):
        model.encode(["warmup"], normalize_embeddings=True)
    logger.info("Model loaded and warmed up")


# ── Endpoints ──


@app.post("/embed", response_model=QueryResponse)
def embed(req: QueryRequest):
    """Embed a single query string."""
    start = time.perf_counter()
    emb = model.encode(
        [f"query: {req.query}"],
        normalize_embeddings=True,
    ).astype("float32")
    latency = (time.perf_counter() - start) * 1000

    emb_b64 = base64.b64encode(emb[0].tobytes()).decode("ascii")
    logger.info("embed | query=%s | latency=%.1fms", req.query[:60], latency)
    return QueryResponse(
        embedding_b64=emb_b64,
        dimension=len(emb[0]),
        latency_ms=round(latency, 2),
    )


@app.post("/embed_batch", response_model=BatchResponse)
def embed_batch(req: BatchRequest):
    """Embed multiple queries in a single batch."""
    start = time.perf_counter()
    prefixed = [f"query: {q}" for q in req.queries]
    embs = model.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=64,
    ).astype("float32")
    latency = (time.perf_counter() - start) * 1000

    embs_b64 = [base64.b64encode(e.tobytes()).decode("ascii") for e in embs]
    logger.info("embed_batch | count=%d | latency=%.1fms", len(req.queries), latency)
    return BatchResponse(
        embeddings_b64=embs_b64,
        dimension=len(embs[0]),
        count=len(embs),
        latency_ms=round(latency, 2),
    )


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "model": "e5-small-v2"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
