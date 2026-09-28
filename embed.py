"""
embed.py - Turn every document and every test query into a 384-number vector.

Run once on a fresh checkout:
    python embed.py

It creates the data/ folder with:
    vectors_normalized.npy        documents, each vector has length 1
    vectors_raw.npy               documents, vectors exactly as the model outputs them
    metadata.json                 id, category, text for every document
    queries.json                  the 5 query strings (read from queries.md)
    query_vectors_normalized.npy  the 5 queries, length-1 vectors
    query_vectors_raw.npy         the 5 queries, raw vectors
"""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from data import load_documents

MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast model; outputs 384-dimensional vectors
DATA_DIR = "data"
QUERIES_FILE = "queries.md"


def read_queries(path=QUERIES_FILE):
    """
    Read the numbered queries from queries.md.
    A query line looks like:  "1. a question about a graphics card driver"
    """
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # A query line starts with a digit followed by ". "
            if len(line) > 3 and line[0].isdigit() and line[1:3] == ". ":
                queries.append(line[3:])   # keep only the text after "N. "
    return queries


def main():
    # Make the data/ folder if it does not exist yet.
    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- 1) Load the documents -------------------------------------------
    documents = load_documents()
    texts = []
    for doc in documents:
        texts.append(doc["text"])

    # ---- 2) Load the embedding model (downloaded the first time) --------
    print(f"\nLoading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    # IMPORTANT: all-MiniLM-L6-v2 is made of 3 layers:
    #     model[0] Transformer  -> one vector per word/token
    #     model[1] Pooling      -> average of the token vectors = one sentence vector
    #     model[2] Normalize    -> divides the sentence vector by its length
    # Because of layer [2], model.encode(..., normalize_embeddings=False)
    # STILL returns length-1 vectors. To get truly raw vectors we build a
    # second model that uses only the first two layers (it shares the same
    # weights, nothing is retrained). Its output is the sentence vector
    # just before the normalize step, with its natural length (about 4-7).
    raw_model = SentenceTransformer(modules=[model[0], model[1]], device="cpu")

    # ---- 3) Embed the documents ------------------------------------------
    print(f"Embedding {len(texts)} documents, normalized (full model) ...")
    vectors_normalized = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,    # every vector gets length 1
    )

    print(f"Embedding {len(texts)} documents, raw (model without Normalize layer) ...")
    vectors_raw = raw_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=False,   # keep the original length of each vector
    )

    # float32 is what Qdrant stores, and it halves the file size.
    vectors_raw = vectors_raw.astype(np.float32)
    vectors_normalized = vectors_normalized.astype(np.float32)

    np.save(os.path.join(DATA_DIR, "vectors_raw.npy"), vectors_raw)
    np.save(os.path.join(DATA_DIR, "vectors_normalized.npy"), vectors_normalized)

    # ---- 4) Save the metadata (id, category, text) -----------------------
    with open(os.path.join(DATA_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(documents, f)

    # ---- 5) Embed the 5 test queries -------------------------------------
    queries = read_queries()
    print(f"\nEmbedding {len(queries)} queries from {QUERIES_FILE}:")
    for i in range(len(queries)):
        print(f"  [{i}] {queries[i]}")

    # Same two models as for the documents.
    query_vectors_raw = raw_model.encode(queries, normalize_embeddings=False)
    query_vectors_normalized = model.encode(queries, normalize_embeddings=True)

    np.save(os.path.join(DATA_DIR, "query_vectors_raw.npy"), query_vectors_raw.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "query_vectors_normalized.npy"), query_vectors_normalized.astype(np.float32))
    with open(os.path.join(DATA_DIR, "queries.json"), "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)

    # ---- 6) Print a short check so we can see it worked ------------------
    raw_lengths = np.linalg.norm(vectors_raw, axis=1)
    norm_lengths = np.linalg.norm(vectors_normalized, axis=1)
    print("\nSaved to data/:")
    print(f"  vectors_raw.npy         shape={vectors_raw.shape}  "
          f"length min={raw_lengths.min():.3f} max={raw_lengths.max():.3f} mean={raw_lengths.mean():.3f}")
    print(f"  vectors_normalized.npy  shape={vectors_normalized.shape}  "
          f"length min={norm_lengths.min():.3f} max={norm_lengths.max():.3f}")
    print(f"  metadata.json           {len(documents)} documents")
    print(f"  query vectors           shape={query_vectors_raw.shape}  "
          f"raw lengths={np.round(np.linalg.norm(query_vectors_raw, axis=1), 3)}")

    # Sanity check: raw vector / its length should equal the normalized vector.
    # If this is True, the two files describe the same directions and differ
    # only in length, which is exactly what Part 2 needs.
    rebuilt = vectors_raw / raw_lengths[:, None]
    same_direction = np.allclose(rebuilt, vectors_normalized, atol=1e-4)
    print(f"  check: raw / length == normalized ? {same_direction}")


if __name__ == "__main__":
    main()
