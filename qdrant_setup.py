"""
qdrant_setup.py - Create all Qdrant collections used in the project.

Safe to re-run: every collection is deleted and created again from scratch.

    python qdrant_setup.py

Collections created:
    capstone_cosine    Distance.COSINE, default HNSW (m=16, ef_construct=100)
    capstone_euclid    Distance.EUCLID, default HNSW
    capstone_dot       Distance.DOT,    default HNSW

All of them get the SAME raw (unnormalized) vectors and the SAME payload,
so the only thing that changes between them is the distance metric.
"""

import json
import time

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionStatus,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

QDRANT_URL = "http://localhost:6333"
BATCH_SIZE = 256        # how many points we send to Qdrant in one request
VECTOR_SIZE = 384       # all-MiniLM-L6-v2 outputs 384 numbers per text

# Qdrant's own default HNSW settings, written out so they are visible.
DEFAULT_HNSW = HnswConfigDiff(
    m=16,                       # each node links to up to 16 neighbours per layer
    ef_construct=100,           # how many candidates are considered while building the graph
    full_scan_threshold=10,     # (KB) never fall back to a full scan because the data is "small"
)

# Optimizer settings that FORCE Qdrant to build the HNSW graph.
#
# indexing_threshold is in kilobytes. Qdrant only builds HNSW for a segment
# once it holds more than this many KB of vectors. The default is 20000 KB.
# Our data is 6000 x 384 x 4 bytes = about 9000 KB, which is BELOW the default,
# so without this change Qdrant would never build HNSW and would silently
# brute-force every search. (Careful: 0 means "never index", not "always".)
#
# default_segment_number=1 asks Qdrant to keep the data in as few segments as
# possible (a segment is a chunk of the collection with its own HNSW graph).
# Qdrant may still keep a second, small segment, which is fine: every segment
# above the threshold gets its own HNSW graph, and we check below that ALL
# 6000 vectors are indexed.
FORCE_INDEXING = OptimizersConfigDiff(
    indexing_threshold=10,
    default_segment_number=1,
)

# name -> distance metric for the Part 2 experiment
METRIC_COLLECTIONS = {
    "capstone_cosine": Distance.COSINE,
    "capstone_euclid": Distance.EUCLID,
    "capstone_dot": Distance.DOT,
}


def create_collection(client, name, distance, vectors, metadata, hnsw_config):
    """Delete (if it exists) and create one collection, then upload all points."""

    # 1) Delete the old collection so the script can be re-run safely.
    if client.collection_exists(name):
        client.delete_collection(name)
        print(f"[{name}] deleted old collection")

    # 2) Create an empty collection with our settings.
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=distance),
        hnsw_config=hnsw_config,
        optimizers_config=FORCE_INDEXING,
    )
    print(f"[{name}] created  (distance={distance.value}, m={hnsw_config.m}, "
          f"ef_construct={hnsw_config.ef_construct})")

    # 3) Upload the points in batches of BATCH_SIZE.
    #    A point = id + vector + payload (extra data stored next to the vector).
    total = len(vectors)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch = []
        for i in range(start, end):
            batch.append(
                PointStruct(
                    id=metadata[i]["id"],               # same id as the row in the .npy file
                    vector=vectors[i].tolist(),         # numpy array -> plain Python list
                    payload={
                        "category": metadata[i]["category"],
                        "text": metadata[i]["text"],
                    },
                )
            )

        # wait=True: only return once Qdrant has stored the batch.
        client.upsert(collection_name=name, points=batch, wait=True)

    print(f"[{name}] uploaded {total} points in batches of {BATCH_SIZE}")


def wait_until_indexed(client, name, timeout_seconds=300):
    """
    Wait until Qdrant has finished building the HNSW index.
    "Done" means: status is GREEN and every point is inside the HNSW index
    (indexed_vectors_count == points_count).
    """
    start_time = time.time()
    while True:
        info = client.get_collection(name)
        points = info.points_count
        indexed = info.indexed_vectors_count

        if info.status == CollectionStatus.GREEN and indexed == points:
            print(f"[{name}] status={info.status.value}  "
                  f"indexed_vectors_count={indexed} / points_count={points}  "
                  f"segments={info.segments_count}  -> HNSW is built")
            return

        if time.time() - start_time > timeout_seconds:
            raise RuntimeError(f"{name}: index not ready after {timeout_seconds}s "
                               f"(status={info.status}, indexed={indexed}/{points})")

        time.sleep(1)   # check again in one second


def main():
    # ---- Load the data made by embed.py ---------------------------------
    vectors_raw = np.load("data/vectors_raw.npy")
    with open("data/metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    # ---- Connect to the local Qdrant server (running in Docker) ---------
    client = QdrantClient(url=QDRANT_URL)

    # ---- Part 2: one collection per distance metric, RAW vectors --------
    # Raw vectors keep their original length. That is what lets DOT and
    # EUCLID disagree with COSINE (COSINE ignores length, the others do not).
    for name in METRIC_COLLECTIONS:
        distance = METRIC_COLLECTIONS[name]
        create_collection(client, name, distance, vectors_raw, metadata, DEFAULT_HNSW)

    # ---- Wait for every HNSW graph to be built --------------------------
    print("\nWaiting for Qdrant to build the HNSW indexes ...")
    for name in METRIC_COLLECTIONS:
        wait_until_indexed(client, name)

    print("\nAll collections ready. Open http://localhost:6333/dashboard to see them.")


if __name__ == "__main__":
    main()
