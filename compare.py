"""
compare.py - Run every search method and compare them.

    python compare.py            # all 5 queries, prints results and writes results/*.json
    python compare.py --query 0  # only query 0 (prints only, does not overwrite results/)

Needs: Qdrant running (docker compose up -d), then embed.py and qdrant_setup.py run once.
"""

import argparse
import json
import os

import numpy as np
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
RESULTS_DIR = "results"
TOP_K = 5              # we always compare the top-5 results
SNIPPET_LENGTH = 70    # how many characters of text to show / save per hit

# The three Part 2 collections. For each one we note whether a HIGHER score
# means "more similar" (cosine, dot) or a LOWER score does (euclid).
#   COSINE -> score = cosine similarity, higher = more similar
#   DOT    -> score = dot product,       higher = more similar
#   EUCLID -> score = Euclidean DISTANCE, LOWER = closer (Qdrant still returns
#             the closest point first, but the number itself is a distance)
METRICS = [
    {"name": "cosine", "collection": "capstone_cosine", "higher_is_better": True},
    {"name": "euclid", "collection": "capstone_euclid", "higher_is_better": False},
    {"name": "dot",    "collection": "capstone_dot",    "higher_is_better": True},
]


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def make_snippet(text):
    """First SNIPPET_LENGTH characters of a text, on one line."""
    one_line = " ".join(text.split())   # turns newlines/tabs/double spaces into single spaces
    if len(one_line) > SNIPPET_LENGTH:
        return one_line[:SNIPPET_LENGTH] + "..."
    return one_line


def qdrant_search(client, collection, query_vector, search_params=None):
    """
    Ask Qdrant for the TOP_K closest points.
    Uses client.query_points (the current API; client.search is deprecated).
    Returns a list of dicts: {"id", "category", "score", "snippet"}.
    """
    response = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),   # numpy -> plain list of 384 floats
        limit=TOP_K,
        search_params=search_params,   # None = normal HNSW search with default ef
        with_payload=True,             # also return category + text
    )

    hits = []
    for point in response.points:
        hits.append({
            "id": point.id,
            "category": point.payload["category"],
            "score": round(point.score, 4),
            "snippet": make_snippet(point.payload["text"]),
        })
    return hits


def get_ids(hits):
    """List of ids from a list of hits (keeps the ranking order)."""
    ids = []
    for hit in hits:
        ids.append(hit["id"])
    return ids


def overlap(ids, reference_ids):
    """Top-k overlap = |intersection| / k.  1.0 = same 5 documents, 0.0 = none shared."""
    shared = set(ids) & set(reference_ids)
    return len(shared) / TOP_K


def print_hits(hits, show_length=False):
    """Print a ranked list of hits, one per line."""
    for rank in range(len(hits)):
        hit = hits[rank]
        line = f"      {rank + 1}. id={hit['id']:<5} score={hit['score']:<8} "
        if show_length:
            line += f"len={hit['vector_length']:<6} "
        line += f"{hit['category']:<25} \"{hit['snippet'][:40]}\""
        print(line)


def print_header(title):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


# ---------------------------------------------------------------------------
# Part 2: distance metrics
# ---------------------------------------------------------------------------

def run_distance_metrics(client, query_vector_raw, doc_lengths):
    """
    Search the same RAW query vector in the cosine, euclid and dot collections.
    All three collections hold the same raw vectors; only the metric differs.
    doc_lengths[i] is the length of raw document vector i. We add it to every
    hit ("len=") because length is exactly what makes the metrics disagree.
    """
    print_header("PART 2 | Distance metrics (same raw vectors, top-5 each)")
    print(f"   query vector length = {np.linalg.norm(query_vector_raw):.3f}")

    results = {}
    for metric in METRICS:
        hits = qdrant_search(client, metric["collection"], query_vector_raw)
        for hit in hits:
            hit["vector_length"] = round(float(doc_lengths[hit["id"]]), 3)
        results[metric["name"]] = hits

        if metric["higher_is_better"]:
            note = "score = similarity, higher is closer"
        else:
            note = "score = distance, LOWER is closer"
        print(f"\n   {metric['name'].upper():<6} ({note})")
        print_hits(hits, show_length=True)

    # Compare euclid and dot against cosine, so differences are easy to spot.
    cosine_ids = get_ids(results["cosine"])
    print()
    for name in ["euclid", "dot"]:
        other_ids = get_ids(results[name])
        same_top1 = other_ids[0] == cosine_ids[0]
        same_order = other_ids == cosine_ids
        print(f"   {name:<6} vs cosine: top-5 overlap {overlap(other_ids, cosine_ids):.1f}, "
              f"same #1: {'yes' if same_top1 else 'NO'}, "
              f"identical ranking: {'yes' if same_order else 'NO'}")

    return results


def normalized_metrics_agree(doc_vectors_normalized, query_vector_normalized):
    """
    Control experiment in numpy: with NORMALIZED vectors, do cosine, euclid
    and dot give the same top-5? (They should: for length-1 vectors
    dot = cosine, and euclid^2 = 2 - 2 * cosine.)
    """
    # dot product = cosine similarity here, because every length is 1
    dot_scores = doc_vectors_normalized @ query_vector_normalized

    # cosine: divide the dot product by the lengths (all 1, but we do it anyway)
    doc_lengths = np.linalg.norm(doc_vectors_normalized, axis=1)
    query_length = np.linalg.norm(query_vector_normalized)
    cosine_scores = dot_scores / (doc_lengths * query_length)

    # Euclidean distance between the query and every document
    differences = doc_vectors_normalized - query_vector_normalized
    euclid_distances = np.linalg.norm(differences, axis=1)

    # Top-5: highest similarity for cosine/dot, LOWEST distance for euclid
    top_dot = np.argsort(dot_scores)[::-1][:TOP_K].tolist()
    top_cosine = np.argsort(cosine_scores)[::-1][:TOP_K].tolist()
    top_euclid = np.argsort(euclid_distances)[:TOP_K].tolist()

    return top_dot == top_cosine and top_cosine == top_euclid


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    # ---- Command line: which queries to run -----------------------------
    parser = argparse.ArgumentParser(description="Compare search methods in Qdrant.")
    parser.add_argument("--query", type=int, default=None,
                        help="run only this query index (0-4); default runs all")
    args = parser.parse_args()

    # ---- Load the data made by embed.py ---------------------------------
    with open("data/queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    query_vectors_raw = np.load("data/query_vectors_raw.npy")
    query_vectors_normalized = np.load("data/query_vectors_normalized.npy")
    doc_vectors_normalized = np.load("data/vectors_normalized.npy")
    doc_vectors_raw = np.load("data/vectors_raw.npy")

    # Length of every raw document vector (used to explain Part 2).
    doc_lengths = np.linalg.norm(doc_vectors_raw, axis=1)

    if args.query is None:
        query_indexes = list(range(len(queries)))
    else:
        query_indexes = [args.query]

    client = QdrantClient(url=QDRANT_URL)

    # One entry per query, filled in below and saved as JSON at the end.
    distance_results = []

    for qi in query_indexes:
        print("\n" + "=" * 78)
        print(f"QUERY {qi}: \"{queries[qi]}\"")
        print("=" * 78)

        # ---- Part 2 ------------------------------------------------------
        metric_hits = run_distance_metrics(client, query_vectors_raw[qi], doc_lengths)
        agree = normalized_metrics_agree(doc_vectors_normalized, query_vectors_normalized[qi])
        print(f"   control: with NORMALIZED vectors, cosine/euclid/dot give the same top-5? "
              f"{'yes' if agree else 'NO'}")

        distance_results.append({
            "query_index": qi,
            "query": queries[qi],
            "cosine": metric_hits["cosine"],
            "euclid": metric_hits["euclid"],
            "dot": metric_hits["dot"],
            "normalized_vectors_all_metrics_agree": agree,
        })

    # ---- Save results (only for a full run, so a demo of one query -------
    # ---- does not overwrite the full results) ---------------------------
    if args.query is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "distance_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(distance_results, f, indent=2)
        print(f"\nSaved {RESULTS_DIR}/distance_metrics.json")


if __name__ == "__main__":
    main()
