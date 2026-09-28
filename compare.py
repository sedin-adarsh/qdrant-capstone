"""
compare.py - Run every search method and compare them.

    python compare.py            # all 5 queries, prints results and writes results/*.json
    python compare.py --query 0  # only query 0 (prints only, does not overwrite results/)

Needs: Qdrant running (docker compose up -d), then embed.py and qdrant_setup.py run once.
"""

import os

# Run numpy's maths on ONE CPU thread. This must be set BEFORE numpy is imported.
# Why: one query is a tiny calculation. With many threads, numpy spends more
# time starting and syncing threads than calculating, the timings jump around
# (we saw 0.1 ms and 13 ms for similar work), and the busy threads also steal
# CPU from the Qdrant server we are timing. One thread gives stable, fair numbers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import time

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from ivf import build_ivf_index, search_ivf

QDRANT_URL = "http://localhost:6333"
RESULTS_DIR = "results"
TOP_K = 5              # we always compare the top-5 results
SNIPPET_LENGTH = 70    # how many characters of text to show / save per hit
LATENCY_RUNS = 10      # each timed search is repeated this many times and averaged

# Part 3: the two HNSW collections and the query-time ef values we try.
# hnsw_ef = how many candidates HNSW keeps while searching the graph.
# Bigger ef = more of the graph explored = slower but more accurate.
HNSW_EF_VALUES = [16, 64, 128]
HNSW_COLLECTIONS = [
    {"method": "HNSW default", "collection": "capstone_cosine",
     "build": "m=16, ef_construct=100"},
    {"method": "HNSW weak", "collection": "capstone_hnsw_weak",
     "build": "m=4, ef_construct=8"},
]

# Part 4: our own IVF index. nprobe = how many clusters we search.
IVF_CLUSTERS = 40
IVF_NPROBE_VALUES = [1, 4, 8, 16]

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


def timed(function):
    """
    Measure how long a search takes.
    `function` is a search wrapped in a lambda, e.g.  timed(lambda: qdrant_search(...))
    so we can call it several times.
      1) call it once as a warm-up (not timed: first calls are often slower)
      2) call it LATENCY_RUNS times, timing each call with time.perf_counter()
    Returns (the search result, average time in milliseconds).
    """
    result = function()   # warm-up call

    total_seconds = 0.0
    for run in range(LATENCY_RUNS):
        start = time.perf_counter()
        result = function()
        total_seconds += time.perf_counter() - start

    average_ms = total_seconds / LATENCY_RUNS * 1000
    return result, average_ms


def summarize(rows):
    """
    Average overlap and latency over all queries, per (method, setting).
    rows is a list like [{"method": "HNSW weak", "setting": "ef=16", "overlap": 0.8, ...}, ...]
    Returns one dict per (method, setting), in the order they first appear.
    """
    groups = {}   # (method, setting) -> list of rows
    for row in rows:
        key = (row["method"], row["setting"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    summary = []
    for key in groups:
        group_rows = groups[key]
        overlaps = []
        latencies = []
        for row in group_rows:
            overlaps.append(row["overlap"])
            latencies.append(row["latency_ms"])
        entry = {
            "method": key[0],
            "setting": key[1],
            "avg_overlap": round(float(np.mean(overlaps)), 3),
            "avg_latency_ms": round(float(np.mean(latencies)), 3),
            "queries": len(group_rows),
        }
        # IVF rows also say how many vectors were compared.
        if "scanned" in group_rows[0]:
            scanned = []
            for row in group_rows:
                scanned.append(row["scanned"])
            entry["avg_scanned"] = round(float(np.mean(scanned)), 1)
        summary.append(entry)
    return summary


def print_summary_table(summary):
    """Print the averaged results as a table."""
    print(f"   {'method':<16} {'setting':<30} {'avg overlap':>11} {'avg latency ms':>15}")
    for entry in summary:
        print(f"   {entry['method']:<16} {entry['setting']:<30} "
              f"{entry['avg_overlap']:>11.2f} {entry['avg_latency_ms']:>15.3f}")


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
# Part 3: exact search (ground truth) and HNSW
# ---------------------------------------------------------------------------

def print_index_status(client):
    """Prove that the HNSW graphs really exist (otherwise Qdrant brute-forces)."""
    print("\nHNSW index check (indexed_vectors_count must equal points_count):")
    for config in HNSW_COLLECTIONS:
        info = client.get_collection(config["collection"])
        print(f"   {config['collection']:<20} {config['build']:<24} status={info.status.value:<6} "
              f"indexed={info.indexed_vectors_count}/{info.points_count}")


def run_exact(client, query_vector_raw, qi):
    """
    Ground truth: exact=True makes Qdrant skip HNSW and compare the query with
    EVERY vector (brute force). Slow, but guaranteed to find the true top-5.
    capstone_cosine uses COSINE, so Qdrant normalizes the raw vectors itself.
    """
    print_header("PART 3 | Exact search = ground truth (capstone_cosine, exact=True)")

    exact_params = SearchParams(exact=True)
    hits, latency_ms = timed(
        lambda: qdrant_search(client, "capstone_cosine", query_vector_raw, exact_params)
    )
    print_hits(hits)
    print(f"   latency: {latency_ms:.3f} ms (average of {LATENCY_RUNS} runs)")

    row = {
        "query_index": qi,
        "method": "Exact (Qdrant)",
        "setting": "exact=True",
        "ids": get_ids(hits),
        "overlap": 1.0,        # exact search agrees with itself by definition
        "latency_ms": round(latency_ms, 3),
        "hits": hits,
    }
    return row


def run_hnsw(client, query_vector_raw, qi, exact_ids):
    """Run every HNSW collection at every hnsw_ef and compare with exact search."""
    print_header("PART 3 | HNSW approximate search vs exact (overlap = shared ids / 5)")
    print(f"   exact ids: {exact_ids}")

    rows = []
    for config in HNSW_COLLECTIONS:
        print(f"\n   {config['method']} ({config['collection']}, {config['build']})")

        for ef in HNSW_EF_VALUES:
            # exact=False (the default) means: use the HNSW graph.
            params = SearchParams(hnsw_ef=ef, exact=False)
            hits, latency_ms = timed(
                lambda: qdrant_search(client, config["collection"], query_vector_raw, params)
            )
            ids = get_ids(hits)
            score = overlap(ids, exact_ids)

            print(f"      hnsw_ef={ef:<4} overlap={score:.1f}  latency={latency_ms:6.3f} ms  ids={ids}")

            rows.append({
                "query_index": qi,
                "method": config["method"],
                "setting": f"{config['build']}, ef={ef}",
                "collection": config["collection"],
                "hnsw_ef": ef,
                "ids": ids,
                "overlap": score,
                "latency_ms": round(latency_ms, 3),
            })
    return rows


# ---------------------------------------------------------------------------
# Part 4: custom IVF index
# ---------------------------------------------------------------------------

def numpy_exact_search(doc_vectors_normalized, query_vector_normalized):
    """Brute force in numpy: cosine with ALL 6000 vectors, keep the best TOP_K."""
    scores = doc_vectors_normalized @ query_vector_normalized
    best = np.argsort(scores)[::-1][:TOP_K]
    ids = []
    for position in best:
        ids.append(int(position))
    return ids


def find_cluster(ivf_index, doc_id):
    """Return the id of the IVF cluster that contains doc_id."""
    for cluster_id in ivf_index["lists"]:
        if doc_id in ivf_index["lists"][cluster_id]:
            return cluster_id
    return None


def centroid_rank(ivf_index, query_vector_normalized, cluster_id):
    """
    How close is this cluster's centroid to the query, as a rank?
    1 = the closest centroid, 2 = second closest, and so on.
    """
    centroid_scores = ivf_index["centroids"] @ query_vector_normalized
    order = np.argsort(centroid_scores)[::-1]      # best centroid first
    for position in range(len(order)):
        if order[position] == cluster_id:
            return position + 1
    return None


def run_ivf(ivf_index, doc_vectors_normalized, query_vector_normalized, qi, exact_ids):
    """
    Search our IVF index at every nprobe and compare with Qdrant's exact search.
    IVF runs inside Python (no network), so we also time a numpy brute-force
    search as a fair "exact" baseline for the IVF latency.
    """
    print_header(f"PART 4 | Custom IVF ({IVF_CLUSTERS} k-means clusters) vs exact")
    print(f"   exact ids (Qdrant): {exact_ids}")

    rows = []

    # ---- In-process exact baseline --------------------------------------
    numpy_ids, numpy_ms = timed(
        lambda: numpy_exact_search(doc_vectors_normalized, query_vector_normalized)
    )
    same = numpy_ids == exact_ids
    print(f"   numpy brute force   scanned={len(doc_vectors_normalized):<5} "
          f"overlap={overlap(numpy_ids, exact_ids):.1f}  latency={numpy_ms:6.3f} ms  "
          f"(same as Qdrant exact: {'yes' if same else 'NO'})")
    rows.append({
        "query_index": qi,
        "method": "Exact (numpy)",
        "setting": "scan all 6000",
        "ids": numpy_ids,
        "overlap": overlap(numpy_ids, exact_ids),
        "latency_ms": round(numpy_ms, 3),
        "scanned": len(doc_vectors_normalized),
    })

    # ---- IVF at each nprobe ---------------------------------------------
    for nprobe in IVF_NPROBE_VALUES:
        result, latency_ms = timed(
            lambda: search_ivf(ivf_index, doc_vectors_normalized, query_vector_normalized,
                               nprobe=nprobe, k=TOP_K)
        )
        score = overlap(result["ids"], exact_ids)

        # Which exact results did IVF miss? And WHY: which cluster is each
        # missed document in, and how close was that cluster's centroid?
        missed = []
        missed_details = []
        for doc_id in exact_ids:
            if doc_id not in result["ids"]:
                missed.append(doc_id)
                cluster_id = find_cluster(ivf_index, doc_id)
                rank = centroid_rank(ivf_index, query_vector_normalized, cluster_id)
                missed_details.append({"id": doc_id, "cluster": cluster_id,
                                       "cluster_rank": rank})

        line = (f"   IVF nprobe={nprobe:<3}      scanned={result['scanned']:<5} "
                f"overlap={score:.1f}  latency={latency_ms:6.3f} ms  ids={result['ids']}")
        if len(missed) > 0:
            line += f"  missed={missed}"
        print(line)
        for detail in missed_details:
            print(f"      -> missed id {detail['id']} lives in cluster {detail['cluster']}, "
                  f"the #{detail['cluster_rank']} closest cluster, but only "
                  f"{nprobe} cluster(s) were searched")

        rounded_scores = []
        for s in result["scores"]:
            rounded_scores.append(round(s, 4))

        rows.append({
            "query_index": qi,
            "method": "IVF",
            "setting": f"nprobe={nprobe}",
            "nprobe": nprobe,
            "ids": result["ids"],
            "scores": rounded_scores,
            "clusters_searched": result["clusters"],
            "missed_exact_ids": missed,
            "missed_details": missed_details,
            "overlap": score,
            "latency_ms": round(latency_ms, 3),
            "scanned": result["scanned"],
        })
    return rows


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
    print_index_status(client)

    # Build the IVF index once (k-means on all 6000 normalized vectors).
    print(f"\nBuilding IVF index ({IVF_CLUSTERS} clusters, k-means seed=42) ...")
    build_start = time.perf_counter()
    ivf_index = build_ivf_index(doc_vectors_normalized, n_clusters=IVF_CLUSTERS, seed=42)
    ivf_build_seconds = time.perf_counter() - build_start
    print(f"   built in {ivf_build_seconds:.2f} s")

    # Filled in below, one entry per query, and saved as JSON at the end.
    distance_results = []
    exact_rows = []
    hnsw_rows = []
    ivf_rows = []

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

        # ---- Part 3 ------------------------------------------------------
        exact_row = run_exact(client, query_vectors_raw[qi], qi)
        exact_rows.append(exact_row)
        exact_ids = exact_row["ids"]

        new_hnsw_rows = run_hnsw(client, query_vectors_raw[qi], qi, exact_ids)
        hnsw_rows.extend(new_hnsw_rows)

        # ---- Part 4 ------------------------------------------------------
        new_ivf_rows = run_ivf(ivf_index, doc_vectors_normalized,
                               query_vectors_normalized[qi], qi, exact_ids)
        ivf_rows.extend(new_ivf_rows)

    # ---- Averages over the queries that were run ------------------------
    print("\n" + "=" * 78)
    print(f"AVERAGES over {len(query_indexes)} query/queries")
    print("=" * 78)
    hnsw_summary = summarize(exact_rows + hnsw_rows)
    ivf_summary = summarize(ivf_rows)
    print_summary_table(hnsw_summary + ivf_summary)

    # ---- Save results (only for a full run, so a demo of one query -------
    # ---- does not overwrite the full results) ---------------------------
    if args.query is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "distance_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(distance_results, f, indent=2)
        print(f"\nSaved {RESULTS_DIR}/distance_metrics.json")

        hnsw_output = {
            "latency_runs_per_search": LATENCY_RUNS,
            "summary": hnsw_summary,
            "exact": exact_rows,
            "hnsw": hnsw_rows,
        }
        with open(os.path.join(RESULTS_DIR, "hnsw.json"), "w", encoding="utf-8") as f:
            json.dump(hnsw_output, f, indent=2)
        print(f"Saved {RESULTS_DIR}/hnsw.json")

        # Cluster sizes, to show how the 6000 vectors were split.
        cluster_sizes = {}
        for cluster_id in ivf_index["lists"]:
            cluster_sizes[cluster_id] = len(ivf_index["lists"][cluster_id])

        ivf_output = {
            "n_clusters": IVF_CLUSTERS,
            "build_seconds": round(ivf_build_seconds, 2),
            "cluster_sizes": cluster_sizes,
            "latency_runs_per_search": LATENCY_RUNS,
            "summary": ivf_summary,
            "results": ivf_rows,
        }
        with open(os.path.join(RESULTS_DIR, "ivf.json"), "w", encoding="utf-8") as f:
            json.dump(ivf_output, f, indent=2)
        print(f"Saved {RESULTS_DIR}/ivf.json")


if __name__ == "__main__":
    main()
