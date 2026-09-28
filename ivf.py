"""
ivf.py - A small, hand-written IVF (Inverted File) index in plain numpy.

Idea of IVF:
  BUILD:  Group all vectors into n_clusters groups with k-means.
          Each group has a centre point called a "centroid".
          For every cluster we remember which vector ids belong to it
          (this list is the "inverted file" / "inverted list").
  SEARCH: 1) Compare the query with the centroids only (40 comparisons, cheap).
          2) Pick the nprobe closest clusters.
          3) Compare the query exactly with the vectors inside those clusters only.
          4) Return the best k.
  More nprobe  = more clusters searched = slower but closer to exact search.

We use NORMALIZED vectors (length 1). For length-1 vectors,
    cosine similarity(a, b) = dot product(a, b)
so a simple matrix multiplication gives us exact cosine scores,
the same metric that Qdrant's capstone_cosine collection uses.

Run on its own for a quick test:
    python ivf.py
"""

import numpy as np
from sklearn.cluster import KMeans


def build_ivf_index(vectors, n_clusters=40, seed=42):
    """
    vectors    : numpy array, shape (number_of_vectors, 384), normalized
    n_clusters : how many clusters (inverted lists) to make
    seed       : fixed random seed so k-means gives the same clusters every run

    Returns a dictionary:
        {
          "centroids": numpy array, shape (n_clusters, 384),
          "lists":     {cluster_id: numpy array of vector ids in that cluster}
        }
    """

    # 1) Run k-means. n_init=10 means k-means is tried 10 times from different
    #    random starts and the best result is kept.
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(vectors)

    # labels[i] is the cluster number that vector i was assigned to.
    labels = kmeans.labels_

    # 2) k-means centroids are averages of normalized vectors, so they are
    #    slightly shorter than length 1. We normalize them too, so that
    #    "dot product with query" is the cosine similarity to the centroid.
    centroids = kmeans.cluster_centers_
    centroid_lengths = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / centroid_lengths

    # 3) Build the inverted lists: for each cluster, which vector ids are in it.
    lists = {}
    for cluster_id in range(n_clusters):
        # np.where returns the positions where the condition is True.
        ids_in_cluster = np.where(labels == cluster_id)[0]
        lists[cluster_id] = ids_in_cluster

    return {"centroids": centroids.astype(np.float32), "lists": lists}


def search_ivf(index, vectors, query_vector, nprobe, k=5):
    """
    index        : the dictionary returned by build_ivf_index
    vectors      : the same normalized vectors used to build the index
    query_vector : one normalized query vector, shape (384,)
    nprobe       : how many of the closest clusters to search
    k            : how many results to return

    Returns a dictionary:
        {
          "ids":      list of the top-k vector ids (best first),
          "scores":   list of their cosine similarity scores,
          "scanned":  how many document vectors we actually compared with the query,
          "clusters": which cluster ids we searched
        }
    """

    centroids = index["centroids"]
    lists = index["lists"]

    # ---- Step 1: score the query against every centroid -----------------
    # centroids is (n_clusters, 384) and query_vector is (384,)
    # so the result is one cosine score per cluster, shape (n_clusters,)
    centroid_scores = centroids @ query_vector

    # ---- Step 2: pick the nprobe clusters with the HIGHEST score --------
    # np.argsort sorts from low to high, so we reverse it with [::-1]
    # to get high to low, then keep the first nprobe cluster ids.
    order = np.argsort(centroid_scores)[::-1]
    chosen_clusters = order[:nprobe]

    # ---- Step 3: collect the ids of all vectors in the chosen clusters --
    candidate_id_lists = []
    for cluster_id in chosen_clusters:
        candidate_id_lists.append(lists[int(cluster_id)])
    candidate_ids = np.concatenate(candidate_id_lists)

    # ---- Step 4: exact cosine similarity, but only for the candidates ---
    candidate_vectors = vectors[candidate_ids]           # shape (n_candidates, 384)
    candidate_scores = candidate_vectors @ query_vector  # shape (n_candidates,)

    # ---- Step 5: keep the best k candidates -----------------------------
    best_positions = np.argsort(candidate_scores)[::-1][:k]

    top_ids = []
    top_scores = []
    for position in best_positions:
        top_ids.append(int(candidate_ids[position]))        # position -> real vector id
        top_scores.append(float(candidate_scores[position]))

    return {
        "ids": top_ids,
        "scores": top_scores,
        "scanned": int(len(candidate_ids)),   # how much work we did
        "clusters": [int(c) for c in chosen_clusters],
    }


# Quick self-test: python ivf.py
if __name__ == "__main__":
    vectors = np.load("data/vectors_normalized.npy")
    queries = np.load("data/query_vectors_normalized.npy")

    print(f"Building IVF index on {len(vectors)} vectors ...")
    index = build_ivf_index(vectors)

    sizes = []
    for cluster_id in index["lists"]:
        sizes.append(len(index["lists"][cluster_id]))
    print(f"{len(sizes)} clusters, size min={min(sizes)} max={max(sizes)} avg={np.mean(sizes):.0f}")

    # Brute force over ALL vectors, for comparison.
    exact_scores = vectors @ queries[0]
    exact_top = np.argsort(exact_scores)[::-1][:5]
    print(f"\nQuery 0 exact top-5 ids: {exact_top.tolist()}")

    for nprobe in [1, 8]:
        result = search_ivf(index, vectors, queries[0], nprobe=nprobe, k=5)
        print(f"nprobe={nprobe:<2} ids={result['ids']} scanned={result['scanned']}")
