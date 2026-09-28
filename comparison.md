# Comparison of search methods

All numbers below come from the final full run of `python compare.py`
(saved in `results/`). Dataset: 6000 posts from 20 Newsgroups, embedded with
`all-MiniLM-L6-v2` (384 dims). 5 test queries from [queries.md](queries.md).

* **Overlap** = |method's top-5 ∩ exact top-5| / 5. 1.00 means the same 5 documents as exact search.
* **Latency** = average of 10 timed calls after 1 warm-up call, measured with
  `time.perf_counter()` in Python.
  Qdrant rows include the HTTP round-trip to the Docker container.
  IVF and "Exact (numpy)" run inside Python, so they should be compared with each other.

---

## 1. Combined table (averaged over the 5 queries)

Ground truth for every row: `capstone_cosine` searched with `SearchParams(exact=True)`.

| Method | Setting | Avg top-5 overlap vs exact | Avg latency ms |
|--------|---------|---------------------------|----------------|
| Exact (Qdrant) | exact=True, brute force | 1.00 | 3.244 |
| HNSW default | m=16, ef_construct=100, hnsw_ef=16 | 1.00 | 2.731 |
| HNSW default | m=16, ef_construct=100, hnsw_ef=64 | 1.00 | 2.798 |
| HNSW default | m=16, ef_construct=100, hnsw_ef=128 | 1.00 | 2.746 |
| HNSW weak | m=4, ef_construct=8, hnsw_ef=16 | 0.60 | 2.308 |
| HNSW weak | m=4, ef_construct=8, hnsw_ef=64 | 0.68 | 2.567 |
| HNSW weak | m=4, ef_construct=8, hnsw_ef=128 | 0.92 | 2.631 |
| Exact (numpy) | scan all 6000, in-process | 1.00 | 0.495 |
| IVF (40 clusters) | nprobe=1 (avg 151 vectors scanned) | 0.96 | 0.024 |
| IVF (40 clusters) | nprobe=4 (avg 619 scanned) | 1.00 | 0.129 |
| IVF (40 clusters) | nprobe=8 (avg 1158.6 scanned) | 1.00 | 0.219 |
| IVF (40 clusters) | nprobe=16 (avg 2287.8 scanned) | 1.00 | 0.345 |

## 2. More work → closer to exact

Both approximate methods moved toward the exact answer as they were allowed
to do more work. The weak HNSW graph (m=4, ef_construct=8) went from 0.60
average overlap at `hnsw_ef=16` to 0.68 at 64 and 0.92 at 128. `hnsw_ef` is
the size of the candidate list kept while walking the graph. A bigger list
explores more nodes, so the search is less likely to get stuck in the wrong
neighbourhood. Query 3 shows this most clearly: 0/5 correct at ef=16 and 64,
5/5 at ef=128. The default graph (m=16, ef_construct=100) already reached
1.00 at ef=16, because a well-connected graph needs much less search effort
to reach the true neighbours, so extra ef bought nothing here. On the weak
graph, query 4 stayed at 0.6 even at ef=128. Query-time effort cannot fully
repair a badly built graph.

IVF behaved the same way. With nprobe=1 it scanned only 151 vectors on
average (2.5% of the data) and still reached 0.96 overlap. With nprobe=4
(619 vectors) it matched exact search on every query; nprobe 8 and 16 only
added cost. Latency grew with the work: 0.024 → 0.129 → 0.219 → 0.345 ms,
against 0.495 ms for scanning all 6000 vectors.

This is the speed/accuracy trade-off every approximate index makes: accuracy
is bought by examining more candidates, and every candidate costs time. The
knob is different (ef for HNSW, nprobe for IVF) but the shape is the same.
At only 6000 documents the Qdrant latencies are dominated by the HTTP
round-trip (≈2.3–3.2 ms for every row), so the differences between ef values
are within measurement noise. The in-process IVF numbers show the cost
curve more clearly. On millions of vectors the gap between exact and
approximate search would be far larger.

## 3. Metric choice and index choice are separate decisions

Query 4, "encryption and government key escrow". `capstone_cosine` and
`capstone_dot` contain **identical raw vectors** with **identical HNSW settings**
(m=16, ef_construct=100), and both were searched with `exact=True`, so the index
plays no part at all. Still, the top result is different:

* COSINE #1: id 1409 (cosine 0.7068, vector length 1.969)
* DOT #1: id 4835 (dot 10.3862, vector length 3.416, but cosine only ≈ 0.425)

The metric decides **what "most similar" means**; the answer changed because
dot product rewards long vectors. Part 3 shows the opposite case: same metric,
same vectors, different index (default vs weak HNSW). There the definition of
"nearest" never changed. Only **how many of the true nearest neighbours were
found, and how fast**, changed. So you first choose the metric that matches
what similarity should mean for your data, and then choose and tune the index
(HNSW m/ef, IVF nprobe) to approximate that metric's exact answer cheaply
enough.

---

## Details – Part 2: distance metrics

Three collections hold **the same raw (unnormalized) vectors** and the same
payload. Only the distance metric differs:
`capstone_cosine` (COSINE), `capstone_euclid` (EUCLID), `capstone_dot` (DOT).
All three are searched with `exact=True`, so only the metric changes.

> **Reading the scores:** for **COSINE** and **DOT** the score is a similarity,
> so **higher = more similar**. For **EUCLID** Qdrant returns the actual
> distance, so **lower = closer**. Qdrant still lists the closest point first
> in every case.

How much euclid and dot agree with cosine (from `results/summary.json`):

| # | Query | euclid vs cosine: overlap / same #1 | dot vs cosine: overlap / same #1 |
|---|-------|-------------------------------------|----------------------------------|
| 0 | a question about a graphics card driver | 1.0 / yes (order changes) | 0.0 / no |
| 1 | debate about gun control laws | 0.6 / **no** | 0.2 / no |
| 2 | space shuttle launch and NASA missions | 0.8 / yes | 0.0 / no |
| 3 | hockey playoff predictions | 0.6 / **no** | 0.0 / no |
| 4 | encryption and government key escrow | 1.0 / yes (identical) | 0.4 / no |

### Query 4 in full: "encryption and government key escrow"

Query vector length = 7.147. `len` = length of the raw document vector.

| Rank | COSINE (higher better) | DOT (higher better) |
|------|------------------------|---------------------|
| 1 | id 1409, score 0.7068, len 1.969 | **id 4835**, score 10.3862, len **3.416** |
| 2 | id 3966, score 0.6934, len 1.875 | id 1409, score 9.9452, len 1.969 |
| 3 | id 4872, score 0.6401, len 1.872 | id 3966, score 9.2923, len 1.875 |
| 4 | id 5256, score 0.6090, len 1.835 | id 2171, score 9.1213, len 2.719 |
| 5 | id 5971, score 0.5597, len 1.993 | id 181, score 8.6421, len 3.033 |

EUCLID gave exactly the same list as COSINE for this query (distances 5.9211 … 6.253).

Why they disagree:

* **Cosine** looks only at the **angle** between query and document.
  Length is ignored. Document 1409 points most nearly in the query's direction
  (cosine 0.7068), so it wins.
* **Dot product** = cos(angle) × |query| × |document|. The query length is the
  same for every document, so the ranking is decided by **cos × document
  length**. Document 4835 has a weaker angle (cosine = 10.3862 / (7.147 × 3.416)
  ≈ 0.425) but its vector is 1.7× longer than 1409's (3.416 vs 1.969), so it is
  boosted to #1. Three of the five dot results (4835, 2171, 181) are long
  vectors that are not in the cosine top-5 at all.
* **Euclidean** distance is the straight-line distance between the two points,
  so it also depends on length:
  distance² = |q|² + |d|² − 2·|q|·|d|·cos.
  In query 1 ("debate about gun control laws", query length 6.901) cosine's #1
  is id 322 (len 1.63), but euclid's #1 is id 1425 (len 3.058). The query vector
  is much longer than most documents (~2), so a longer document sits physically
  closer to it: distance 5.4836 for 1425 vs 5.9489 for 322, even though 322 has
  the better angle (0.6616 vs 0.6374).

### With normalized vectors all three metrics agree

`compare.py` repeats the search in numpy on the **normalized** vectors (every
vector has length 1). For **all 5 queries** cosine, euclid and dot gave exactly
the same top-5 (`normalized_vectors_all_metrics_agree: true`). This is expected:

* dot = cos × 1 × 1 = cos, so dot and cosine are the same number;
* euclid² = 1 + 1 − 2·cos = 2 − 2·cos, so a smaller distance always means a
  larger cosine. Sorting by "smallest distance" gives the same order as
  sorting by "largest cosine".

So the metrics only disagree when vector **length** carries information.

## Details – Part 3: HNSW (default vs weak)

* **Default**: `capstone_cosine`, m=16, ef_construct=100.
  **Weak**: `capstone_hnsw_weak`, same vectors, m=4, ef_construct=8.
* The graphs were really built: `indexing_threshold` was lowered to 10 KB
  (the default of 20000 KB is larger than our ~9000 KB of vectors, and 0 would
  disable indexing). `qdrant_setup.py` and `compare.py` both print
  `indexed=6000/6000`, status green, before any search.

Weak-graph overlap per query (hnsw_ef = 16 / 64 / 128):

| Query | ef=16 | ef=64 | ef=128 |
|-------|-------|-------|--------|
| 0 | 1.0 | 1.0 | 1.0 |
| 1 | 0.6 | 0.8 | 1.0 |
| 2 | 0.8 | 1.0 | 1.0 |
| 3 | **0.0** | **0.0** | 1.0 |
| 4 | 0.6 | 0.6 | 0.6 |

The default graph scored 1.0 on every query at every ef.

## Details – Part 4: custom IVF (`ivf.py`)

* k-means, 40 clusters, seed 42, on the 6000 **normalized** vectors (so cosine =
  dot product). Build time 2.74 s. Cluster sizes 64 to 248.
* A numpy brute-force scan returned exactly the same top-5 as Qdrant's exact
  search for every query, which confirms both use the same metric.
* The only miss: **query 2** at nprobe=1 lost the true #1 (id 2079). That
  document lives in cluster 27, which is the **2nd** closest cluster to the
  query, so searching only the closest cluster never looked at it. With
  nprobe=4 the cluster is included and the top-5 matches exact search.

> Note: HNSW graph construction in Qdrant is not fully deterministic. If you
> re-run `qdrant_setup.py`, the weak-graph numbers can change a little.
> Latencies also vary run to run. The IVF clusters are fixed by the seed.
