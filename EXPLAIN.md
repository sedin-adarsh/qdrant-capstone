# EXPLAIN.md – study guide for the review

This file explains every script in simple English, gives model answers to the
four review questions using **my actual results**, and ends with the exact
demo script.

---

## Part A – The big picture in 6 sentences

1. I took 6000 newsgroup posts and turned each one into a list of 384 numbers
   (a **vector**) with the `all-MiniLM-L6-v2` model. Texts with similar meaning get vectors that point in similar directions.
2. I stored those vectors in **Qdrant** (a vector database running in Docker).
3. **Part 2:** I stored the same vectors three times with three different
   **distance metrics** (cosine, euclidean, dot) and checked whether the results change.
4. **Part 3:** I compared Qdrant's **exact search** (checks all 6000 vectors)
   with **HNSW** (a graph shortcut), once with a good graph and once with a weak graph.
5. **Part 4:** I wrote my own **IVF** index (group vectors into clusters, only
   search the closest clusters) and compared it with exact search too.
6. **Part 5:** `compare.py` runs everything and measures **overlap** (how many of
   the true top-5 were found) and **latency** (how long it took).

Key words:

* **Vector length (norm)**: sqrt(x1² + x2² + … + x384²). "Normalizing" = dividing a vector by its length so the length becomes 1.
* **Exact / brute-force search**: compare the query with every vector. Always correct, slow for big data.
* **Approximate (ANN) search**: HNSW, IVF. Only looks at some vectors, so it is fast but might miss some true neighbours.
* **Overlap@5**: |approx top-5 ∩ exact top-5| / 5. 1.0 = perfect.

---

## Part B – Walk through each file

### `docker-compose.yml`

* `image: qdrant/qdrant`: the official Qdrant server.
* `ports 6333` = REST API + dashboard, `6334` = gRPC API.
* `volumes ./qdrant_storage:/qdrant/storage:z`: the data is saved in a folder
  on my machine, so it survives a restart. `:z` is a SELinux label (harmless elsewhere).
* Start with `docker compose up -d` (`-d` = run in the background).

### `data.py` – load and sample the dataset

1. `fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))`
   downloads all ~18,800 posts. `remove` strips email headers, signatures and
   quoted replies, so only the author's own words remain. This matters because otherwise the model would match on headers or copied text.
2. The loop strips each text and **drops posts under 100 characters** (empty or
   near-empty posts give meaningless vectors). 17,080 posts survive.
3. `random.Random(42).sample(kept, 6000)` takes a random sample with a **fixed
   seed**, so everybody who runs it gets the same 6000 posts.
4. Each document becomes `{"id", "category", "text"}`. The **id is the
   position in the list**, and the same number is used as the row in the `.npy`
   files and as the Qdrant point id. That is how everything links together.
5. The `if __name__ == "__main__":` block only runs with `python data.py` and
   prints how many posts per category (about 300 each, all 20 categories).

### `embed.py` – make the vectors

1. `read_queries()` reads lines like `1. hockey playoff predictions` from
   `queries.md` and keeps the text after `"N. "`.
2. `SentenceTransformer("all-MiniLM-L6-v2")` loads the model.
3. **The important trick:** this model has 3 layers: Transformer →
   Pooling (average of word vectors) → **Normalize**. Because the last layer
   always normalizes, `normalize_embeddings=False` alone would *still* return
   length-1 vectors. I checked this: lengths were 1.0. So I build
   `raw_model = SentenceTransformer(modules=[model[0], model[1]])`, which is the same
   model without the Normalize layer. Same weights, nothing retrained.
4. Documents are encoded twice:
   * `model.encode(..., normalize_embeddings=True)` → `vectors_normalized.npy` (length 1).
   * `raw_model.encode(..., normalize_embeddings=False)` → `vectors_raw.npy`
     (natural length, **1.12 to 5.51**, mean 2.25).
5. `.astype(np.float32)`: Qdrant stores 32-bit floats; this also halves file size.
6. `metadata.json` saves id, category and text.
7. The 5 queries are embedded the same two ways (raw query lengths 5.87–7.41).
8. The final check `raw / length == normalized ? True` proves both files have
   the same **directions** and differ only in **length**.

### `qdrant_setup.py` – create the collections

* `DEFAULT_HNSW = HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10)`
  * `m` = how many neighbours each node links to in the graph.
  * `ef_construct` = how many candidates are checked when inserting a node while building. Bigger means better links.
  * `full_scan_threshold=10` (KB) stops Qdrant from deciding "this is small, just brute force".
* `WEAK_HNSW = HnswConfigDiff(m=4, ef_construct=8, ...)`: few links, careless build.
* `FORCE_INDEXING = OptimizersConfigDiff(indexing_threshold=10, default_segment_number=1)`
  * Qdrant only builds HNSW for a segment above `indexing_threshold` KB.
    Default = 20000 KB; my data = 6000 × 384 × 4 bytes ≈ 9000 KB. So **without
    this change Qdrant would never build HNSW** and would secretly brute-force.
  * `0` would mean "never index", so I use a small positive value (10).
* `create_collection(...)`:
  1. `collection_exists` → `delete_collection`: makes the script **safe to re-run**.
  2. `create_collection(vectors_config=VectorParams(size=384, distance=...))`.
  3. Upload in **batches of 256** `PointStruct(id, vector, payload={category, text})`
     with `upsert(wait=True)`. Batching avoids one giant request.
* `wait_until_indexed(...)`: polls `get_collection` every second until
  `status == GREEN` **and** `indexed_vectors_count == points_count` (6000/6000),
  then prints it. This is the **proof that HNSW was really built**.
* `main()`: loads the **raw** vectors, creates `capstone_cosine`,
  `capstone_euclid`, `capstone_dot` (default HNSW) and `capstone_hnsw_weak`
  (cosine, weak HNSW), then waits for all four.

Why raw vectors for all collections? With normalized vectors the three metrics
always give the same ranking (see Question 2), so nothing interesting would happen.
For `capstone_cosine`, Qdrant normalizes the vectors itself, so its results are
exactly the same as using the normalized file.

### `ivf.py` – my own IVF index

`build_ivf_index(vectors, n_clusters=40, seed=42)`:
1. `KMeans(n_clusters=40, random_state=42, n_init=10).fit(vectors)` groups the 6000 vectors into 40 clusters. `n_init=10` tries 10 random starts and keeps the best.
2. `labels_[i]` = which cluster vector i belongs to.
3. Centroids (cluster centres) are averages, so they are a bit shorter than 1.
   I normalize them so `centroid @ query` is a real cosine.
4. `lists[cluster_id] = np.where(labels == cluster_id)[0]`: the **inverted
   list**, which holds the ids of the vectors in each cluster.

`search_ivf(index, vectors, query_vector, nprobe, k=5)`:
1. `centroids @ query_vector`: 40 cosine scores (cheap).
2. `np.argsort(...)[::-1][:nprobe]`: sort high→low, keep the best `nprobe` clusters.
3. `np.concatenate` the id lists of those clusters = **candidates**.
4. `vectors[candidate_ids] @ query_vector`: exact cosine, but only for candidates
   (works because all vectors are normalized, so dot = cosine).
5. Keep the best `k`, turn positions back into real ids.
6. Return ids, scores, `scanned` (= how many vectors were compared = the work done),
   and which clusters were searched.

### `compare.py` – the experiment runner (and live demo)

* **Top of the file:** `OMP/OPENBLAS/MKL_NUM_THREADS = "1"` must be set **before**
  `import numpy`. One query is tiny maths; with 18 threads numpy spent more time
  managing threads than calculating (similar searches jumped between ~0.1 and ~15 ms)
  and stole CPU from Qdrant. One thread gives stable timings.
* **Constants:** `TOP_K=5`, `LATENCY_RUNS=10`, `HNSW_EF_VALUES=[16,64,128]`,
  `IVF_NPROBE_VALUES=[1,4,8,16]`, and `METRICS` with a note that **EUCLID
  scores are distances (lower = closer)** while COSINE/DOT are similarities (higher = closer).
* **Helpers:**
  * `qdrant_search()` calls `client.query_points(...)` (the current API; `search()` is deprecated) and returns id, category, score, snippet.
  * `overlap()` = `len(set(a) & set(b)) / 5`.
  * `timed(function)` makes 1 warm-up call, then 10 calls timed with `time.perf_counter()`, and returns the average in ms. The search is passed as a `lambda` so it can be called repeatedly.
  * `summarize()` groups rows by (method, setting) and averages overlap and latency over the queries.
* **Part 2 `run_distance_metrics()`:** the same raw query goes to the three
  collections with `exact=True`. It is exact on purpose, so HNSW approximation
  cannot mix into the metric comparison (I first ran it without `exact=True`, and one
  euclid result changed after the graphs were rebuilt; that was an index effect, not a metric effect).
  It prints each hit's vector length and how much euclid/dot agree with cosine.
  `normalized_metrics_agree()` repeats it in numpy with normalized vectors as a control.
* **Part 3:**
  * `print_index_status()` shows `indexed=6000/6000` for both HNSW collections.
  * `run_exact()` uses `SearchParams(exact=True)` on `capstone_cosine`. This is the **ground truth**.
  * `run_hnsw()` loops over both collections × `hnsw_ef` 16/64/128 with `SearchParams(hnsw_ef=ef, exact=False)` and records overlap and latency.
* **Part 4:**
  * `numpy_exact_search()` scores all 6000 vectors in-process. This is the fair latency baseline for IVF and also checks it matches Qdrant exact (it does).
  * `run_ivf()` runs each nprobe. For a missed document it prints its cluster
    and that cluster's rank (`find_cluster`, `centroid_rank`), which explains *why* it was missed.
* **main():**
  1. Reads `--query`.
  2. Loads the data.
  3. Checks the HNSW index.
  4. Builds IVF once.
  5. Loops over the queries (Parts 2–4).
  6. Prints the Part 5 summary.
  7. Writes `results/*.json`, only on a full run, so a one-query demo doesn't overwrite them.

---

## Part C – Model answers to the 4 review questions

### Q1. Why does my IVF top-5 sometimes disagree with exact search at low nprobe, and how does raising nprobe close the gap?

IVF only compares the query with vectors inside the `nprobe` clusters whose
**centroids** are closest. But a centroid is just the average of its cluster.
A document can be very close to the query while sitting in a *neighbouring*
cluster, for example near the border between two clusters. If that cluster is
not among the `nprobe` searched, the document is never even looked at, so it
cannot appear in the results.

My real example: **query 2, "space shuttle launch and NASA missions"**. Exact
top-5 = `[2079, 1898, 5623, 2137, 1177]`. With **nprobe=1** IVF returned
`[1898, 5623, 2137, 1177, 3497]`: it **lost the true #1, id 2079**.
`compare.py` shows why: 2079 lives in **cluster 27, the 2nd closest
cluster**, and nprobe=1 searched only the closest one. With **nprobe=4** the
cluster was included and the top-5 matched exact search.

Raising nprobe searches more neighbouring clusters, so the chance that a true
neighbour sits in an unsearched cluster falls. The cost is more work: average
vectors scanned went 151 → 619 → 1158.6 → 2287.8 for nprobe 1/4/8/16, and
latency 0.024 → 0.129 → 0.219 → 0.345 ms. Average overlap went 0.96 → 1.00.
If nprobe = all 40 clusters, IVF *is* exact search.

### Q2. Why can the same query give a different top result under cosine vs dot product with identical vectors?

Because of **vector length (magnitude)**.
* cosine(q, d) = (q · d) / (|q| × |d|) measures only the **angle**. Length is divided out.
* dot(q, d) = |q| × |d| × cos(angle). For one query, |q| is fixed, so the ranking
  is by **cos × |d|**. A document with a slightly worse angle but a longer vector can win.

My real example: **query 4, "encryption and government key escrow"**
(both collections hold identical raw vectors, both searched exactly):
* COSINE #1 = **id 1409**: cosine 0.7068, length 1.969.
* DOT #1 = **id 4835**: dot 10.3862, length **3.416**. Its cosine is only
  10.3862 / (7.147 × 3.416) ≈ 0.425, but it is 1.7× longer than 1409, so dot ranks it first.

Across all 5 queries, dot shared only 0–2 of cosine's top-5 (overlap 0.0, 0.2,
0.0, 0.0, 0.4) and never had the same #1.

If the vectors are **normalized** (all length 1), dot = cosine exactly, so
the rankings are identical. My numpy control confirmed this for all 5 queries
(euclid too, because euclid² = 2 − 2·cos for unit vectors).

### Q3. What single change would make the weak HNSW behave like the default one, and why?

There are two places to change it:

**Build time (the real fix): raise `m` (and `ef_construct`) and rebuild.**
`m=4` means each node has only 4 links, so the graph is sparse and some regions are
badly connected. `ef_construct=8` means that while building, each new node
looked at only 8 candidates to choose its neighbours, so many links are to
not-quite-nearest nodes. The greedy search can then walk into a dead end. Raising
**`m` to 16** is the single most important change: more links = more
routes to the true neighbours. Raising `ef_construct` to ~100 makes those links
better quality. Setting both to the defaults *is* the default collection, which scored
**1.00 at every ef**. Cost: a bigger graph (more memory) and a slower build.
(I did not test changing `m` alone. Both were changed together in my experiment.)

**Query time (a partial fix): raise `hnsw_ef`.** This keeps a bigger candidate
list while walking the graph, so the search explores more and gets stuck
less often. No rebuild needed, but every query gets slower. In my results it
raised the weak graph from **0.60 (ef=16) → 0.68 (ef=64) → 0.92 (ef=128)**.
Query 3 went from **0/5 to 5/5** at ef=128. But **query 4 stayed at 0.6 even at ef=128**:
more searching cannot fully make up for links that are missing from the graph.

So: `hnsw_ef` = search the bad graph harder (per query, cheap to change);
`m`/`ef_construct` = build a better graph (one-time cost, fixes the cause).

### Q4. When would a real system want IVF instead of HNSW?

* **Huge datasets / memory limits:** HNSW keeps a graph with `m` links per vector,
  normally in RAM. IVF only stores centroids + id lists, so it is much smaller.
* **Combined with quantization (IVF-PQ):** vectors inside each cluster can be
  compressed with product quantization to a few bytes each. This is how
  billion-scale search fits in memory (e.g. FAISS IVF-PQ).
* **Fast builds / frequent rebuilds:** k-means + assigning vectors is fast and
  easy to redo (my build: 2.74 s). HNSW insertion is slower, especially with
  high `ef_construct`. Good when the data is rebuilt nightly.
* **Disk and GPU friendly:** each cluster is one contiguous block, so you can read a
  few clusters from disk, or do one big matrix multiply per cluster on a GPU.
  HNSW jumps around memory randomly, which is bad for disk and GPUs.
* **Simple, predictable cost:** work is roughly (nprobe / n_clusters) × N and
  easy to tune with a single knob.

HNSW is usually better when everything fits in RAM and you want the best
recall at very low latency, like Qdrant's default setup for my 6000 docs.

---

## Part D – Live demo script

Run these in order in a WSL/Linux terminal inside the project folder.

```bash
# 1. Start Qdrant and show it is running
docker compose up -d
curl http://localhost:6333/collections
#    -> lists capstone_cosine, capstone_euclid, capstone_dot, capstone_hnsw_weak

# 2. Activate the Python environment
source .venv/bin/activate

# (Only on a fresh machine, before the demo, NOT during it:)
#    python embed.py          # ~2-3 min, creates data/
#    python qdrant_setup.py   # creates the 4 collections, prints indexed 6000/6000
#    python compare.py        # full run, writes results/

# 3. The best single query: shows ALL effects
python compare.py --query 3
#    Point out, in order:
#    - HNSW index check: indexed=6000/6000 (HNSW really built)
#    - Part 2: dot shares 0 of cosine's top-5; long vectors (len 3.1-4.5) win
#      under dot; EUCLID score is a distance (lower = closer)
#    - Part 3: weak HNSW overlap 0.0 at ef=16 and 64, then 1.0 at ef=128;
#      default HNSW is 1.0 everywhere
#    - Part 4: IVF scans ~200 of 6000 vectors and still gets 1.0

# 4. The IVF miss
python compare.py --query 2
#    - IVF nprobe=1 misses id 2079 -> "lives in the #2 closest cluster"
#    - nprobe=4 fixes it

# 5. The cosine vs dot example
python compare.py --query 4
#    - cosine #1 = 1409 (len 1.969); dot #1 = 4835 (len 3.416)

# 6. Show the combined table and write-up
cat comparison.md     # or open it in the editor

# 7. Show the dashboard
#    open http://localhost:6333/dashboard in the browser
```

Tip: don't re-run `qdrant_setup.py` right before the demo. HNSW builds are
not fully deterministic, so the weak-graph numbers might shift a little from
the ones in `comparison.md`.
