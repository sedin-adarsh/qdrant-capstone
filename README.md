# Compare Search Algorithms in Qdrant

Capstone project comparing distance metrics (cosine / euclidean / dot),
HNSW settings (default vs weak, different `hnsw_ef`) and a hand-written IVF
index, all measured against Qdrant's exact (brute-force) search.

## 0. Start Qdrant (one command)

```bash
docker compose up -d
```

Check that it is running:

```bash
curl http://localhost:6333/collections
# -> {"result":{"collections":[]},"status":"ok",...}
```

Then open the dashboard in a browser: <http://localhost:6333/dashboard>

> **Note:** take a screenshot of the dashboard for the submission. It is most
> useful after step 3 below, when the collections (`capstone_cosine`,
> `capstone_euclid`, `capstone_dot`, `capstone_hnsw_weak`) are visible.

Stop Qdrant with `docker compose down` (the data stays in `./qdrant_storage`).

## 1. Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
# optional, avoids downloading the big GPU build of PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. Dataset and embeddings

```bash
python data.py    # optional: shows the 6000-document sample per category
python embed.py   # downloads the model + dataset, writes everything into data/
```

`embed.py` creates `data/vectors_normalized.npy`, `data/vectors_raw.npy`,
`data/metadata.json`, `data/queries.json` and the two query-vector files.
The queries come from [queries.md](queries.md).

Note: `all-MiniLM-L6-v2` ends with a built-in `Normalize` layer, so
`normalize_embeddings=False` alone would still give length-1 vectors. For the
raw vectors, `embed.py` uses the model without that last layer.

## 3. Create the Qdrant collections

```bash
python qdrant_setup.py
```

Deletes and recreates every collection (safe to re-run), uploads the vectors in
batches, then waits until Qdrant has actually built the HNSW index
(`indexed_vectors_count == points_count`). This is a good moment for the
dashboard screenshot.

## 4. Run the comparison

```bash
python compare.py            # all 5 queries, writes results/*.json
python compare.py --query 0  # one query only (for the demo; does not overwrite results/)
```

Output files:

| File | Contents |
|------|----------|
| `results/distance_metrics.json` | Part 2: top-5 per query for cosine / euclid / dot (id, category, score, snippet, vector length) |
| `results/hnsw.json` | Part 3: exact search + default/weak HNSW at hnsw_ef 16/64/128 (ids, overlap, latency) |
| `results/ivf.json` | Part 4: IVF at nprobe 1/4/8/16 (ids, scores, overlap, latency, vectors scanned) |
| `results/summary.json` | Part 5: the combined averaged table |

Findings are written up in [comparison.md](comparison.md). A study guide and
the live-demo script are in [EXPLAIN.md](EXPLAIN.md).
