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
