# Comparison of search methods

All numbers below come from `python compare.py` (saved in `results/`).
Dataset: 6000 posts from 20 Newsgroups, embedded with `all-MiniLM-L6-v2` (384 dims).

## Part 2 – Distance metrics

Three collections hold **the same raw (unnormalized) vectors** and the same
payload. Only the distance metric is different:
`capstone_cosine` (COSINE), `capstone_euclid` (EUCLID), `capstone_dot` (DOT).

> **Reading the scores:** for **COSINE** and **DOT** the score is a similarity,
> so **higher = more similar**. For **EUCLID** Qdrant returns the actual
> distance, so **lower = closer**. Qdrant still returns the closest point first
> in every case.

How often the top-5 changed compared with cosine (from `results/distance_metrics.json`):

| # | Query | euclid vs cosine: overlap / same #1 | dot vs cosine: overlap / same #1 |
|---|-------|-------------------------------------|----------------------------------|
| 0 | a question about a graphics card driver | 1.0 / yes (order changes) | 0.0 / no |
| 1 | debate about gun control laws | 0.8 / **no** | 0.2 / no |
| 2 | space shuttle launch and NASA missions | 0.8 / yes | 0.0 / no |
| 3 | hockey playoff predictions | 0.6 / **no** | 0.0 / no |
| 4 | encryption and government key escrow | 1.0 / yes (identical) | 0.4 / no |

### Example where the metric changes the ranking: query 4, "encryption and government key escrow"

Query vector length = 7.147. `len` = length of the raw document vector.

| Rank | COSINE (higher better) | DOT (higher better) |
|------|------------------------|---------------------|
| 1 | id 1409, score 0.7068, len 1.969 | **id 4835**, score 10.3862, len **3.416** |
| 2 | id 3966, score 0.6934, len 1.875 | id 1409, score 9.9452, len 1.969 |
| 3 | id 4872, score 0.6401, len 1.872 | id 3966, score 9.2923, len 1.875 |
| 4 | id 5256, score 0.6090, len 1.835 | id 2171, score 9.1213, len 2.719 |
| 5 | id 5971, score 0.5597, len 1.993 | id 181, score 8.6421, len 3.033 |

Why they disagree:

* **Cosine** looks only at the **angle** between the query and the document.
  Length is ignored. Document 1409 points most nearly in the same direction as
  the query (cosine 0.7068), so it wins.
* **Dot product** = cos(angle) × |query| × |document|. The query length is the
  same for every document, so the ranking is decided by **cos × document
  length**. Document 4835 has a weaker angle (cosine = 10.3862 / (7.147 × 3.416)
  ≈ 0.425) but its vector is 1.7× longer than 1409's (3.416 vs 1.969), so it is
  boosted to #1. Three of the five dot results (4835, 2171, 181) are long
  vectors that are not in the cosine top-5 at all.
* **Euclidean** distance is the straight-line distance between the two
  points, so it is also affected by length:
  distance² = |q|² + |d|² − 2·|q|·|d|·cos.
  In query 1 ("debate about gun control laws", query length 6.901) cosine's
  #1 is id 322 (len 1.63), but euclid's #1 is id 1425 (len 3.058). Because the
  query vector is much longer than most documents (~2), a longer document sits
  physically closer to it: 5.484 for 1425 vs 5.949 for 322.

### With normalized vectors all three metrics agree

`compare.py` also repeats the search in numpy on the **normalized** vectors
(every vector has length 1). For **all 5 queries** cosine, euclid and dot gave
exactly the same top-5 (`normalized_vectors_all_metrics_agree: true` in the JSON).
This is expected:

* dot = cos × 1 × 1 = cos, so dot and cosine are the same number;
* euclid² = 1 + 1 − 2·cos = 2 − 2·cos, so a smaller distance always means a
  larger cosine. Sorting by "smallest distance" gives the same order as
  sorting by "largest cosine".

So the metrics only disagree when vector **length** carries information. With
normalized embeddings the choice between them does not change results.
