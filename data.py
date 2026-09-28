"""
data.py - Load the 20 Newsgroups dataset and take a random sample.

Run on its own to see a quick summary:
    python data.py
"""

import random

from sklearn.datasets import fetch_20newsgroups

# Settings for the sample. Keeping them here makes them easy to find and change.
SAMPLE_SIZE = 6000   # how many documents we keep
RANDOM_SEED = 42     # fixed seed, so every run picks the SAME 6000 documents
MIN_CHARS = 100      # documents shorter than this are dropped (too little text to embed)


def load_documents(sample_size=SAMPLE_SIZE, seed=RANDOM_SEED, min_chars=MIN_CHARS):
    """
    Returns a list of dictionaries, one per document:
        {"id": 0, "category": "sci.space", "text": "..."}
    The id is simply the position in the list (0, 1, 2, ...). We use the same
    id as the Qdrant point id and as the row number in the .npy vector files.
    """

    # 1) Download (first time only, then cached) the full dataset.
    #    subset="all" gives train + test together (~18,800 posts).
    #    remove=(...) strips email headers, signatures and quoted replies,
    #    so the model sees only the text the author actually wrote.
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )

    # dataset.data         -> list of post texts
    # dataset.target       -> list of category numbers (0..19)
    # dataset.target_names -> list that turns a category number into its name
    print(f"Loaded {len(dataset.data)} posts from 20 Newsgroups")

    # 2) Keep only documents that have enough text.
    kept = []
    for i in range(len(dataset.data)):
        text = dataset.data[i].strip()          # remove spaces/newlines at both ends
        if len(text) < min_chars:
            continue                            # skip empty or very short posts
        category_number = dataset.target[i]
        category_name = dataset.target_names[category_number]
        kept.append({"category": category_name, "text": text})

    print(f"Kept {len(kept)} posts with at least {min_chars} characters")

    # 3) Take a random sample with a fixed seed (reproducible).
    #    random.Random(seed) makes a private random generator, so the sample
    #    does not depend on any other code that uses random numbers.
    rng = random.Random(seed)
    sample = rng.sample(kept, sample_size)

    # 4) Give every sampled document an id equal to its position.
    documents = []
    for new_id in range(len(sample)):
        documents.append({
            "id": new_id,
            "category": sample[new_id]["category"],
            "text": sample[new_id]["text"],
        })

    print(f"Sampled {len(documents)} documents (seed={seed})")
    return documents


# This block only runs when you type "python data.py",
# not when another script does "import data".
if __name__ == "__main__":
    docs = load_documents()

    # Count how many documents each category has in the sample.
    counts = {}
    for doc in docs:
        counts[doc["category"]] = counts.get(doc["category"], 0) + 1

    print("\nDocuments per category:")
    for category in sorted(counts):
        print(f"  {category:<28} {counts[category]}")

    print("\nExample document:")
    print(f"  id={docs[0]['id']}  category={docs[0]['category']}")
    print("  " + docs[0]["text"][:200].replace("\n", " ") + " ...")
