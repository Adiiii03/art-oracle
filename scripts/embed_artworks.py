"""
Embed all artworks from data/met_artworks.json into a ChromaDB vector store.

Each artwork is embedded using sentence-transformers (all-MiniLM-L6-v2) based on
a rich text description built from its metadata. The resulting ChromaDB collection
lives in embeddings/artworks_db and can be queried semantically.
"""

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
ARTWORKS_FILE = DATA_DIR / "met_artworks.json"
COLLECTION_NAME = "artworks"
BATCH_SIZE = 64


def _s(val) -> str:
    """Safely convert a value to a stripped string."""
    return (val or "").strip() if isinstance(val, (str, type(None))) else str(val).strip()


def build_description(artwork: dict) -> str:
    """Build a rich text description from artwork metadata for embedding."""
    parts = []

    title = _s(artwork.get("title"))
    if title:
        parts.append(title)

    artist = _s(artwork.get("artistDisplayName"))
    if artist:
        nationality = _s(artwork.get("artistNationality"))
        if nationality:
            parts.append(f"by {artist} ({nationality})")
        else:
            parts.append(f"by {artist}")

    date = _s(artwork.get("objectDate"))
    if date:
        parts.append(f"created {date}")

    medium = _s(artwork.get("medium"))
    if medium:
        parts.append(f"Medium: {medium}")

    culture = _s(artwork.get("culture"))
    if culture:
        parts.append(f"Culture: {culture}")

    period = _s(artwork.get("period"))
    if period:
        parts.append(f"Period: {period}")

    department = _s(artwork.get("department"))
    if department:
        parts.append(f"Department: {department}")

    classification = _s(artwork.get("classification"))
    if classification:
        parts.append(f"Classification: {classification}")

    tags = artwork.get("tags") or []
    if tags:
        parts.append(f"Tags: {', '.join(str(t) for t in tags)}")

    return ". ".join(parts) if parts else "Unknown artwork"


def main():
    if not ARTWORKS_FILE.exists():
        print(f"Error: {ARTWORKS_FILE} not found. Run scrape_combined.py first.")
        sys.exit(1)

    print("Loading artworks...")
    artworks = json.loads(ARTWORKS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(artworks)} artworks")

    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Initializing ChromaDB at {EMBEDDINGS_DIR}...")
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR / "artworks_db"))

    # Delete existing collection if present to rebuild from scratch
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection, rebuilding...")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Art Oracle artwork embeddings"},
    )

    # Build descriptions and embed in batches
    descriptions = []
    ids = []
    metadatas = []

    for i, artwork in enumerate(artworks):
        desc = build_description(artwork)
        descriptions.append(desc)
        ids.append(str(artwork.get("objectID", i)))
        metadatas.append({
            "title": (artwork.get("title") or "")[:500],
            "artist": (artwork.get("artistDisplayName") or "")[:200],
            "date": (artwork.get("objectDate") or "")[:100],
            "medium": (artwork.get("medium") or "")[:300],
            "department": (artwork.get("department") or "")[:100],
            "classification": (artwork.get("classification") or "")[:100],
            "culture": (artwork.get("culture") or "")[:200],
            "source": artwork.get("source", "unknown"),
            "objectURL": (artwork.get("objectURL") or "")[:500],
        })

    print(f"Embedding {len(descriptions)} artwork descriptions...")

    for start in range(0, len(descriptions), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(descriptions))
        batch_descs = descriptions[start:end]
        batch_ids = ids[start:end]
        batch_metas = metadatas[start:end]

        embeddings = model.encode(batch_descs, show_progress_bar=False).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_descs,
            metadatas=batch_metas,
        )

        progress = end
        print(f"  [{progress}/{len(descriptions)}] embedded")

    # Verify
    count = collection.count()
    print(f"\nDone! ChromaDB collection '{COLLECTION_NAME}' has {count} entries.")
    print(f"Database stored at: {EMBEDDINGS_DIR / 'artworks_db'}")

    # Quick sanity test: query for "impressionist landscape painting"
    print("\n--- Sanity check: querying 'impressionist landscape painting' ---")
    test_embedding = model.encode(["impressionist landscape painting"]).tolist()
    results = collection.query(query_embeddings=test_embedding, n_results=3)
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  > {meta.get('title', 'Untitled')} by {meta.get('artist', 'Unknown')}")
        print(f"    {doc[:120]}...")


if __name__ == "__main__":
    main()
