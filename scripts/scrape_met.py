"""
Scrape artwork metadata from the Metropolitan Museum of Art Open Access API.
Fetches 1000 artworks with images and saves to data/met_artworks.json.

API docs: https://metmuseum.github.io/
"""

import json
import os
import random
import time
from pathlib import Path

import requests

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "met_artworks.json"
TARGET_COUNT = 1000
BATCH_SIZE = 50


def get_object_ids(department_ids=None):
    """Fetch all object IDs that have images (isHighlight or hasImages)."""
    params = {"hasImages": "true"}
    if department_ids:
        params["departmentIds"] = "|".join(str(d) for d in department_ids)

    resp = requests.get(f"{BASE_URL}/search", params={**params, "q": "*"})
    resp.raise_for_status()
    data = resp.json()
    return data.get("objectIDs", [])


def fetch_object(object_id):
    """Fetch a single object's metadata."""
    resp = requests.get(f"{BASE_URL}/objects/{object_id}")
    if resp.status_code in (403, 404):
        return None
    resp.raise_for_status()
    return resp.json()


def extract_fields(obj):
    """Extract the fields we care about from a raw API object."""
    return {
        "objectID": obj.get("objectID"),
        "title": obj.get("title", ""),
        "artistDisplayName": obj.get("artistDisplayName", ""),
        "artistNationality": obj.get("artistNationality", ""),
        "objectDate": obj.get("objectDate", ""),
        "medium": obj.get("medium", ""),
        "dimensions": obj.get("dimensions", ""),
        "department": obj.get("department", ""),
        "classification": obj.get("classification", ""),
        "culture": obj.get("culture", ""),
        "period": obj.get("period", ""),
        "dynasty": obj.get("dynasty", ""),
        "primaryImage": obj.get("primaryImage", ""),
        "primaryImageSmall": obj.get("primaryImageSmall", ""),
        "objectURL": obj.get("objectURL", ""),
        "tags": [t["term"] for t in (obj.get("tags") or []) if "term" in t],
        "isPublicDomain": obj.get("isPublicDomain", False),
    }


def scrape(target=TARGET_COUNT):
    """Main scrape loop: fetch object IDs, sample, and download metadata."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from existing file if present
    existing = []
    seen_ids = set()
    if OUTPUT_FILE.exists():
        existing = json.loads(OUTPUT_FILE.read_text())
        seen_ids = {a["objectID"] for a in existing}
        print(f"Resuming: {len(existing)} artworks already saved.")

    if len(existing) >= target:
        print(f"Already have {len(existing)} artworks. Done.")
        return existing

    print("Fetching object IDs from Met API...")
    all_ids = get_object_ids()
    print(f"Found {len(all_ids)} objects with images.")

    # Shuffle and filter out already-fetched IDs
    candidates = [oid for oid in all_ids if oid not in seen_ids]
    random.shuffle(candidates)

    artworks = list(existing)
    failures = 0

    for i, oid in enumerate(candidates):
        if len(artworks) >= target:
            break

        try:
            obj = fetch_object(oid)
            if obj and obj.get("primaryImage"):
                artworks.append(extract_fields(obj))

                if len(artworks) % BATCH_SIZE == 0:
                    OUTPUT_FILE.write_text(json.dumps(artworks, indent=2))
                    print(f"  [{len(artworks)}/{target}] saved checkpoint")

        except requests.RequestException as e:
            failures += 1
            print(f"  Warning: failed to fetch {oid}: {e}")
            if failures > 50:
                print("Too many failures, stopping early.")
                break

        # Rate limiting: ~80 requests/min to be polite
        if i % 10 == 0:
            time.sleep(0.5)

    # Final save
    OUTPUT_FILE.write_text(json.dumps(artworks, indent=2))
    print(f"\nDone! Saved {len(artworks)} artworks to {OUTPUT_FILE}")
    return artworks


if __name__ == "__main__":
    scrape()
