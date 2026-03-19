"""
Scrape artwork metadata from two sources:
1. Met Museum Open Access CSV (GitHub)
2. Art Institute of Chicago API

Saves 1000 artworks total to data/met_artworks.json in a unified format.
"""

import csv
import io
import json
import random
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "met_artworks.json"
TARGET_COUNT = 1000
MET_TARGET = 500
AIC_TARGET = 500

AIC_BASE = "https://api.artic.edu/api/v1/artworks"
AIC_IMAGE_BASE = "https://www.artic.edu/iiif/2"
MET_CSV_URL = "https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv"
# Fallback: the raw GitHub URL redirects to media for LFS files
MET_CSV_FALLBACK = "https://raw.githubusercontent.com/metmuseum/openaccess/master/MetObjects.csv"


def scrape_aic(target=AIC_TARGET):
    """Fetch artworks from the Art Institute of Chicago API."""
    print(f"[AIC] Fetching {target} artworks from Art Institute of Chicago...")
    artworks = []
    page = 1
    per_page = 100
    fields = "id,title,artist_display,date_display,medium_display,dimensions,department_title,classification_title,place_of_origin,image_id,term_titles,is_public_domain"

    while len(artworks) < target:
        try:
            resp = requests.get(AIC_BASE, params={
                "page": page,
                "limit": per_page,
                "fields": fields,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [AIC] Error on page {page}: {e}")
            break

        records = data.get("data", [])
        if not records:
            break

        for r in records:
            if len(artworks) >= target:
                break
            image_id = r.get("image_id")
            if not image_id:
                continue
            artworks.append({
                "objectID": f"aic_{r['id']}",
                "title": r.get("title", ""),
                "artistDisplayName": r.get("artist_display", ""),
                "artistNationality": "",
                "objectDate": r.get("date_display", ""),
                "medium": r.get("medium_display", ""),
                "dimensions": r.get("dimensions", ""),
                "department": r.get("department_title", ""),
                "classification": r.get("classification_title", ""),
                "culture": r.get("place_of_origin", ""),
                "period": "",
                "dynasty": "",
                "primaryImage": f"{AIC_IMAGE_BASE}/{image_id}/full/843,/0/default.jpg",
                "primaryImageSmall": f"{AIC_IMAGE_BASE}/{image_id}/full/200,/0/default.jpg",
                "objectURL": f"https://www.artic.edu/artworks/{r['id']}",
                "tags": r.get("term_titles", []) or [],
                "isPublicDomain": r.get("is_public_domain", False),
                "source": "aic",
            })

        print(f"  [AIC] {len(artworks)}/{target} artworks collected (page {page})")
        page += 1
        time.sleep(0.3)

    print(f"[AIC] Done: {len(artworks)} artworks")
    return artworks


def scrape_met_csv(target=MET_TARGET):
    """Download and parse the Met Museum Open Access CSV from GitHub."""
    print(f"[MET] Downloading Met Museum CSV from GitHub...")

    try:
        resp = requests.get(MET_CSV_URL, timeout=120, stream=True)
        if resp.status_code != 200:
            print(f"  [MET] Primary URL returned {resp.status_code}, trying fallback...")
            resp = requests.get(MET_CSV_FALLBACK, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [MET] Failed to download CSV: {e}")
        return []

    print("[MET] Parsing CSV...")
    content = resp.content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(content))

    # Collect rows that have images
    candidates = []
    for row in reader:
        img = row.get("Link Resource", "") or row.get("Primary Image", "")
        if img and row.get("Object ID"):
            candidates.append(row)

    print(f"[MET] Found {len(candidates)} artworks with images in CSV")
    random.shuffle(candidates)
    candidates = candidates[:target]

    artworks = []
    for row in candidates:
        tags_raw = row.get("Tags", "")
        tags = [t.strip() for t in tags_raw.split("|") if t.strip()] if tags_raw else []

        artworks.append({
            "objectID": int(row.get("Object ID", 0)),
            "title": row.get("Title", ""),
            "artistDisplayName": row.get("Artist Display Name", ""),
            "artistNationality": row.get("Artist Nationality", ""),
            "objectDate": row.get("Object Date", ""),
            "medium": row.get("Medium", ""),
            "dimensions": row.get("Dimensions", ""),
            "department": row.get("Department", ""),
            "classification": row.get("Classification", ""),
            "culture": row.get("Culture", ""),
            "period": row.get("Period", ""),
            "dynasty": row.get("Dynasty", ""),
            "primaryImage": row.get("Primary Image", ""),
            "primaryImageSmall": row.get("Primary Image Small", ""),
            "objectURL": row.get("Object URL", "") or f"https://www.metmuseum.org/art/collection/search/{row.get('Object ID', '')}",
            "tags": tags,
            "isPublicDomain": row.get("Is Public Domain", "").lower() == "true",
            "source": "met",
        })

    print(f"[MET] Done: {len(artworks)} artworks")
    return artworks


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Scrape both sources
    aic_works = scrape_aic(AIC_TARGET)
    met_works = scrape_met_csv(MET_TARGET)

    # If one source underdelivered, try to compensate with the other
    combined = met_works + aic_works
    combined = combined[:TARGET_COUNT]

    OUTPUT_FILE.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(combined)} artworks to {OUTPUT_FILE}")
    print(f"  - Met Museum: {sum(1 for a in combined if a.get('source') == 'met')}")
    print(f"  - Art Institute of Chicago: {sum(1 for a in combined if a.get('source') == 'aic')}")


if __name__ == "__main__":
    main()
