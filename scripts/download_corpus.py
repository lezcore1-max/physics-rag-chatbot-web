"""
scripts/download_corpus.py
Uses the OpenStax CMS API to get fresh PDF download URLs and downloads all 3 volumes.
Falls back to hardcoded CDN URLs if the API fails.
"""

import os
import sys
from curl_cffi import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, OPENSTAX_PAGES, OPENSTAX_CDN_FALLBACK, MIN_PDF_SIZE_MB

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

CMS_API = "https://openstax.org/apps/cms/api/v2/pages/"

# Slug → filename mapping
OPENSTAX_SLUGS = {
    "openstax_vol1.pdf": "university-physics-volume-1",
    "openstax_vol2.pdf": "university-physics-volume-2",
    "openstax_vol3.pdf": "university-physics-volume-3",
}


def get_pdf_url_from_cms(slug: str) -> str | None:
    """Query the OpenStax CMS API to get the live high-res PDF download URL."""
    try:
        resp = requests.get(
            CMS_API, params={"slug": slug}, headers=HEADERS, impersonate="chrome124", timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("items"):
            return None
        page_id = data["items"][0]["id"]

        # Get the detail page which has pdf download URLs
        detail_resp = requests.get(
            f"{CMS_API}{page_id}/", headers=HEADERS, impersonate="chrome124", timeout=15
        )
        detail_resp.raise_for_status()
        detail = detail_resp.json()

        # Prefer high-res PDF, fall back to low-res
        return detail.get("high_resolution_pdf_url") or detail.get("low_resolution_pdf_url")
    except Exception as e:
        print(f"  Warning: CMS API lookup failed: {e}")
        return None


def download_file(url: str, dest_path: str, label: str) -> bool:
    """Download a file with a tqdm progress bar using curl_cffi for Cloudflare bypass."""
    try:
        resp = requests.get(
            url, headers=HEADERS, impersonate="chrome124", stream=True, timeout=120
        )
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            desc=f"  {label}",
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=80,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def verify_pdf(path: str) -> bool:
    """Check file exists and is large enough to be a valid PDF."""
    if not os.path.exists(path):
        return False
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < MIN_PDF_SIZE_MB:
        print(f"  File too small ({size_mb:.1f} MB) — likely corrupt")
        return False
    with open(path, "rb") as f:
        header = f.read(4)
    return header == b"%PDF"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 60)
    print("  OpenStax University Physics -- Corpus Downloader")
    print("=" * 60)
    print(f"  Saving to: {DATA_DIR}\n")

    success_count = 0

    for filename, slug in OPENSTAX_SLUGS.items():
        dest = os.path.join(DATA_DIR, filename)
        label = slug.replace("university-physics-volume-", "Vol ")

        print(f"[{label}]")

        if verify_pdf(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  Already exists ({size_mb:.1f} MB) -- skipping\n")
            success_count += 1
            continue

        # Try CMS API for live URL
        print(f"  Fetching PDF URL from OpenStax CMS API...")
        pdf_url = get_pdf_url_from_cms(slug)

        if pdf_url:
            print(f"  URL: {pdf_url[:90]}...")
        else:
            print(f"  CMS API failed -- using hardcoded fallback URL")
            pdf_url = OPENSTAX_CDN_FALLBACK.get(filename)

        if not pdf_url:
            print(f"  ERROR: No URL available for {filename}\n")
            continue

        ok = download_file(pdf_url, dest, label)

        if ok and verify_pdf(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  Downloaded successfully ({size_mb:.1f} MB)\n")
            success_count += 1
        else:
            print(f"  Download verification failed for {filename}\n")

    print("=" * 60)
    print(f"  Result: {success_count}/{len(OPENSTAX_SLUGS)} volumes ready")
    if success_count == len(OPENSTAX_SLUGS):
        print("  All volumes downloaded! Run: python src/ingest.py")
    else:
        print("  Some volumes missing. Check internet connection.")
    print("=" * 60)


if __name__ == "__main__":
    main()
