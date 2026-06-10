"""
scripts/feynman_scraper.py
Scrapes The Feynman Lectures on Physics (Vol I–III) from feynmanlectures.caltech.edu.

PRIVATE USE ONLY — Do NOT commit scraped output to GitHub.
The site is protected by Cloudflare. We use curl_cffi to impersonate Chrome's TLS fingerprint
and download all chapters with LaTeX equations intact.

Usage:
    python scripts/feynman_scraper.py

Output:
    data/raw/feynman/vol1/I_01.txt ... I_52.txt
    data/raw/feynman/vol2/II_01.txt ... II_42.txt
    data/raw/feynman/vol3/III_01.txt ... III_21.txt

Each .txt file has a JSON metadata header (lines starting with #META:),
followed by the plain text with MathJax LaTeX inline as \\(...\\).
"""

import os
import sys
import json
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEYNMAN_DIR, FEYNMAN_CHAPTERS, FEYNMAN_BASE_URL, FEYNMAN_SCRAPE_DELAY

# Volume metadata
VOL_TITLES = {
    "I":   "Mechanics, Radiation, and Heat",
    "II":  "Electromagnetism and Matter",
    "III": "Quantum Mechanics",
}


def save_chapter(vol: str, chapter_num: int, title: str, text: str) -> str:
    """Save a chapter as a .txt file with JSON metadata header."""
    vol_dir = os.path.join(FEYNMAN_DIR, f"vol{['I', 'II', 'III'].index(vol) + 1}")
    os.makedirs(vol_dir, exist_ok=True)

    ch_str = f"{chapter_num:02d}"
    filename = f"{vol}_{ch_str}.txt"
    filepath = os.path.join(vol_dir, filename)

    metadata = {
        "source": "Feynman Lectures on Physics",
        "volume": f"Vol {vol}",
        "volume_title": VOL_TITLES[vol],
        "chapter": chapter_num,
        "chapter_id": f"{vol}_{ch_str}",
        "title": title,
        "url": f"{FEYNMAN_BASE_URL}/{vol}_{ch_str}.html",
        "copyright": "California Institute of Technology - For personal/educational use only",
        "latex_preserved": True,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        # Write JSON metadata header
        f.write(f"#META: {json.dumps(metadata)}\n\n")
        f.write(text)

    return filepath


def extract_text_from_html(html: str) -> tuple[str, str]:
    """
    Extract chapter title and clean text from HTML.
    Preserves inline LaTeX as \\(...\\).
    Returns (title, text).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Get chapter title
    title = ""
    h1 = soup.find("h1", class_="mathjax") or soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # Find main content — try div#main, then div.chapter-content, then body
    content = (
        soup.find("div", id="main")
        or soup.find("div", class_="chapter-content")
        or soup.find("article")
        or soup.body
    )

    if not content:
        return title, ""

    # Remove navigation, headers, footers, scripts, styles
    for tag in content.find_all(["nav", "header", "footer", "script", "style",
                                   "button", "aside", "noscript"]):
        tag.decompose()

    # Remove toolbars and other UI elements
    for tag in content.find_all(class_=re.compile(r"toolbar|nav|menu|footer|header|toc")):
        tag.decompose()

    # Replace MathJax spans with their LaTeX source
    # MathJax renders LaTeX in <script type="math/tex"> or data attributes
    for script in content.find_all("script", type="math/tex"):
        latex = script.string or ""
        # Replace with inline LaTeX notation
        script.replace_with(f"\\({latex}\\)")

    for script in content.find_all("script", type="math/tex; mode=display"):
        latex = script.string or ""
        script.replace_with(f"\n\\[\n{latex}\n\\]\n")

    # Get text, preserving paragraph breaks
    paragraphs = []
    for elem in content.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        text = elem.get_text(separator=" ", strip=True)
        if text and len(text) > 10:  # Skip very short fragments
            paragraphs.append(text)

    text = "\n\n".join(paragraphs)

    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return title, text.strip()


def scrape_with_curl_cffi():
    """Scrape all Feynman chapters using curl_cffi to bypass Cloudflare."""
    from curl_cffi import requests

    total_chapters = sum(FEYNMAN_CHAPTERS.values())
    scraped = 0
    skipped = 0
    failed = []

    print("=" * 60)
    print("  Feynman Lectures Scraper (curl_cffi)")
    print("=" * 60)
    print(f"  Target: {total_chapters} chapters across 3 volumes")
    print(f"  Output: {FEYNMAN_DIR}")
    print(f"  Delay: {FEYNMAN_SCRAPE_DELAY}s between requests (polite crawling)")
    print()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for vol, n_chapters in FEYNMAN_CHAPTERS.items():
        vol_num = ["I", "II", "III"].index(vol) + 1
        vol_dir = os.path.join(FEYNMAN_DIR, f"vol{vol_num}")
        os.makedirs(vol_dir, exist_ok=True)

        print(f"Vol {vol} — {VOL_TITLES[vol]} ({n_chapters} chapters)")
        print("-" * 50)

        for ch in range(1, n_chapters + 1):
            ch_str = f"{ch:02d}"
            filename = f"{vol}_{ch_str}.txt"
            filepath = os.path.join(vol_dir, filename)

            # Skip if already downloaded and valid
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        existing_text = f.read()
                    if "blocked" not in existing_text.lower() and len(existing_text) > 500:
                        print(f"  [{vol}_{ch_str}] Already exists — skipping")
                        skipped += 1
                        scraped += 1
                        continue
                except Exception:
                    pass

            url = f"{FEYNMAN_BASE_URL}/{vol}_{ch_str}.html"

            try:
                print(f"  [{vol}_{ch_str}] Fetching {url}...")
                r = requests.get(url, headers=headers, impersonate="chrome120", timeout=30)
                
                if r.status_code != 200:
                    raise Exception(f"HTTP status {r.status_code}")
                
                html = r.text
                if "blocked" in html.lower() or "security service" in html.lower():
                    raise Exception("Cloudflare blocked page detected")

                title, text = extract_text_from_html(html)

                if len(text) < 200:
                    raise Exception(f"Very short text extracted: {len(text)} chars")

                filepath = save_chapter(vol, ch, title, text)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"  [{vol}_{ch_str}] '{title[:50]}' — {len(text):,} chars → {size_kb:.1f} KB")
                scraped += 1

            except Exception as e:
                print(f"  [{vol}_{ch_str}] FAILED: {e}")
                failed.append(f"{vol}_{ch_str}")

            # Polite delay between requests
            if ch < n_chapters:
                time.sleep(FEYNMAN_SCRAPE_DELAY)

        print()

    print("=" * 60)
    print(f"  Scraped: {scraped}/{total_chapters}")
    print(f"  Skipped (cached): {skipped}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
        print("  Re-run the script to retry failed chapters.")
    else:
        print("  All chapters scraped successfully!")
    print()
    print("  Next step: python src/ingest.py")
    print("=" * 60)


def main():
    scrape_with_curl_cffi()


if __name__ == "__main__":
    main()
