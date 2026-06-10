from curl_cffi import requests
from bs4 import BeautifulSoup
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEYNMAN_DIR
from scripts.feynman_scraper import extract_text_from_html, save_chapter

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.feynmanlectures.caltech.edu/"
}

failed_chapters = [
    ("I", 37, "I_37"),
    ("III", 1, "III_01"),
    ("III", 5, "III_05"),
    ("III", 6, "III_06")
]

def main():
    print("Failed chapters downloader starting...")
    session = requests.Session()
    
    # Step 1: Visit homepage to acquire cookies / CF credentials
    print("Visiting homepage...")
    try:
        r = session.get("https://www.feynmanlectures.caltech.edu/", headers=headers, impersonate="chrome120", timeout=30)
        print("Homepage status:", r.status_code)
        time.sleep(3)
    except Exception as e:
        print("Homepage visit failed:", e)
        return

    # Step 2: Fetch each failed chapter
    for vol, ch_num, ch_id in failed_chapters:
        url = f"https://www.feynmanlectures.caltech.edu/{ch_id}.html"
        print(f"Fetching {url}...")
        try:
            r = session.get(url, headers=headers, impersonate="chrome120", timeout=30)
            print("Status:", r.status_code)
            html = r.text
            if "blocked" in html.lower() or "security service" in html.lower():
                print(f"Blocked by Cloudflare for {ch_id}!")
            else:
                title, text = extract_text_from_html(html)
                if len(text) > 200:
                    filepath = save_chapter(vol, ch_num, title, text)
                    print(f"Successfully saved {ch_id} to {filepath} ({len(text)} chars)")
                else:
                    print(f"Extracted text too short for {ch_id}!")
        except Exception as e:
            print(f"Error fetching {ch_id}:", e)
        
        # Polite delay
        time.sleep(3)

if __name__ == "__main__":
    main()
