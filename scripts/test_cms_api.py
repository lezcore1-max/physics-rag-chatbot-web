from curl_cffi import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

slugs = [
    "university-physics-volume-1",
    "university-physics-volume-2",
    "university-physics-volume-3",
]

for slug in slugs:
    # Get page id
    r = requests.get(f"https://openstax.org/apps/cms/api/v2/pages/?slug={slug}", headers=headers, impersonate="chrome124", timeout=30)
    page_id = r.json()["items"][0]["id"]
    
    # Get detail
    r2 = requests.get(f"https://openstax.org/apps/cms/api/v2/pages/{page_id}/", headers=headers, impersonate="chrome124", timeout=30)
    detail = r2.json()
    
    pdf = detail.get("high_resolution_pdf_url") or detail.get("low_resolution_pdf_url")
    print(f"{slug}: {pdf}")
