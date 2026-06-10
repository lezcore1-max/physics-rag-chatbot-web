from curl_cffi import requests
from bs4 import BeautifulSoup

url = "https://www.feynmanlectures.caltech.edu/I_02.html"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

try:
    r = requests.get(url, headers=headers, impersonate="chrome124", timeout=30)
    print("Status:", r.status_code)
    print("HTML length:", len(r.text))
    if "Blocked" in r.text or "security service" in r.text:
        print("Blocked by Cloudflare!")
    else:
        print("Success! First 500 chars of HTML:")
        print(r.text[:500])
except Exception as e:
    print("Error:", e)
