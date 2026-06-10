from curl_cffi import requests

url = "https://openstax.org/details/books/university-physics-volume-1"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

try:
    r = requests.get(url, headers=headers, impersonate="chrome124", timeout=30)
    print("Status:", r.status_code)
    print("HTML length:", len(r.text))
    print("HTML snippet:", r.text[:1000])
except Exception as e:
    print("Error:", e)
