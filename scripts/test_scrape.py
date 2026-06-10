import urllib.request
import urllib.error

url = "https://www.feynmanlectures.caltech.edu/I_01.html"
req = urllib.request.Request(
    url,
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
)

try:
    with urllib.request.urlopen(req, timeout=10) as response:
        html = response.read().decode('utf-8')
        print("Status:", response.status)
        print("HTML length:", len(html))
        print("HTML snippet:", html[:500])
except Exception as e:
    print("Error:", e)
