import re
import urllib.request

url = "https://research.stem.open.ac.uk/ouanalyse/dataset/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=15) as res:
        html = res.read().decode("utf-8", errors="ignore")
        links = re.findall(r'href=["\']([^"\']+)["\']', html)
        print("Links found on dataset page:")
        for l in set(links):
            if any(
                ext in l.lower()
                for ext in [
                    ".zip",
                    ".csv",
                    ".tar",
                    ".gz",
                    "download",
                    "drive.google",
                    "kaggle",
                    "zenodo",
                    "figshare",
                    "archive.org",
                ]
            ):
                print(" ->", l)
except Exception as e:  # noqa: BLE001
    print("Error:", e)
