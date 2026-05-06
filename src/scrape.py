"""
colecteaza_linkuri_reale.py
===========================
Colectează linkuri de articole REALE din RSS-urile surselor credibile românești
și le salvează într-un fișier text, câte unul pe linie.

Rulare:
  pip install requests
  python colecteaza_linkuri_reale.py

  python colecteaza_linkuri_reale.py --output linkuri.txt   # fișier custom
  python colecteaza_linkuri_reale.py --target 500           # câte linkuri vrei (default: 300)

Apoi deschizi linkuri_reale.txt, verifici manual ce vrei să păstrezi
și dai copy-paste în scraperul principal.
"""

import requests
import xml.etree.ElementTree as ET
import argparse
import os
import sys
from datetime import datetime

# ── Surse RSS ──────────────────────────────────────────────────────────────────

REAL_RSS = [
    ("digi24",      "https://www.digi24.ro/rss"),
    ("hotnews",     "https://www.hotnews.ro/rss/actualitate"),
    ("g4media",     "https://www.g4media.ro/feed"),
    ("protv",       "https://stirileprotv.ro/rss/stiri.xml"),
    ("libertatea",  "https://www.libertatea.ro/rss"),
    ("adevarul",    "https://adevarul.ro/rss/adevarul.xml"),
    ("ziare",       "https://www.ziare.com/rss/ultimele_stiri.xml"),
    ("euractiv_ro", "https://www.euractiv.ro/feed/"),
    ("rfi_ro",      "https://www.rfi.ro/rss.xml"),
    ("pressone",    "https://pressone.ro/feed/"),
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
})

# ── Funcții ────────────────────────────────────────────────────────────────────

def fetch_rss(url, timeout=15):
    try:
        r = SESSION.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.content
    except Exception as e:
        print(f"   ✗ Eroare fetch: {e}")
    return None


def parse_links(content):
    """Extrage linkuri de articole din RSS 2.0 sau Atom."""
    links = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"   ✗ XML invalid: {e}")
        return links

    # RSS 2.0
    for item in root.iter("item"):
        link = item.findtext("link", "").strip()
        if not link:
            el = item.find("link")
            if el is not None and el.text:
                link = el.text.strip()
        if link and link.startswith("http"):
            links.append(link)

    # Atom (fallback)
    if not links:
        ns = "http://www.w3.org/2005/Atom"
        for entry in root.iter(f"{{{ns}}}entry"):
            el = entry.find(f"{{{ns}}}link")
            if el is not None:
                href = el.get("href", "").strip()
                if href.startswith("http"):
                    links.append(href)

    return links


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Colectează linkuri reale din RSS-uri românești."
    )
    parser.add_argument(
        "--output", default="linkuri_reale.txt",
        help="Fișierul de ieșire (default: linkuri_reale.txt)"
    )
    parser.add_argument(
        "--target", type=int, default=300,
        help="Numărul maxim de linkuri de colectat (default: 300)"
    )
    args = parser.parse_args()

    all_links   = []   # [(sursa, url), ...]
    seen        = set()

    print(f"\n{'='*55}")
    print(f"  COLECTARE LINKURI REALE")
    print(f"  Target : {args.target} linkuri")
    print(f"  Output : {args.output}")
    print(f"{'='*55}\n")

    for source_name, rss_url in REAL_RSS:
        if len(all_links) >= args.target:
            break

        print(f"[{source_name}] {rss_url}")
        content = fetch_rss(rss_url)

        if not content:
            print(f"   ✗ Inaccesibil — sară\n")
            continue

        links = parse_links(content)

        added = 0
        for url in links:
            if len(all_links) >= args.target:
                break
            if url not in seen:
                seen.add(url)
                all_links.append((source_name, url))
                added += 1

        print(f"   ✓ {added} linkuri noi (total RSS: {len(links)})\n")

    # ── Scrie fișierul ──────────────────────────────────────────────────────────

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# Linkuri articole REALE — generate {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"# Total: {len(all_links)}\n")
        f.write(f"# Format: sursa<TAB>url\n")
        f.write("#\n")
        f.write("# Șterge ce nu vrei, păstrează ce vrei, apoi dă copy-paste în scraper.\n")
        f.write("#\n\n")

        current_source = None
        for source, url in all_links:
            if source != current_source:
                f.write(f"\n# --- {source.upper()} ---\n")
                current_source = source
            f.write(f"{url}\n")

    print(f"{'='*55}")
    print(f"  GATA!")
    print(f"  {len(all_links)} linkuri salvate în: {args.output}")
    print(f"{'='*55}\n")
    print("  Deschide fișierul, verifică manual, șterge ce nu vrei,")
    print("  apoi dă copy-paste linkurile în scraperul principal.\n")


if __name__ == "__main__":
    main()