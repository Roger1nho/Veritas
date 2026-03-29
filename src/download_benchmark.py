"""
prepare_benchmark.py
====================
Transformă CSV-urile FakeNewsNet (politifact / gossipcop) în formatul
benchmark_index.csv cerut de VeritasDataset.

Folosește TITLUL articolului ca text și încearcă să descarce imaginea
principală de la news_url. Dacă imaginea nu e accesibilă, salvează o
imagine placeholder albă și continuă.

Rulare:
    python prepare_benchmark.py

Output:
    C:\\Veritas\\benchmark_data\\benchmark_index.csv
    C:\\Veritas\\benchmark_data\\real\\*.jpg
    C:\\Veritas\\benchmark_data\\fake\\*.jpg
"""

import os
import time
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# ── Configurare ──────────────────────────────────────────────────────────────

# Pune aici calea unde ai salvat cele 4 CSV-uri descărcate
CSV_DIR = r"C:\Veritas"

# Unde se creează dataset-ul benchmark
BENCHMARK_DIR = r"C:\Veritas\benchmark_data"

# Câte articole iei din fiecare sursă (None = toate)
# Recomandat: 300 per clasă ca să fie echilibrat și rapid de antrenat
MAX_PER_CLASS = 300

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Inițializare foldere ─────────────────────────────────────────────────────

os.makedirs(os.path.join(BENCHMARK_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(BENCHMARK_DIR, "fake"), exist_ok=True)


def make_placeholder(path):
    """Salvează o imagine albă 224x224 ca placeholder."""
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    img.save(path)


def download_image(url, save_path, timeout=6):
    """
    Încearcă să descarce imaginea principală de la URL.
    Strategii:
      1. Caută og:image în meta tag-uri
      2. Prima imagine din pagină
      3. Placeholder dacă nimic nu merge
    Returnează True dacă a reușit.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")

        # Dacă URL-ul e direct o imagine
        if "image" in content_type:
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(save_path)
            return True

        # Altfel parsăm HTML-ul după og:image
        from html.parser import HTMLParser

        class MetaParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.og_image = None

            def handle_starttag(self, tag, attrs):
                if tag == "meta":
                    attrs_dict = dict(attrs)
                    if attrs_dict.get("property") == "og:image":
                        self.og_image = attrs_dict.get("content")

        parser = MetaParser()
        parser.feed(resp.text[:50000])  # primii 50k chars, suficient pt header

        if parser.og_image and parser.og_image.startswith("http"):
            img_resp = requests.get(parser.og_image, headers=HEADERS, timeout=timeout)
            img_resp.raise_for_status()
            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            img.save(save_path)
            return True

    except Exception:
        pass

    make_placeholder(save_path)
    return False


def process_df(df, label, prefix, max_items):
    """
    Procesează un DataFrame FakeNewsNet și returnează lista de înregistrări.
    label: 0 = real, 1 = fake
    prefix: 'politifact' sau 'gossipcop'
    """
    folder_name = "real" if label == 0 else "fake"
    records = []
    df = df.dropna(subset=["title", "news_url"])

    if max_items:
        df = df.head(max_items)

    total = len(df)
    print(f"\n{'─'*55}")
    print(f"  [{prefix.upper()} | {folder_name.upper()}] {total} articole de procesat")
    print(f"{'─'*55}")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        text = str(row["title"]).strip()
        url = str(row["news_url"]).strip()
        if not url.startswith("http"):
            url = "https://" + url

        filename = f"{prefix}_{folder_name}_{i}.jpg"
        save_path = os.path.join(BENCHMARK_DIR, folder_name, filename)

        ok = download_image(url, save_path)
        status = "✅" if ok else "⬜"  # ⬜ = placeholder

        print(f"  {status} [{i:>4}/{total}] {text[:60]}")

        records.append({
            "text": text,
            "filename": filename,
            "folder": folder_name,
            "label": label,
        })

        time.sleep(0.3)  # politicos cu serverele

    return records


def main():
    print("=" * 55)
    print("  VERITAS – Pregătire Dataset Benchmark FakeNewsNet")
    print("=" * 55)

    # ── Citire CSV-uri ────────────────────────────────────────────────────────
    pf_real = pd.read_csv(os.path.join(CSV_DIR, "politifact_real.csv"))
    pf_fake = pd.read_csv(os.path.join(CSV_DIR, "politifact_fake.csv"))
    gc_real = pd.read_csv(os.path.join(CSV_DIR, "gossipcop_real.csv"))
    gc_fake = pd.read_csv(os.path.join(CSV_DIR, "gossipcop_fake.csv"))

    print(f"\nDate găsite:")
    print(f"  PolitiFact real : {len(pf_real):>6} articole")
    print(f"  PolitiFact fake : {len(pf_fake):>6} articole")
    print(f"  GossipCop real  : {len(gc_real):>6} articole")
    print(f"  GossipCop fake  : {len(gc_fake):>6} articole")

    max_pf = MAX_PER_CLASS // 2  # jumătate din buget pentru PolitiFact
    max_gc = MAX_PER_CLASS // 2  # jumătate pentru GossipCop

    all_records = []
    all_records += process_df(pf_real, label=0, prefix="politifact", max_items=max_pf)
    all_records += process_df(pf_fake, label=1, prefix="politifact", max_items=max_pf)
    all_records += process_df(gc_real, label=0, prefix="gossipcop",  max_items=max_gc)
    all_records += process_df(gc_fake, label=1, prefix="gossipcop",  max_items=max_gc)

    # ── Salvare index ─────────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_records)
    out_path = os.path.join(BENCHMARK_DIR, "benchmark_index.csv")
    df_out.to_csv(out_path, index=False)

    real_count = (df_out["label"] == 0).sum()
    fake_count = (df_out["label"] == 1).sum()

    print("\n" + "=" * 55)
    print("  GATA!")
    print(f"  Total înregistrări : {len(df_out)}")
    print(f"  Real               : {real_count}")
    print(f"  Fake               : {fake_count}")
    print(f"  Index salvat în    : {out_path}")
    print("=" * 55)
    print("\nPasul următor: rulează benchmark_train.py")


if __name__ == "__main__":
    main()