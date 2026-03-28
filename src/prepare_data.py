import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
from io import BytesIO
import time
import trafilatura

BASE_DIR = r"C:\Veritas\data"
REAL_SOURCES_PATH = r"C:\Veritas\real_sources.csv"
FAKE_SOURCES_PATH = r"C:\Veritas\fake_sources.csv"

# Ne asigurăm că există folderele
os.makedirs(os.path.join(BASE_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "fake"), exist_ok=True)


def scrape_and_save(url, label, index):
    try:
        #Folosim Trafilatura pentru descărcare și extragere
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            print(f"{url} - Nu s-a putut descărca.")
            return None

        # Extragem textul "curat"
        text_content = trafilatura.extract(downloaded, include_comments=False)
        metadata = trafilatura.extract_metadata(downloaded)

        if not text_content or len(text_content) < 100:
            return None

        title = metadata.title if metadata and metadata.title else ""
        full_text = f"{title}. {text_content}"

        # Curățăm caracterele invizibile
        full_text = " ".join(full_text.split())

        # 2. Extragere Imagine
        image_url = metadata.image if metadata else None

        # Fallback imagine (dacă trafilatura nu găsește)
        if not image_url:
            pass

        if not image_url or not image_url.startswith('http'):
            return None

        #Descărcare și Salvare
        headers = {'User-Agent': 'Mozilla/5.0'}
        img_resp = requests.get(image_url, headers=headers, timeout=5)
        img = Image.open(BytesIO(img_resp.content)).convert('RGB')

        filename = f"{'real' if label == 0 else 'fake'}_{index}.jpg"
        folder_name = "real" if label == 0 else "fake"
        save_path = os.path.join(BASE_DIR, folder_name, filename)
        img.save(save_path)

        print(f"✅ [{folder_name.upper()}] Salvat: {title[:40]}...")

        return {
            "text": full_text,
            "filename": filename,
            "folder": folder_name,
            "label": label
        }

    except Exception as e:
        print(f"❌ [EROARE] {url}: {e}")
        return None


def process_csv(csv_path, label):
    """
    Citește un CSV cu URL-uri și le procesează.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Fișierul {csv_path} nu există!")
        return []

    df = pd.read_csv(csv_path)
    results = []

    label_name = "REAL" if label == 0 else "FAKE"
    print(f"\n--- Procesare {len(df)} link-uri {label_name} ---")

    for i, row in df.iterrows():
        url = row['url']
        data = scrape_and_save(url, label, i)
        if data:
            results.append(data)
        time.sleep(0.5)  # Pauză scurtă

    return results


def main():
    all_data = []

    #Procesăm Real (Label 0)
    real_data = process_csv(REAL_SOURCES_PATH, label=0)
    all_data.extend(real_data)

    #Procesăm Fake (Label 1)
    fake_data = process_csv(FAKE_SOURCES_PATH, label=1)
    all_data.extend(fake_data)

    #Salvare Dataset Final
    if all_data:
        df_final = pd.DataFrame(all_data)
        csv_path = os.path.join(BASE_DIR, "dataset_index.csv")
        df_final.to_csv(csv_path, index=False)
        print(f"\n🎉 Gata! Dataset creat cu {len(df_final)} articole.")
        print(f"Index salvat în: {csv_path}")
    else:
        print("\n⚠️ Nu s-au descărcat date. Verifică CSV-urile sau conexiunea.")


if __name__ == "__main__":
    main()