import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
from io import BytesIO
import time

# --- CONFIGURARE ---
BASE_DIR = r"C:\Veritas\data"
REAL_SOURCES_PATH = r"C:\Veritas\real_sources.csv"
FAKE_SOURCES_PATH = r"C:\Veritas\fake_sources.csv"

# Ne asigurăm că există folderele
os.makedirs(os.path.join(BASE_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "fake"), exist_ok=True)


def scrape_and_save(url, label, index):
    """
    Descarcă articolul, salvează imaginea și returnează datele pentru CSV-ul final.
    Label: 0 = Real, 1 = Fake
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0'}
        response = requests.get(url, headers=headers, timeout=10)

        # Verificăm dacă link-ul e valid
        if response.status_code != 200:
            print(f"⚠️ [SKIP] {url} - Cod {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Extragere Titlu
        title_tag = soup.find('h1')
        if not title_tag:
            return None
        title = title_tag.get_text().strip()

        # 2. Extragere Text (paragrafe)
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text() for p in paragraphs])
        text_content = " ".join(text_content.split())  # Curățare spații

        if len(text_content) < 100:
            return None

        clean_paragraphs = []
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 30:
                clean_paragraphs.append(text)

        text_content = " ".join(clean_paragraphs)

        # Eliminăm limita de caractere complet!
        full_text = f"{title}. {text_content}"

        # 3. Extragere Imagine
        image_url = ""
        meta_image = soup.find('meta', property='og:image')
        if meta_image:
            image_url = meta_image['content']
        else:
            img = soup.find('img')
            if img and 'src' in img.attrs:
                image_url = img['src']

        if not image_url or not image_url.startswith('http'):
            return None

        # Descărcare imagine
        img_resp = requests.get(image_url, headers=headers, timeout=5)
        img = Image.open(BytesIO(img_resp.content)).convert('RGB')

        # Salvare Imagine
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
        # Folosim indexul global pentru a nu suprascrie fișierele dacă rulăm de mai multe ori
        # (Aici folosim 'i' simplu, dar ai grijă să ștergi folderul 'data' dacă reiei de la zero)
        data = scrape_and_save(url, label, i)
        if data:
            results.append(data)
        time.sleep(0.5)  # Pauză scurtă

    return results


def main():
    all_data = []

    # 1. Procesăm Real (Label 0)
    real_data = process_csv(REAL_SOURCES_PATH, label=0)
    all_data.extend(real_data)

    # 2. Procesăm Fake (Label 1)
    fake_data = process_csv(FAKE_SOURCES_PATH, label=1)
    all_data.extend(fake_data)

    # 3. Salvare Dataset Final
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