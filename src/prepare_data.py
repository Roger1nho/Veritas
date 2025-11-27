import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# --- CONFIGURARE ---
DATA_DIR = r"C:\Veritas\data"
USE_ONLY_DEMO_DATA = True  # <--- AM PUS PE TRUE CA SĂ MERGĂ SIGUR ACUM


def setup_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    for label in ['real', 'fake']:
        path = os.path.join(DATA_DIR, label)
        if not os.path.exists(path):
            os.makedirs(path)


def download_image(url, save_path):
    try:
        # User-agent ca să nu fim blocați de Wikipedia
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.thumbnail((300, 300))
            image.save(save_path, "JPEG")
            return True
    except Exception as e:
        print(f"Eroare download {url}: {e}")
        return False
    return False


def create_demo_dataset():
    print("\nGenerare DATASET DEMO (pentru testarea codului)...")

    # Imagini stabile de pe Wikipedia
    demo_data = [
        # (URL, Text, Label: 0=Real, 1=Fake)
        (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/640px-President_Barack_Obama.jpg",
        "Barack Obama zambeste", 0),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/FullMoon2010.jpg/640px-FullMoon2010.jpg",
         "Luna plina pe cer senin", 0),
        (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Giraffe_Mikumi_National_Park.jpg/640px-Giraffe_Mikumi_National_Park.jpg",
        "O girafa in habitat natural", 0),

        # Fake-uri simulate
        (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/JPEG_Mans_Face_Compression_Artifacts.jpg/603px-JPEG_Mans_Face_Compression_Artifacts.jpg",
        "Imagine distorsionata digital cu artefacte", 1),
        ("https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
         "Zaruri editate digital", 1),
    ]

    metadata = []
    for i, (url, text, label) in enumerate(demo_data):
        folder = "fake" if label == 1 else "real"
        filename = f"demo_{folder}_{i}.jpg"
        save_path = os.path.join(DATA_DIR, folder, filename)

        print(f"Descarc imagine {i + 1}/5: {filename}...")
        if download_image(url, save_path):
            metadata.append({
                "filename": filename,
                "text": text,
                "label": label,
                "folder": folder
            })
        else:
            print("   -> Eșuat.")

    # Salvăm CSV-ul doar dacă am descărcat ceva
    if metadata:
        df = pd.DataFrame(metadata)
        csv_path = os.path.join(DATA_DIR, "dataset_index.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✅ SUCCESS! Dataset creat în {csv_path}")
        print(f"   Număr imagini: {len(metadata)}")
    else:
        print("\n❌ EROARE: Nu s-a putut descărca nicio imagine demo. Verifică internetul.")


def build_dataset():
    # Dacă e activat modul DEMO, sărim peste Fakeddit
    if USE_ONLY_DEMO_DATA:
        create_demo_dataset()
        return

    # Aici ar fi codul pentru Fakeddit, dar momentan îl ocolim
    # pentru a debloca proiectul.
    create_demo_dataset()


if __name__ == "__main__":
    setup_directories()
    build_dataset()