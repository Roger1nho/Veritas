import torch
from transformers import AutoTokenizer, ViTImageProcessor
from PIL import Image
import requests
from io import BytesIO
import os
from fusion_model import MultimodalFakeNewsModel

BASE_DIR = r"C:\Veritas"
MODEL_PATH = os.path.join(BASE_DIR, "veritas_model_v1.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_trained_model():
    print(f"Se încarcă modelul de pe {DEVICE}...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Nu găsesc modelul la: {MODEL_PATH}. Rulează train.py întâi!")

    model = MultimodalFakeNewsModel()

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def predict(model, text, image_source):
    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    image = None

    # Imagine de pe internet
    if image_source.startswith("http"):
        try:
            # Adăugăm headers ca să nu fim blocați de site-uri
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            print(f"Descarc imaginea de la: {image_source[:50]}...")
            response = requests.get(image_source, headers=headers, timeout=10)
            response.raise_for_status()  # Verifică dacă e eroare 404/403
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"❌ EROARE la descărcarea imaginii: {e}")
            return

    # Imagine locală
    else:
        if os.path.exists(image_source):
            try:
                print(f"Încarc imagine locală: {image_source}")
                image = Image.open(image_source).convert("RGB")
            except Exception as e:
                print(f"❌ EROARE la citirea imaginii locale: {e}")
                return
        else:
            print(f"❌ EROARE: Fișierul nu există: {image_source}")
            return

    text_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    image_inputs = image_processor(images=image, return_tensors="pt")

    input_ids = text_inputs['input_ids'].to(DEVICE)
    attention_mask = text_inputs['attention_mask'].to(DEVICE)
    pixel_values = image_inputs['pixel_values'].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, pixel_values)
        probs = torch.nn.functional.softmax(logits, dim=1)

    fake_prob = probs[0][1].item()
    real_prob = probs[0][0].item()

    print("\n" + "=" * 50)
    print(f"REZULTAT VERITAS")
    print("=" * 50)
    print(f"Știre: {text[:80]}...")
    print("-" * 20)

    if fake_prob > 0.5:
        print(f"🔴 FAKE / SUSPECT (Probabilitate: {fake_prob * 100:.2f}%)")
    else:
        print(f"🟢 REAL / CREDIBIL (Probabilitate: {real_prob * 100:.2f}%)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        model = load_trained_model()
        data_path = r"C:\Veritas\data"

        # găsim automat o imagine 'fake' și una 'real'
        real_img_path = None
        fake_img_path = None

        if os.path.exists(os.path.join(data_path, "real")):
            files = os.listdir(os.path.join(data_path, "real"))
            if files: real_img_path = os.path.join(data_path, "real", files[0])

        if os.path.exists(os.path.join(data_path, "fake")):
            files = os.listdir(os.path.join(data_path, "fake"))
            if files: fake_img_path = os.path.join(data_path, "fake", files[0])

        print("\n--- TEST CU IMAGINI LOCALE ---")
        if real_img_path:
            predict(model, "Aceasta este o știre adevărată despre un eveniment real.", real_img_path)

        if fake_img_path:
            predict(model, "Extratereștrii au invadat pământul și vând covrigi.", fake_img_path)

        print("\n--- TEST ONLINE (Google Logo) ---")
        predict(model, "Google a lansat un nou motor de căutare.",
                "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png")

    except Exception as e:
        print(f"Eroare generală: {e}")
