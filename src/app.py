import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, ViTImageProcessor
from PIL import Image
import os
import sys
import requests
from bs4 import BeautifulSoup
from io import BytesIO

# Adăugăm folderul curent la path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fusion_model import VeritasMultimodal

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="Veritas - Detector Fake News",
    page_icon="🕵️",
    layout="wide"
)

# --- CONFIGURĂRI CĂI ---
BASE_DIR = r"C:\Veritas"
MODEL_PATH = os.path.join(BASE_DIR, "veritas_model_v1.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- FUNCȚII UTILITARE ---

@st.cache_resource
def load_veritas_model():
    """Încarcă modelul și îl ține în cache."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Lipsă model: {MODEL_PATH}")
        return None, None, None

    model = VeritasMultimodal()
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Eroare la încărcare model: {e}")
        return None, None, None

    model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    return model, tokenizer, processor


def scrape_article(url):
    """
    Extrage textul și imaginea principală dintr-un URL.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Extragere Titlu
        title = soup.find('h1')
        title_text = title.get_text().strip() if title else "Titlu necunoscut"

        # 2. Extragere Text (Paragrafe)
        # Luăm toate tag-urile <p> și le unim
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text() for p in paragraphs])

        if len(text_content) < 50:
            return None, None, "Nu am putut extrage suficient text. Site-ul ar putea fi protejat."

        # 3. Extragere Imagine (Meta tags og:image)
        image_url = ""
        meta_image = soup.find('meta', property='og:image')
        if meta_image:
            image_url = meta_image['content']
        else:
            # Fallback: prima imagine din articol
            img_tag = soup.find('img')
            if img_tag and 'src' in img_tag.attrs:
                image_url = img_tag['src']

        # Descărcăm imaginea efectivă
        pil_image = None
        if image_url:
            if not image_url.startswith('http'):
                # Tratăm linkurile relative (ex: /img/logo.png)
                from urllib.parse import urljoin
                image_url = urljoin(url, image_url)

            img_resp = requests.get(image_url, headers=headers, timeout=5)
            pil_image = Image.open(BytesIO(img_resp.content)).convert("RGB")

        full_text = f"{title_text}. {text_content[:2000]}"  # Trunchiem textul dacă e prea lung
        return full_text, pil_image, None

    except Exception as e:
        return None, None, f"Eroare la scraping: {str(e)}"


# --- LOGICA PRINCIPALĂ ---

# Încărcare resurse
model, tokenizer, image_processor = load_veritas_model()

st.title("🕵️ Veritas: Analiză Automată Știri")
st.markdown("Detector de dezinformare bazat pe Inteligență Artificială Multimodală.")

# TAB-uri pentru modurile de lucru
tab1, tab2 = st.tabs(["🔗 Analiză Link (Automat)", "📂 Încărcare Manuală"])

final_image = None
final_text = None
start_analysis = False

# --- TAB 1: SCRAPING AUTOMAT ---
with tab1:
    st.info("Introduceți link-ul unei știri (ex: Digi24, Hotnews, CNN) pentru analiză automată.")
    url_input = st.text_input("URL Articol", placeholder="https://www.exemplu.ro/stire-socanta")

    if st.button("Extrage și Analizează", key="btn_scrape"):
        if not url_input:
            st.warning("Introduceți un link valid.")
        else:
            with st.spinner('Conectare la site și extragere date...'):
                extracted_text, extracted_image, error = scrape_article(url_input)

                if error:
                    st.error(error)
                elif not extracted_image:
                    st.warning(
                        "Am găsit textul, dar nu am putut identifica o imagine relevantă. Încearcă încărcarea manuală.")
                    st.text_area("Text extras:", extracted_text, height=100)
                else:
                    final_text = extracted_text
                    final_image = extracted_image
                    start_analysis = True

                    # Afișăm ce am găsit
                    col_p1, col_p2 = st.columns([1, 2])
                    with col_p1:
                        st.image(final_image, caption="Imagine extrasă", use_container_width=True)
                    with col_p2:
                        st.caption("Text extras (previzualizare):")
                        st.write(final_text[:500] + " [...]")

# --- TAB 2: MANUAL (BACKUP) ---
with tab2:
    st.write("Dacă link-ul nu funcționează, poți încărca datele manual.")
    uploaded_file = st.file_uploader("Imagine", type=["jpg", "png"])
    manual_text = st.text_area("Text Știre", height=150)

    if st.button("Analizează Manual", key="btn_manual"):
        if uploaded_file and manual_text:
            final_image = Image.open(uploaded_file).convert("RGB")
            final_text = manual_text
            start_analysis = True
        else:
            st.warning("Completează ambele câmpuri.")

# --- EXECUȚIE ANALIZĂ (COMUNĂ) ---
if start_analysis and model:
    st.divider()
    st.markdown("### 🔍 Rezultatul Modelului Veritas")

    progress_bar = st.progress(0, text="Inițializare inferență...")

    # 1. Procesare
    try:
        # Text
        progress_bar.progress(40, text="Analiză semantică text (BERT)...")
        text_inputs = tokenizer(final_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

        # Imagine
        progress_bar.progress(60, text="Analiză vizuală pixeli (Vision Transformer)...")
        image_inputs = image_processor(images=final_image, return_tensors="pt")

        # Device
        input_ids = text_inputs['input_ids'].to(DEVICE)
        attention_mask = text_inputs['attention_mask'].to(DEVICE)
        pixel_values = image_inputs['pixel_values'].to(DEVICE)

        # Predicție
        with torch.no_grad():
            logits = model(input_ids, attention_mask, pixel_values)
            probs = F.softmax(logits, dim=1)

        fake_prob = probs[0][1].item()
        real_prob = probs[0][0].item()

        progress_bar.progress(100, text="Gata!")

        # Afișare
        c1, c2 = st.columns(2)

        with c1:
            st.metric("Probabilitate FAKE", f"{fake_prob * 100:.2f}%")
            if fake_prob > 0.5:
                st.error("🔴 VERDICT: SUSPECT / FAKE")
            else:
                st.success("🟢 VERDICT: REAL / CREDIBIL")

        with c2:
            st.caption("Încrederea modelului:")
            st.bar_chart({"Fake": fake_prob, "Real": real_prob}, color=["#FF0000", "#00FF00"])

    except Exception as e:
        st.error(f"Eroare în timpul analizei AI: {e}")