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

# --- CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="Veritas - Detector Fake News",
    page_icon="🕵️",
    layout="wide"
)

# --- 1. REPARARE CĂI (IMPORT CRITIC) ---
# Adăugăm folderul 'src' la calea sistemului pentru a găsi fusion_model.py
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Acum putem importa clasa corectă
try:
    from fusion_model import MultimodalFakeNewsModel
except ImportError:
    st.error("❌ Nu găsesc fișierul 'src/fusion_model.py'. Asigură-te că folderul 'src' este lângă 'app.py'.")
    st.stop()

# --- CONFIGURĂRI MODEL ---
# Folosim calea relativă pentru portabilitate
MODEL_PATH = "src/veritas_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_veritas_model():
    """Încarcă modelul și îl ține în cache."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Lipsă fișier model: {MODEL_PATH}. Ai rulat train.py?")
        return None, None, None

    # Inițializăm arhitectura
    model = MultimodalFakeNewsModel(num_labels=2)

    try:
        # Încărcăm greutățile antrenate
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"❌ Eroare la încărcare parametri model: {e}")
        return None, None, None

    model.to(DEVICE)
    model.eval()

    # ATENȚIE: Folosim 'cased' pentru că așa am antrenat!
    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    return model, tokenizer, processor


def scrape_article(url):
    """
    Extrage textul și imaginea principală dintr-un URL.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, None, f"Eroare accesare site: Cod {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Extragere Titlu
        title = soup.find('h1')
        title_text = title.get_text().strip() if title else ""

        # 2. Extragere Text
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text() for p in paragraphs])

        if len(text_content) < 50:
            return None, None, "Text insuficient extras. Site-ul poate avea protecție anti-bot."

        # 3. Extragere Imagine
        image_url = ""
        meta_image = soup.find('meta', property='og:image')
        if meta_image:
            image_url = meta_image['content']
        else:
            img_tag = soup.find('img')
            if img_tag and 'src' in img_tag.attrs:
                image_url = img_tag['src']

        # Descărcăm imaginea
        pil_image = None
        if image_url:
            if not image_url.startswith('http'):
                from urllib.parse import urljoin
                image_url = urljoin(url, image_url)

            try:
                img_resp = requests.get(image_url, headers=headers, timeout=5)
                pil_image = Image.open(BytesIO(img_resp.content)).convert("RGB")
            except:
                pil_image = None  # Imaginea nu a putut fi descărcată

        full_text = f"{title_text}. {text_content[:2000]}"
        return full_text, pil_image, None

    except Exception as e:
        return None, None, f"Eroare generală: {str(e)}"


# --- INTERFAȚA GRAFICĂ ---

# Încărcare resurse
model, tokenizer, image_processor = load_veritas_model()

if not model:
    st.stop()

st.title("🕵️ Veritas: Analiză Automată Știri")
st.markdown("**Sistem de Licență** - Detecție Multimodală (Text + Imagine)")
st.markdown("---")

# TAB-uri
tab1, tab2 = st.tabs(["🔗 Analiză Link (Automat)", "📂 Încărcare Manuală"])

final_image = None
final_text = None
start_analysis = False

# --- TAB 1: AUTOMAT ---
with tab1:
    st.info("Lipește un link de la Digi24, Recorder sau un site de satiră.")
    col_url, col_btn = st.columns([3, 1])
    with col_url:
        url_input = st.text_input("URL Articol", label_visibility="collapsed", placeholder="https://...")
    with col_btn:
        scrape_btn = st.button("Extrage Date", key="btn_scrape", use_container_width=True)

    if scrape_btn:
        if not url_input:
            st.warning("Introduceți un link.")
        else:
            with st.spinner('Analizez structura paginii web...'):
                extracted_text, extracted_image, error = scrape_article(url_input)

                if error:
                    st.error(error)
                else:
                    final_text = extracted_text
                    final_image = extracted_image

                    if not final_image:
                        st.warning("Am găsit textul, dar nu și o imagine relevantă. Rezultatul va fi mai puțin precis.")
                        # Creăm o imagine neagră dummy dacă nu găsim una, ca să nu crape modelul
                        final_image = Image.new('RGB', (224, 224), color='black')

                    start_analysis = True

                    # Preview
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.image(final_image, caption="Imagine Identificată", use_container_width=True)
                    with c2:
                        st.text_area("Text Identificat", final_text, height=200)

# --- TAB 2: MANUAL ---
with tab2:
    uploaded_file = st.file_uploader("Încarcă o imagine", type=["jpg", "png", "jpeg"])
    manual_text = st.text_area("Scrie textul știrii aici", height=150)

    if st.button("Analizează Manual", key="btn_manual", type="primary"):
        if uploaded_file and manual_text:
            final_image = Image.open(uploaded_file).convert("RGB")
            final_text = manual_text
            start_analysis = True
        else:
            st.warning("Te rog încarcă și imaginea și textul.")

# --- ANALIZA FINALĂ ---
if start_analysis and final_text and final_image:
    st.markdown("---")
    st.subheader("🔍 Rezultat Analiză Veritas")

    with st.spinner("Procesare neuroni (BERT + ViT)..."):
        try:
            # 1. Procesare Text
            text_inputs = tokenizer(
                final_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            )

            # 2. Procesare Imagine
            image_inputs = image_processor(images=final_image, return_tensors="pt")

            # 3. Mutare pe GPU/CPU
            input_ids = text_inputs['input_ids'].to(DEVICE)
            attention_mask = text_inputs['attention_mask'].to(DEVICE)
            pixel_values = image_inputs['pixel_values'].to(DEVICE)

            # 4. Inferență
            with torch.no_grad():
                logits = model(input_ids, attention_mask, pixel_values)
                probs = F.softmax(logits, dim=1)

            # Interpretare
            # Clasa 0 = Real, Clasa 1 = Fake
            prob_real = probs[0][0].item()
            prob_fake = probs[0][1].item()

            # Afișare rezultate
            col_res1, col_res2, col_res3 = st.columns([2, 1, 1])

            with col_res1:
                if prob_fake > 0.50:
                    st.error(f"🚨 **VERDICT: FAKE NEWS / SATIRĂ**")
                    st.write(f"Sistemul este **{prob_fake * 100:.1f}%** sigur că aceasta este o știre falsă.")
                    st.progress(prob_fake)
                else:
                    st.success(f"✅ **VERDICT: ȘTIRE REALĂ**")
                    st.write(f"Sistemul este **{prob_real * 100:.1f}%** sigur că sursa este legitimă.")
                    st.progress(prob_real)

            with col_res2:
                st.metric("Scor FAKE", f"{prob_fake:.2%}")

            with col_res3:
                st.metric("Scor REAL", f"{prob_real:.2%}")

        except Exception as e:
            st.error(f"Eroare internă model: {e}")