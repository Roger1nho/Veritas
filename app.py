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
import trafilatura
from fpdf import FPDF
import datetime

st.set_page_config(
    page_title="Veritas - Detector Fake News",
    page_icon="🕵️",
    layout="wide"
)

# Adăugăm folderul 'src' la calea sistemului pentru a găsi fusion_model.py
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from fusion_model import MultimodalFakeNewsModel
except ImportError:
    st.error("❌ Nu găsesc fișierul 'src/fusion_model.py'. Asigură-te că folderul 'src' este lângă 'app.py'.")
    st.stop()

# Folosim calea relativă pentru portabilitate
MODEL_PATH = "src/veritas_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_pdf_report(text, prob_fake, prob_real, image_weight, verdict):
    """Generează un raport PDF cu rezultatele analizei."""
    pdf = FPDF()
    pdf.add_page()

    # Titlu
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Raport Analiza Veritas - XAI", ln=True, align='C')
    pdf.ln(10)

    # Metadate
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt=f"Data generarii: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    # Statistici Principale
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. Statistici Predictie", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"Verdict Final: {verdict}", ln=True)
    pdf.cell(200, 8, txt=f"Probabilitate FAKE: {prob_fake:.2%}", ln=True)
    pdf.cell(200, 8, txt=f"Probabilitate REAL: {prob_real:.2%}", ln=True)

    # Explicabilitate (XAI)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Explicabilitate Multimodala (XAI)", ln=True)
    pdf.set_font("Arial", '', 11)

    # Explicație text pentru valoarea Z
    infl_text = f"Influenta Imaginii in decizia finala: {image_weight:.2%}. "
    if image_weight < 0.5:
        infl_text += "Modelul s-a bazat preponderent pe analiza textului pentru a lua decizia."
    else:
        infl_text += "Imaginea a jucat un rol major in decizia finala a modelului."

    pdf.multi_cell(0, 8, txt=infl_text)

    # Textul analizat
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="3. Continut Analizat (Extras)", ln=True)
    pdf.set_font("Arial", '', 10)

    # Curățăm textul de caractere care ar putea bloca FPDF
    safe_text = text[:1500].encode('latin-1', 'replace').decode('latin-1') + ("..." if len(text) > 1500 else "")
    pdf.multi_cell(0, 6, txt=safe_text)

    # Exportăm în format byte
    return pdf.output(dest="S").encode("latin-1")


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

    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    return model, tokenizer, processor


def scrape_article(url):
    """Extrage textul CURAT și imaginea principală dintr-un URL."""
    try:
        downloaded = trafilatura.fetch_url(url)

        if downloaded is None:
            return None, None, "Nu am putut accesa site-ul (blocaj sau link invalid)."

        text_content = trafilatura.extract(downloaded, include_comments=False, include_tables=False)

        if not text_content or len(text_content) < 50:
            return None, None, "Text insuficient extras. Site-ul poate avea protecție avansată."

        metadata = trafilatura.extract_metadata(downloaded)
        title_text = metadata.title if metadata and metadata.title else ""
        image_url = metadata.image if metadata and metadata.image else ""

        if not image_url:
            headers = {'User-Agent': 'Mozilla/5.0'}
            try:
                r = requests.get(url, headers=headers, timeout=5)
                soup = BeautifulSoup(r.content, 'html.parser')
                meta_image = soup.find('meta', property='og:image')
                if meta_image:
                    image_url = meta_image['content']
            except:
                pass

        pil_image = None
        if image_url:
            try:
                if not image_url.startswith('http'):
                    from urllib.parse import urljoin
                    image_url = urljoin(url, image_url)

                headers = {'User-Agent': 'Mozilla/5.0'}
                img_resp = requests.get(image_url, headers=headers, timeout=5)
                pil_image = Image.open(BytesIO(img_resp.content)).convert("RGB")
            except Exception as e:
                print(f"Eroare download imagine: {e}")
                pil_image = None

        full_text = f"{title_text}. {text_content}"

        return full_text, pil_image, None

    except Exception as e:
        return None, None, f"Eroare generală: {str(e)}"


# Inițializare model
model, tokenizer, image_processor = load_veritas_model()

if not model:
    st.stop()

st.title("🕵️ Veritas: Analiză Automată Știri")
st.markdown("Detecție Multimodală cu Explicabilitate (XAI)")
st.markdown("---")

# TAB-uri
tab1, tab2 = st.tabs(["🔗 Analiză Link (Automat)", "📂 Încărcare Manuală"])

final_image = None
final_text = None
start_analysis = False

with tab1:
    st.info("Dă paste la linkul oricărui articol scris în română.")
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
                        final_image = Image.new('RGB', (224, 224), color='black')

                    start_analysis = True

                    # Preview
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.image(final_image, caption="Imagine Identificată", use_container_width=True)
                    with c2:
                        st.text_area("Text Identificat", final_text, height=200)

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

# Secțiunea de Inferență și Rezultate
if start_analysis and final_text and final_image:
    st.markdown("---")
    st.subheader("🔍 Rezultat Analiză Veritas")

    with st.spinner("Procesare neuroni (BERT + ViT)..."):
        try:
            # Procesare Text
            text_inputs = tokenizer(
                final_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            )

            # Procesare Imagine
            image_inputs = image_processor(images=final_image, return_tensors="pt")

            # Mutare pe GPU/CPU
            input_ids = text_inputs['input_ids'].to(DEVICE)
            attention_mask = text_inputs['attention_mask'].to(DEVICE)
            pixel_values = image_inputs['pixel_values'].to(DEVICE)

            # Inferență - Despachetăm tuplul!
            with torch.no_grad():
                logits, z_gate = model(input_ids, attention_mask, pixel_values)
                probs = F.softmax(logits, dim=1)

            # Interpretare
            prob_real = probs[0][0].item()
            prob_fake = probs[0][1].item()
            image_influence = z_gate[0][0].item()  # Extragem ponderea imaginii

            verdict_text = "FAKE NEWS" if prob_fake > 0.50 else "ȘTIRE REALĂ"

            # Afișare rezultate principale
            col_res1, col_res2, col_res3 = st.columns([2, 1, 1])

            with col_res1:
                if prob_fake > 0.50:
                    st.error(f"🚨 **VERDICT: {verdict_text}**")
                    st.write(f"Sistemul este **{prob_fake * 100:.1f}%** sigur că aceasta este o știre falsă.")
                    st.progress(prob_fake)
                else:
                    st.success(f"✅ **VERDICT: {verdict_text}**")
                    st.write(f"Sistemul este **{prob_real * 100:.1f}%** sigur că sursa este legitimă.")
                    st.progress(prob_real)

            with col_res2:
                st.metric("Scor FAKE", f"{prob_fake:.2%}")
            with col_res3:
                st.metric("Scor REAL", f"{prob_real:.2%}")

            st.markdown("---")

            # Secțiunea XAI (Explicabilitate) și PDF
            st.markdown("### 🧠 Explicabilitatea Modelului (XAI)")
            col_xai1, col_xai2 = st.columns(2)

            with col_xai1:
                st.info("**Analiza Fuziunii (Text vs. Imagine)**")
                st.write(f"Cât a influențat **imaginea** decizia finală: **{image_influence:.2%}**")
                st.progress(image_influence)
                if image_influence > 0.5:
                    st.write("📸 *Modelul a detectat semnale puternice în imagine care i-au influențat decizia.*")
                else:
                    st.write("📝 *Modelul s-a bazat aproape exclusiv pe text.*")

            with col_xai2:
                st.info("**Export Rezultate**")
                st.write("Descarcă raportul detaliat în format PDF pentru arhivare sau partajare.")

                # Generare PDF folosind funcția nouă
                pdf_bytes = generate_pdf_report(
                    text=final_text,
                    prob_fake=prob_fake,
                    prob_real=prob_real,
                    image_weight=image_influence,
                    verdict=verdict_text
                )

                # Buton de download Streamlit
                st.download_button(
                    label="📄 Descarcă Raport PDF",
                    data=pdf_bytes,
                    file_name="Raport_Veritas.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Eroare internă model: {e}")