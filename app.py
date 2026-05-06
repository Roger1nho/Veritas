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
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import gdown


st.set_page_config(
    page_title="Veritas — Detector Fake News",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

model_path = "veritas_model.pth"
# Verifică te rog dacă acel "_" de la final face parte cu adevărat din ID, e cam neobișnuit.
file_id = "1oBTuNksGw6RcTA8O_rR8Gk2tzlSUbqj_"

if not os.path.exists(model_path):
    # Folosim st.spinner ca să vadă utilizatorul de ce durează la prima pornire
    with st.spinner("Serverul descarcă modelul AI (procesul durează ~1-2 minute la prima rulare)..."):
        # Folosim id= (nu url=) pentru a trece peste avertismentul de virus scan al Google
        gdown.download(id=file_id, output=model_path, quiet=False)

# Apoi continui codul tau normal...
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# ─────────────────────────────────────────────
# CONFIG PAGINĂ
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# CSS GLOBAL — temă dark, tipografie, carduri
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:       #0d0f14;
    --surface:  #141720;
    --border:   #1e2330;
    --accent:   #e8ff3c;
    --fake:     #ff4c6a;
    --real:     #3cffa0;
    --muted:    #5a6070;
    --text:     #e8eaf0;
}

html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Header principal */
.veritas-hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.veritas-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -2px;
    color: var(--text);
    margin: 0;
}
.veritas-hero h1 span { color: var(--accent); }
.veritas-hero p {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* Carduri statistici */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.stat-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.stat-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}

/* Verdict banner */
.verdict-fake {
    background: linear-gradient(135deg, #1a0810 0%, #2d0f18 100%);
    border: 1px solid var(--fake);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}
.verdict-real {
    background: linear-gradient(135deg, #081a10 0%, #0f2d18 100%);
    border: 1px solid var(--real);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}
.verdict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
}
.verdict-text {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0.2rem 0;
}
.verdict-fake .verdict-text { color: var(--fake); }
.verdict-real .verdict-text { color: var(--real); }
.verdict-sub {
    font-size: 0.88rem;
    color: var(--muted);
}

/* Bare progress custom */
.progress-wrapper { margin: 0.6rem 0; }
.progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    margin-bottom: 4px;
}
.progress-bar-bg {
    background: var(--border);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Butoane */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    background: #d4e836 !important;
    transform: translateY(-1px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 7px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--border) !important;
    color: var(--text) !important;
}

/* Input */
.stTextInput input, .stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,255,60,0.15) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* Divider */
hr { border-color: var(--border) !important; opacity: 0.5 !important; }

/* Metric */
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
}

.tag {
    display: inline-block;
    background: var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin: 2px;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.info-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: var(--muted);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "session_stats" not in st.session_state:
    st.session_state.session_stats = {"fake": 0, "real": 0, "scores": []}

# ─────────────────────────────────────────────
# IMPORTS MODEL
# ─────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from fusion_model import MultimodalFakeNewsModel
except ImportError:
    st.error("❌ Nu găsesc `src/fusion_model.py`.")
    st.stop()

MODEL_PATH = "src/veritas_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# FUNCȚII HELPER
# ─────────────────────────────────────────────
def clean_for_pdf(s):
    if not s: return ""
    for ro, en in {'ă':'a','â':'a','î':'i','ș':'s','ț':'t','Ă':'A','Â':'A',
                   'Î':'I','Ș':'S','Ț':'T','ş':'s','ţ':'t','Ş':'S','Ţ':'T'}.items():
        s = s.replace(ro, en)
    return s.encode('latin-1', 'ignore').decode('latin-1')


def generate_pdf_report(text, prob_fake, prob_real, image_weight, verdict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Raport Analiza Veritas - XAI", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt=f"Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. Statistici Predictie", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"Verdict: {clean_for_pdf(verdict)}", ln=True)
    pdf.cell(200, 8, txt=f"Probabilitate FAKE: {prob_fake:.2%}", ln=True)
    pdf.cell(200, 8, txt=f"Probabilitate REAL: {prob_real:.2%}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Explicabilitate XAI", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, txt=clean_for_pdf(
        f"Influenta Imaginii: {image_weight:.2%}. Influenta Textului: {1-image_weight:.2%}."))
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="3. Continut Analizat", ln=True)
    pdf.set_font("Arial", '', 10)
    safe = clean_for_pdf(text[:1500]) + ("..." if len(text) > 1500 else "")
    pdf.multi_cell(0, 6, txt=safe)
    return pdf.output(dest="S").encode("latin-1")


@st.cache_resource
def load_veritas_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = MultimodalFakeNewsModel(
        num_labels=2,
        text_model_name="dumitrescustefan/bert-base-romanian-cased-v1"
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    return model, tokenizer, processor


def scrape_article(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None, None, "Nu am putut accesa site-ul."
        text_content = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if not text_content:
            return None, None, "Text insuficient extras."
        metadata = trafilatura.extract_metadata(downloaded)
        title_text = metadata.title if metadata and metadata.title else ""
        image_url = metadata.image if metadata and metadata.image else ""
        if not image_url:
            headers = {'User-Agent': 'Mozilla/5.0'}
            try:
                r = requests.get(url, headers=headers, timeout=5)
                soup = BeautifulSoup(r.content, 'html.parser')
                meta_image = soup.find('meta', property='og:image')
                if meta_image: image_url = meta_image['content']
            except: pass
        pil_image = None
        if image_url:
            try:
                if not image_url.startswith('http'):
                    from urllib.parse import urljoin
                    image_url = urljoin(url, image_url)
                img_resp = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                pil_image = Image.open(BytesIO(img_resp.content)).convert("RGB")
            except: pass
        return f"{title_text}. {text_content}", pil_image, None
    except Exception as e:
        return None, None, f"Eroare: {str(e)}"


def make_gauge(value, color, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 28, 'color': color, 'family': 'Syne'}},
        title={'text': title, 'font': {'size': 13, 'color': '#5a6070', 'family': 'IBM Plex Mono'}},
        gauge={
            # Am schimbat 'transparent' cu 'rgba(0,0,0,0)' aici:
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': 'rgba(0,0,0,0)',
                     'tickfont': {'color': '#5a6070', 'size': 10}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#1e2330',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': '#141720'},
                {'range': [50, 100], 'color': '#1a1e28'}
            ],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.8, 'value': value * 100}
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e8eaf0'
    )
    return fig


def make_donut(image_w, text_w):
    fig = go.Figure(go.Pie(
        values=[image_w * 100, text_w * 100],
        labels=['Imagine', 'Text'],
        hole=0.65,
        marker_colors=['#e8ff3c', '#3cffa0'],
        textinfo='none',
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            font=dict(family='IBM Plex Mono', size=11, color='#5a6070'),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True
    )
    return fig


def make_history_chart(history):
    if len(history) < 2:
        return None
    df = pd.DataFrame([{
        'nr': i + 1,
        'fake_score': h['prob_fake'] * 100,
        'verdict': h['verdict']
    } for i, h in enumerate(history)])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['nr'], y=df['fake_score'],
        mode='lines+markers',
        line=dict(color='#e8ff3c', width=2),
        marker=dict(
            color=['#ff4c6a' if v == 'FAKE NEWS' else '#3cffa0' for v in df['verdict']],
            size=10, line=dict(color='#141720', width=2)
        ),
        hovertemplate='Analiza #%{x}<br>Scor FAKE: %{y:.1f}%<extra></extra>'
    ))
    fig.add_hline(y=50, line_dash='dash', line_color='#5a6070', line_width=1)
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(family='IBM Plex Mono', size=10, color='#5a6070'),
            title=dict(text='Analiza #', font=dict(color='#5a6070', size=10))
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#1e2330', zeroline=False,
            tickfont=dict(family='IBM Plex Mono', size=10, color='#5a6070'),
            range=[0, 100]
        )
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem; border-bottom: 1px solid #1e2330; margin-bottom: 1rem;">
        <div style="font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 800; color: #e8eaf0;">
            VERITAS <span style="color: #e8ff3c;">·</span>
        </div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #5a6070; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px;">
            Fake News Detector
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Statistici sesiune
    total = len(st.session_state.history)
    n_fake = st.session_state.session_stats["fake"]
    n_real = st.session_state.session_stats["real"]

    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 1rem;">
        <div class="stat-card" style="text-align:center; padding: 0.8rem;">
            <div class="label">Total</div>
            <div class="value" style="font-size:1.6rem;">{total}</div>
        </div>
        <div class="stat-card" style="text-align:center; padding: 0.8rem;">
            <div class="label">Fake</div>
            <div class="value" style="font-size:1.6rem; color:#ff4c6a;">{n_fake}</div>
        </div>
        <div class="stat-card" style="text-align:center; padding: 0.8rem;">
            <div class="label">Real</div>
            <div class="value" style="font-size:1.6rem; color:#3cffa0;">{n_real}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Grafic evoluție sesiune
    if len(st.session_state.history) >= 2:
        chart = make_history_chart(st.session_state.history)
        if chart:
            st.markdown('<div class="section-title" style="font-size:0.8rem;">Evoluție sesiune</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})

    # Istoric
    st.markdown('<div class="section-title" style="font-size:0.8rem;">Istoric analize</div>',
                unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            '<div class="info-box">Nicio analiză în această sesiune.</div>',
            unsafe_allow_html=True
        )
    else:
        for item in reversed(st.session_state.history[-8:]):
            icon = "🚨" if item['fake'] else "✅"
            color = "#ff4c6a" if item['fake'] else "#3cffa0"
            score = item.get('prob_fake', 0.5)
            st.markdown(f"""
            <div style="background:#141720; border:1px solid #1e2330; border-left: 3px solid {color};
                        border-radius:8px; padding:0.6rem 0.8rem; margin-bottom:6px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#5a6070;">
                        {item['time']}
                    </span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:{color};">
                        {score:.0%}
                    </span>
                </div>
                <div style="font-family:'Syne',sans-serif; font-size:0.82rem; color:{color}; font-weight:700; margin:2px 0;">
                    {icon} {item['verdict']}
                </div>
                <div style="font-size:0.72rem; color:#5a6070; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    {item['snippet']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Info model
    st.markdown("""
    <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid #1e2330;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#5a6070; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Model Info</div>
        <span class="tag">RoBERT-ro</span>
        <span class="tag">ViT-B/16</span>
        <span class="tag">Gated Fusion</span>
        <span class="tag">2-class</span>
    </div>
    """, unsafe_allow_html=True)

    device_color = "#e8ff3c" if DEVICE == "cuda" else "#5a6070"
    st.markdown(f"""
    <div style="margin-top:0.6rem;">
        <span class="tag" style="color:{device_color};">⚡ {DEVICE.upper()}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="veritas-hero">
    <h1>VER<span>ITAS</span></h1>
    <p>Multimodal Fake News Detection · Romanian Language · BERT + ViT · Gated Fusion</p>
</div>
""", unsafe_allow_html=True)

# Inițializare model
model, tokenizer, image_processor = load_veritas_model()

if model is None:
    st.markdown("""
    <div class="info-box" style="border-left-color:#ff4c6a; margin-bottom:1rem;">
        ⚠️ Model neîncărcat. Pune <code>veritas_model.pth</code> în <code>src/</code> și repornește.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS INPUT
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔗  Analiză URL", "📂  Upload Manual"])

final_image = None
final_text = None
start_analysis = False

with tab1:
    st.markdown('<div class="info-box">Introdu link-ul unui articol în română. Veritas extrage automat textul și imaginea principală.</div>',
                unsafe_allow_html=True)
    st.write("")
    col_url, col_btn = st.columns([4, 1])
    with col_url:
        url_input = st.text_input("URL", label_visibility="collapsed",
                                  placeholder="https://digi24.ro/stiri/...")
    with col_btn:
        scrape_btn = st.button("Analizează →", key="btn_scrape", use_container_width=True)

    if scrape_btn and url_input:
        with st.spinner('Se extrag date din articol...'):
            extracted_text, extracted_image, error = scrape_article(url_input)
            if error:
                st.error(error)
            else:
                final_text = extracted_text
                final_image = extracted_image or Image.new('RGB', (224, 224), color=(13, 15, 20))
                start_analysis = True

with tab2:
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        uploaded_file = st.file_uploader("Imagine articol", type=["jpg", "png", "jpeg"])
    with col_m2:
        manual_text = st.text_area("Text știre", height=120,
                                   placeholder="Paste textul articolului aici...")
    if st.button("Analizează Manual →", key="btn_manual", use_container_width=False):
        if uploaded_file and manual_text:
            final_image = Image.open(uploaded_file).convert("RGB")
            final_text = manual_text
            start_analysis = True
        else:
            st.warning("Încarcă o imagine ȘI scrie un text.")

# ─────────────────────────────────────────────
# ANALIZĂ & REZULTATE
# ─────────────────────────────────────────────
if start_analysis and final_text and final_image and model is not None:

    # Preview articol
    with st.expander("📖 Conținut extras din articol", expanded=False):
        p1, p2 = st.columns([1, 2])
        with p1:
            st.image(final_image, use_container_width=True)
        with p2:
            st.text_area("Text", final_text, height=180, disabled=True)

    st.markdown("---")

    # INFERENTA
    with st.spinner("Procesare neuroni..."):
        text_inputs = tokenizer(final_text, return_tensors="pt", truncation=True,
                                padding="max_length", max_length=512)
        image_inputs = image_processor(images=final_image, return_tensors="pt")

        input_ids = text_inputs['input_ids'].to(DEVICE)
        attention_mask = text_inputs['attention_mask'].to(DEVICE)
        pixel_values = image_inputs['pixel_values'].to(DEVICE)

        with torch.no_grad():
            logits, z_gate = model(input_ids, attention_mask, pixel_values)
            probs = F.softmax(logits, dim=1)

        prob_real = probs[0][0].item()
        prob_fake = probs[0][1].item()
        image_influence = z_gate[0][0].item()
        text_influence = 1.0 - image_influence
        is_fake = prob_fake > 0.50
        verdict_text = "FAKE NEWS" if is_fake else "ȘTIRE REALĂ"
        logit_real = logits[0][0].item()
        logit_fake = logits[0][1].item()

        # Update session stats
        st.session_state.session_stats["fake" if is_fake else "real"] += 1
        st.session_state.session_stats["scores"].append(prob_fake)
        st.session_state.history.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "verdict": verdict_text,
            "fake": is_fake,
            "prob_fake": prob_fake,
            "snippet": final_text[:55] + "..."
        })

    # ── VERDICT BANNER ───────────────────────────────────────────
    verdict_class = "verdict-fake" if is_fake else "verdict-real"
    verdict_icon = "🚨" if is_fake else "✅"
    confidence = prob_fake if is_fake else prob_real
    detail = (f"Modelul este <strong>{confidence*100:.1f}%</strong> sigur că aceasta este o manipulare."
              if is_fake else
              f"Modelul este <strong>{confidence*100:.1f}%</strong> sigur că sursa este legitimă.")

    st.markdown(f"""
    <div class="{verdict_class}">
        <div class="verdict-label">Verdict Final</div>
        <div class="verdict-text">{verdict_icon} {verdict_text}</div>
        <div class="verdict-sub">{detail}</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ── RÂND 1: Gauge-uri + donut XAI ───────────────────────────
    col_g1, col_g2, col_g3 = st.columns([1, 1, 1])

    with col_g1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(prob_fake, "#ff4c6a", "SCOR FAKE"),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_g2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(prob_real, "#3cffa0", "SCOR REAL"),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_g3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="label" style="font-family:\'IBM Plex Mono\',monospace; font-size:0.68rem; letter-spacing:2px; text-transform:uppercase; color:#5a6070; margin-bottom:4px;">PONDERE MODALĂ (XAI)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_donut(image_influence, text_influence),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RÂND 2: Bare orizontale + stats text + PDF ───────────────
    st.write("")
    col_bars, col_meta = st.columns([1, 1])

    with col_bars:
        st.markdown('<div class="section-title">Distribuție scor</div>', unsafe_allow_html=True)

        fake_pct = int(prob_fake * 100)
        real_pct = int(prob_real * 100)
        img_pct = int(image_influence * 100)
        txt_pct = int(text_influence * 100)

        st.markdown(f"""
        <div class="progress-wrapper">
            <div class="progress-label">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#ff4c6a;">FAKE</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#5a6070;">{fake_pct}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width:{fake_pct}%; background:#ff4c6a;"></div>
            </div>
        </div>
        <div class="progress-wrapper">
            <div class="progress-label">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#3cffa0;">REAL</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#5a6070;">{real_pct}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width:{real_pct}%; background:#3cffa0;"></div>
            </div>
        </div>
        <div style="margin-top:1.2rem;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#5a6070; letter-spacing:1px; text-transform:uppercase; margin-bottom:0.6rem;">Pondere modală (XAI)</div>
        </div>
        <div class="progress-wrapper">
            <div class="progress-label">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#e8ff3c;">📸 Imagine</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#5a6070;">{img_pct}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width:{img_pct}%; background:#e8ff3c;"></div>
            </div>
        </div>
        <div class="progress-wrapper">
            <div class="progress-label">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#3cffa0;">📝 Text</span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#5a6070;">{txt_pct}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width:{txt_pct}%; background:#3cffa0;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_meta:
        st.markdown('<div class="section-title">Date tehnice</div>', unsafe_allow_html=True)
        words = final_text.split()
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:0.5rem;">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;">
                <div>
                    <div class="label">Cuvinte</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700;">{len(words)}</div>
                </div>
                <div>
                    <div class="label">Caractere</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700;">{len(final_text)}</div>
                </div>
                <div>
                    <div class="label">Logit [FAKE]</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem; color:#ff4c6a;">{logit_fake:.4f}</div>
                </div>
                <div>
                    <div class="label">Logit [REAL]</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem; color:#3cffa0;">{logit_real:.4f}</div>
                </div>
                <div>
                    <div class="label">Rezoluție img</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#5a6070;">{final_image.width}×{final_image.height}px</div>
                </div>
                <div>
                    <div class="label">Device</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#e8ff3c;">{DEVICE.upper()}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pdf_bytes = generate_pdf_report(final_text, prob_fake, prob_real, image_influence, verdict_text)
        st.download_button(
            label="📄 Descarcă Raport PDF",
            data=pdf_bytes,
            file_name=f"Veritas_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # ── XAI EXPLICAȚIE ───────────────────────────────────────────
    st.write("")
    with st.expander("🧠 Cum a decis modelul? (XAI Explicat)", expanded=False):
        dominant = "imagine" if image_influence > text_influence else "text"
        dominant_pct = max(image_influence, text_influence)
        st.markdown(f"""
        <div class="info-box">
            <strong>Mecanism:</strong> Gated Fusion — modelul calculează un scor <code>z ∈ [0,1]</code> care
            reprezintă cât de mult contează imaginea față de text în decizia finală.<br><br>
            <strong>În acest articol:</strong> Modelul a ponderat predominant <strong>{dominant}ul</strong>
            ({dominant_pct:.1%}) pentru a trage concluzia.<br><br>
            {'⚡ Imaginea conținea probabil semnale vizuale puternice (imagine scoasă din context, editată, clickbait vizual).'
             if dominant == 'imagine' else
             '⚡ Limbajul textului a fost factorul decisiv — structura frazelor, tonul sau vocabularul au declanșat semnale de dezinformare.'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1rem; display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
            <div class="stat-card">
                <div class="label">Formula fuziune</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.82rem; color:#e8ff3c; margin-top:4px;">
                    fused = text + (z × img_proj)
                </div>
            </div>
            <div class="stat-card">
                <div class="label">Valoare z (gate)</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.82rem; color:#e8ff3c; margin-top:4px;">
                    z = {image_influence:.4f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)