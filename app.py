
# ================================================================
# app.py — Fake News Detection — Streamlit App
# Author  : Your Name
# Dataset : Kaggle True.csv / Fake.csv
# Model   : Logistic Regression + TF-IDF
# ================================================================

import streamlit as st
import joblib
import json
import re
import string
import time
import os
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# ── NLTK ──────────────────────────────────────────────────────
import nltk
for resource in ['punkt', 'punkt_tab',
                 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource
                       else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title = "Fake News Detector",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ================================================================
# CUSTOM CSS — Professional Dark-Accented Theme
# ================================================================
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit header/footer ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Custom header ── */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1rem 1rem;
    margin-bottom: 1.5rem;
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    font-weight: 400;
    background: linear-gradient(90deg, #e2e8f0, #94a3b8, #e2e8f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.app-header .subtitle {
    color: #64748b;
    font-size: 1.05rem;
    font-weight: 400;
    letter-spacing: 0.3px;
}
.app-header .badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.25rem 0.85rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.2rem;
}

/* ── Result cards ── */
.result-real {
    background: linear-gradient(135deg,
        rgba(16,185,129,0.12) 0%,
        rgba(16,185,129,0.05) 100%);
    border: 1.5px solid rgba(16,185,129,0.35);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: fadeSlideUp 0.5s ease forwards;
}
.result-fake {
    background: linear-gradient(135deg,
        rgba(239,68,68,0.12) 0%,
        rgba(239,68,68,0.05) 100%);
    border: 1.5px solid rgba(239,68,68,0.35);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: fadeSlideUp 0.5s ease forwards;
}
.result-rejected {
    background: linear-gradient(135deg,
        rgba(245,158,11,0.10) 0%,
        rgba(245,158,11,0.04) 100%);
    border: 1.5px solid rgba(245,158,11,0.30);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: fadeSlideUp 0.5s ease forwards;
}

/* ── Result text ── */
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-weight: 400;
    line-height: 1.1;
    margin: 0.5rem 0;
}
.result-label-real  { color: #10b981; }
.result-label-fake  { color: #ef4444; }
.result-label-warn  { color: #f59e0b; }
.result-confidence  {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0.3rem 0 1rem 0;
    font-weight: 400;
}
.result-icon { font-size: 3.2rem; margin-bottom: 0.3rem; }

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 50px;
    height: 12px;
    width: 100%;
    margin: 0.8rem 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}

/* ── Prob pills ── */
.prob-row {
    display: flex;
    gap: 0.8rem;
    justify-content: center;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.prob-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.2px;
}
.pill-real {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #6ee7b7;
}
.pill-fake {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
}

/* ── Disclaimer strip ── */
.disclaimer {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1.1rem;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-top: 1rem;
    line-height: 1.5;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15,15,26,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
.sidebar-metric {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-metric .label {
    color: #64748b;
    font-size: 0.82rem;
}
.sidebar-metric .value {
    color: #e2e8f0;
    font-weight: 600;
    font-size: 0.9rem;
}

/* ── History item ── */
.hist-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #94a3b8;
}
.hist-item strong { color: #e2e8f0; }
.hist-real { border-left: 3px solid #10b981; }
.hist-fake { border-left: 3px solid #ef4444; }
.hist-rej  { border-left: 3px solid #f59e0b; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.45) !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Animations ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%,100% { transform: scale(1); }
    50%      { transform: scale(1.06); }
}
.pulse { animation: pulse 2s infinite; }

/* ── Section labels ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.6rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    gap: 2px;
    padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# PATHS — Adjust if running outside Colab
# ================================================================
BASE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH     = f"{BASE}/model.pkl"
VEC_PATH       = f"{BASE}/vectorizer.pkl"
LABEL_PATH     = f"{BASE}/label_encoder.pkl"
META_PATH      = f"{BASE}/model_metadata.json"


# ================================================================
# NLP PREPROCESSING (same pipeline as training)
# ================================================================
@st.cache_resource(show_spinner=False)
def load_nlp_tools():
    lemmatizer = WordNetLemmatizer()
    sw         = set(stopwords.words('english'))
    return lemmatizer, sw

lemmatizer, STOP_WORDS = load_nlp_tools()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t)
              for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


# ================================================================
# LOAD MODEL ASSETS
# ================================================================
@st.cache_resource(show_spinner=False)
def load_assets():
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    labels     = joblib.load(LABEL_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
    return model, vectorizer, labels, metadata

try:
    model, vectorizer, label_map, metadata = load_assets()
    THRESHOLD  = metadata['deployment_config']['confidence_threshold']
    MIN_WORDS  = metadata['deployment_config']['min_word_count']
    ASSETS_OK  = True
except Exception as e:
    ASSETS_OK  = False
    LOAD_ERROR = str(e)


# ================================================================
# PREDICTION PIPELINE
# ================================================================
def predict_news(raw_text: str) -> dict:
    """Full prediction pipeline with all safety guards."""

    result = dict(status=None, label=None, confidence=None,
                  prob_real=None, prob_fake=None,
                  cleaned=None, word_count=0,
                  message=None, reason=None)

    # Guard 1 – empty
    if not isinstance(raw_text, str) or not raw_text.strip():
        result.update(status='rejected', reason='empty',
            message='⚠️ Please enter some text first.')
        return result

    # Guard 2 – too short (raw)
    if len(raw_text.split()) < 5:
        result.update(status='rejected', reason='too_short',
            message=('⚠️ Input is too short.\n\n'
                     'Please paste at least a few sentences from the news article.'))
        return result

    # Preprocess
    cleaned = preprocess_text(raw_text)
    result['cleaned']    = cleaned
    result['word_count'] = len(cleaned.split())

    # Guard 3 – empty after cleaning
    if not cleaned.strip():
        result.update(status='rejected', reason='empty_after_clean',
            message=('⚠️ Could not extract meaningful content.\n\n'
                     'Please enter a real news article in English.'))
        return result

    # Guard 4 – too few meaningful words
    if result['word_count'] < MIN_WORDS:
        result.update(status='rejected', reason='too_few_words',
            message=(f'⚠️ Only {result["word_count"]} meaningful '
                     f'words detected (minimum {MIN_WORDS}).\n\n'
                     'Please provide a longer article.'))
        return result

    # Vectorize
    vec = vectorizer.transform([cleaned])

    # Guard 5 – out of vocabulary
    if vec.nnz == 0:
        result.update(status='rejected', reason='oov',
            message=('🤔 I cannot confidently predict this because the input appears '
                     'outside the trained dataset domain.\n\n'
                     'This model is trained on English political and world news. '
                     'Please enter a relevant news article.'))
        return result

    # Predict
    pred        = model.predict(vec)[0]
    proba       = model.predict_proba(vec)[0]
    prob_fake   = float(proba[0])
    prob_real   = float(proba[1])
    confidence  = max(prob_fake, prob_real)

    result['prob_fake']  = prob_fake
    result['prob_real']  = prob_real
    result['confidence'] = confidence

    # Guard 6 – low confidence
    if confidence < THRESHOLD:
        result.update(status='rejected', reason='low_confidence',
            message=(f'🤔 I cannot confidently predict this because the input appears '
                     f'outside the trained dataset domain.\n\n'
                     f'Confidence: {confidence*100:.1f}% (below the {THRESHOLD*100:.0f}% threshold).\n\n'
                     f'Please enter a complete English news article about political '
                     f'or world events.'))
        return result

    # Success
    label     = 'REAL' if pred == 1 else 'FAKE'
    certainty = ('Very High' if confidence >= 0.95 else
                 'High'      if confidence >= 0.85 else
                 'Moderate'  if confidence >= 0.70 else
                 'Borderline')

    result.update(status='success', label=label,
                  prediction=int(pred), certainty=certainty)
    return result


# ================================================================
# SESSION STATE — Prediction history
# ================================================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:

    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem 0;'>
        <span style='font-size:2.2rem;'>🔍</span>
        <div style='font-family:"DM Serif Display",serif;
                    font-size:1.2rem; color:#e2e8f0;
                    margin-top:0.3rem;'>
            Fake News Detector
        </div>
        <div style='color:#475569; font-size:0.78rem;
                    margin-top:0.2rem;'>
            NLP + Machine Learning
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # ── Model metrics ──
    st.markdown(
        '<div class="section-label">Model Performance</div>',
        unsafe_allow_html=True)

    if ASSETS_OK:
        perf = metadata['performance']
        metrics = [
            ("Accuracy",  f"{perf['test_accuracy']*100:.2f}%"),
            ("Macro F1",  f"{perf['macro_f1']*100:.2f}%"),
            ("ROC-AUC",   f"{perf['roc_auc']:.4f}"),
            ("Test Size", f"{perf['test_size']:,}"),
            ("Errors",    f"{perf['total_test_errors']}"),
        ]
        for label, val in metrics:
            st.markdown(f"""
            <div class="sidebar-metric">
                <span class="label">{label}</span>
                <span class="value">{val}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.error("Models not loaded")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Model details ──
    st.markdown(
        '<div class="section-label">Model Details</div>',
        unsafe_allow_html=True)

    if ASSETS_OK:
        mi = metadata['model_info']
        vi = metadata['vectorizer_info']
        dc = metadata['deployment_config']
        details = [
            ("Algorithm",  mi['name']),
            ("Features",   f"{vi['vocabulary_size']:,}"),
            ("N-grams",    str(vi['ngram_range'])),
            ("Threshold",  f"{dc['confidence_threshold']*100:.0f}%"),
            ("Domain",     dc['domain']),
            ("Language",   dc['language']),
        ]
        for label, val in details:
            st.markdown(f"""
            <div class="sidebar-metric">
                <span class="label">{label}</span>
                <span class="value">{val}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Prediction history ──
    st.markdown(
        '<div class="section-label">Recent Predictions</div>',
        unsafe_allow_html=True)

    if st.session_state.history:
        for h in reversed(st.session_state.history[-6:]):
            if h['status'] == 'success':
                cls = 'hist-real' if h['label'] == 'REAL'                       else 'hist-fake'
                icon = '✅' if h['label'] == 'REAL' else '🚫'
                conf = f"{h['confidence']*100:.1f}%"
                preview = h['text'][:45] + '..'
                st.markdown(f"""
                <div class="hist-item {cls}">
                    {icon} <strong>{h['label']}</strong>
                    ({conf})<br>{preview}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="hist-item hist-rej">
                    ⚠️ <strong>Rejected</strong><br>
                    {h['text'][:45]}..
                </div>""", unsafe_allow_html=True)

        if st.button("🗑  Clear History", use_container_width=True):
            st.session_state.history  = []
            st.session_state.last_result = None
            st.rerun()
    else:
        st.markdown(
            '<div style="color:#475569; font-size:0.83rem; text-align:center; padding:0.5rem;">'
            'No predictions yet.</div>',
            unsafe_allow_html=True)


# ================================================================
# MAIN CONTENT
# ================================================================

# ── Header ──
st.markdown("""
<div class="app-header">
    <div class="badge">AI-Powered Detection</div>
    <h1>🔍 Fake News Detector</h1>
    <div class="subtitle">
        Paste a news article below and our ML model will
        classify it as <strong style="color:#10b981">Real</strong>
        or <strong style="color:#ef4444">Fake</strong>
        — with a confidence score.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Asset load check ──
if not ASSETS_OK:
    st.error(f"""
    ❌ **Failed to load model files.**
    Error: `{LOAD_ERROR}`

    **Fix:** Make sure these files exist in `{BASE}`:
    - `model.pkl`
    - `vectorizer.pkl`
    - `label_encoder.pkl`
    - `model_metadata.json`

    Re-run Steps 9 of the notebook to regenerate them.
    """)
    st.stop()

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(
    ["🔍  Detect", "📊  Analysis", "ℹ️  About"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — DETECT
# ════════════════════════════════════════════════════════════════
with tab1:

    col_left, col_right = st.columns([3, 2], gap="large")

    # ── LEFT: Input ──────────────────────────────────────────
    with col_left:

        st.markdown(
            '<div class="section-label">Input Article</div>',
            unsafe_allow_html=True)

        # ── Example loader ──
        examples = {
            "── Select an example ──": "",
            "📰 Reuters: Senate passes infrastructure bill": """
                WASHINGTON (Reuters) - The U.S. Senate passed a
                sweeping bipartisan infrastructure bill on Tuesday,
                allocating 1.2 trillion dollars for roads, bridges,
                broadband internet and other public works projects.
                The legislation passed with 69 votes in favor and
                30 against, securing the bipartisan support President
                Biden had sought. Senate Majority Leader Chuck Schumer
                called the vote a historic moment for the country.
                The bill now moves to the House of Representatives
                where it faces a more uncertain path.
            """,
            "🚫 Deep State Exposed!! Secret Arrests!!": """
                BREAKING EXCLUSIVE: Hillary Clinton was secretly
                arrested by military tribunals last night at her
                Chappaqua estate. Sources close to the Q movement
                confirm Obama and Soros are also being detained.
                The mainstream media is completely blacklisting this
                story. Facebook and Twitter are censoring the truth.
                Share this before they delete it! The deep state is
                finally collapsing. The great awakening has begun.
                WWG1WGA. Pray for our country. Justice is coming soon.
            """,
            "📰 BBC: UK and EU agree post-Brexit trade deal": """
                LONDON (Reuters) - Britain and the European Union
                reached a post-Brexit trade agreement on Thursday,
                averting a chaotic no-deal split that businesses had
                feared would disrupt supply chains and damage economies
                on both sides of the English Channel. The deal covers
                660 billion pounds of annual trade in goods but leaves
                many questions about financial services unresolved.
                Prime Minister Boris Johnson called it a great deal
                for the United Kingdom and said Britain had taken back
                control of its laws and its destiny.
            """,
            "🚫 SHOCKING: George Soros Funds Antifa": """
                SHOCKING TRUTH EXPOSED: George Soros has been secretly
                funding Antifa terrorists with over 50 million dollars
                to destroy America from within. Leaked documents from
                his Open Society Foundation reveal a master plan to
                overthrow the United States government and install a
                globalist puppet regime. Facebook and Google are
                actively hiding this truth. The patriot movement is
                fighting back. Share this with every American before
                the globalists scrub it from the internet forever.
            """,
            "⚠️ Out-of-domain (recipe — will be rejected)": """
                The best chocolate lava cake recipe requires 200 grams
                of dark chocolate, half a cup of butter, two eggs, two
                egg yolks, and a quarter cup of flour. Preheat your
                oven to 220 degrees Celsius. Melt the chocolate and
                butter together in a double boiler, then whisk in the
                eggs and fold in the flour. Pour into greased ramekins
                and bake for exactly 12 minutes for a perfectly molten
                center.
            """,
        }

        selected = st.selectbox(
            "Load an example article",
            options = list(examples.keys()),
            key     = "example_select"
        )

        # Populate textarea with selected example
        default_text = (examples[selected]
                        if selected != "── Select an example ──"
                        else "")

        user_input = st.text_area(
            label       = "Or type / paste your own article here",
            value       = default_text,
            height      = 260,
            placeholder = (
                "Paste a news article headline + body text here...\n\n"
                "For best results, include at least 2-3 sentences. \n"
                "The model is trained on English political and world news."
            ),
            key = "news_input",
        )

        # ── Word count indicator ──
        raw_words = len(user_input.split()) if user_input else 0
        wc_color  = ("#10b981" if raw_words >= 30 else
                     "#f59e0b" if raw_words >= 10 else
                     "#ef4444")
        wc_msg    = ("✅ Good length" if raw_words >= 30 else
                     "⚠️ Short"      if raw_words >= 5  else
                     "❌ Too short"  if raw_words > 0   else "")

        st.markdown(f"""
        <div style='display:flex; justify-content:space-between;
                    color:#475569; font-size:0.8rem;
                    margin-top:-0.5rem; margin-bottom:0.8rem;'>
            <span>{raw_words} words</span>
            <span style='color:{wc_color};'>{wc_msg}</span>
        </div>""", unsafe_allow_html=True)

        # ── Predict button ──
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            predict_btn = st.button(
                "🔍  Analyze Article",
                use_container_width = True,
                type                = "primary",
                disabled            = not bool(user_input.strip()),
            )
        with col_btn2:
            clear_btn = st.button(
                "✕  Clear",
                use_container_width = True,
            )

        if clear_btn:
            st.session_state.last_result = None
            st.rerun()

    # ── RIGHT: Result ─────────────────────────────────────────
    with col_right:

        st.markdown(
            '<div class="section-label">Result</div>',
            unsafe_allow_html=True)

        if predict_btn and user_input.strip():

            # ── Spinner while predicting ──
            with st.spinner("🔍 Analyzing article..."):
                time.sleep(0.4)   # Brief pause for UX
                result = predict_news(user_input)

            st.session_state.last_result = result

            # ── Add to history ──
            st.session_state.history.append({
                'text'      : user_input[:80],
                'status'    : result['status'],
                'label'     : result.get('label'),
                'confidence': result.get('confidence'),
            })

        result = st.session_state.last_result

        if result is None:
            # ── Placeholder ──
            st.markdown("""
            <div class="card" style="text-align:center;
                                     padding:3rem 1.5rem;">
                <div style="font-size:3rem; margin-bottom:0.8rem;
                            opacity:0.3;">🔍</div>
                <div style="color:#475569; font-size:0.95rem;">
                    Enter a news article and click<br>
                    <strong style="color:#6366f1;">
                    Analyze Article</strong>
                    to see the prediction.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif result['status'] == 'success':

            label      = result['label']
            conf       = result['confidence']
            prob_real  = result['prob_real']
            prob_fake  = result['prob_fake']
            certainty  = result['certainty']

            is_real    = (label == 'REAL')
            card_cls   = 'result-real' if is_real else 'result-fake'
            lbl_cls    = 'result-label-real' if is_real else 'result-label-fake'
            bar_color  = '#10b981' if is_real else '#ef4444'
            icon       = '✅' if is_real else '🚫'
            verdict    = 'REAL NEWS' if is_real else 'FAKE NEWS'

            conf_pct   = conf * 100
            bar_width  = conf_pct

            st.markdown(f"""
            <div class="{card_cls}">
                <div class="result-icon pulse">{icon}</div>
                <div class="result-label {lbl_cls}">{verdict}</div>
                <div class="result-confidence">
                    {certainty} Confidence
                </div>

                <!-- Confidence bar -->
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill"
                         style="width:{bar_width:.1f}%;
                                background:{bar_color};
                                opacity:0.85;">
                    </div>
                </div>
                <div style="font-size:0.95rem; font-weight:700;
                            color:{bar_color};
                            margin-bottom:0.5rem;">
                    {conf_pct:.2f}% Confidence
                </div>

                <!-- Probability pills -->
                <div class="prob-row">
                    <span class="prob-pill pill-real">
                        ✅ Real: {prob_real*100:.1f}%
                    </span>
                    <span class="prob-pill pill-fake">
                        🚫 Fake: {prob_fake*100:.1f}%
                    </span>
                </div>
            </div>

            <div class="disclaimer">
                ⚠️ <strong>Disclaimer:</strong> This model is for
                educational purposes only. Always verify news from
                multiple trusted sources.
            </div>
            """, unsafe_allow_html=True)

            # ── Word count stats ──
            st.markdown(f"""
            <div class="card" style="margin-top:0.8rem;
                                     padding:1rem 1.2rem;">
                <div class="section-label">Input Stats</div>
                <div style="display:flex; gap:1.5rem;
                            flex-wrap:wrap; margin-top:0.3rem;">
                    <div>
                        <div style="color:#475569;font-size:0.75rem;">
                            Raw Words</div>
                        <div style="color:#e2e8f0;font-weight:600;">
                            {raw_words}</div>
                    </div>
                    <div>
                        <div style="color:#475569;font-size:0.75rem;">
                            Clean Words</div>
                        <div style="color:#e2e8f0;font-weight:600;">
                            {result['word_count']}</div>
                    </div>
                    <div>
                        <div style="color:#475569;font-size:0.75rem;">
                            Threshold</div>
                        <div style="color:#e2e8f0;font-weight:600;">
                            {THRESHOLD*100:.0f}%</div>
                    </div>
                    <div>
                        <div style="color:#475569;font-size:0.75rem;">
                            Status</div>
                        <div style="color:#10b981;font-weight:600;">
                            Confident ✓</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif result['status'] == 'rejected':

            reason_icons = {
                'empty'            : '📭',
                'too_short'        : '📏',
                'empty_after_clean': '🔇',
                'too_few_words'    : '📝',
                'oov'              : '🌐',
                'low_confidence'   : '🤔',
            }
            icon = reason_icons.get(result['reason'], '⚠️')

            st.markdown(f"""
            <div class="result-rejected">
                <div class="result-icon">{icon}</div>
                <div class="result-label result-label-warn">
                    Cannot Predict
                </div>
                <div style="color:#94a3b8; font-size:0.9rem;
                            margin-top:0.5rem; line-height:1.7;
                            white-space:pre-line;">
                    {result['message']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Helpful tips ──
            with st.expander("💡 Tips for better results"):
                st.markdown("""
                **For accurate predictions, your article should:**
                - Be at least **30+ words** long
                - Be written in **English**
                - Be about **political or world news events**
                - Contain real sentences (not just a headline)

                **This model works best for:**
                - Political news (US, UK, World)
                - Government and policy articles
                - Election and campaign news
                - International relations

                **This model may struggle with:**
                - Sports, entertainment, science news
                - Very recent events after the training cutoff
                - Non-English text
                - Satire or opinion pieces
                """)

        elif result['status'] == 'error':
            st.error(f"❌ An error occurred: {result['message']}")


# ════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="section-label" style="margin-bottom:1rem;">
    Model Internals & Token Analysis
    </div>
    """, unsafe_allow_html=True)

    r = st.session_state.last_result

    if r is None or r['status'] != 'success':
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem;">
            <div style="font-size:2.5rem; opacity:0.3;">📊</div>
            <div style="color:#475569; margin-top:0.8rem;">
                Run a successful prediction first to see analysis.
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        cleaned   = r['cleaned']
        tokens    = cleaned.split()
        n_tokens  = len(tokens)

        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            # ── Probability gauge chart ──
            st.markdown(
                '<div class="section-label">Prediction Breakdown</div>',
                unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#0f0f1a')

            categories = ['REAL', 'FAKE']
            values     = [r['prob_real']*100, r['prob_fake']*100]
            colors_bar = ['#10b981', '#ef4444']
            bars       = ax.barh(categories, values,
                                 color=colors_bar,
                                 height=0.45,
                                 edgecolor='none')
            for bar, val in zip(bars, values):
                ax.text(val + 0.8, bar.get_y()+bar.get_height()/2,
                        f'{val:.1f}%', va='center',
                        color='#e2e8f0', fontsize=11,
                        fontweight='bold')

            ax.set_xlim(0, 115)
            ax.set_xlabel("Probability (%)",
                          color='#475569', fontsize=9)
            ax.tick_params(colors='#94a3b8', labelsize=10)
            ax.spines[:].set_visible(False)
            ax.axvline(50, color='#475569',
                       linestyle='--', linewidth=0.8, alpha=0.5)
            ax.text(50.5, -0.65, '50%',
                    color='#475569', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Confidence gauge ──
            st.markdown(
                '<div class="section-label" style="margin-top:1rem;">                Confidence Gauge</div>',
                unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(
                figsize=(5, 2.6),
                subplot_kw=dict(polar=False))
            fig2.patch.set_facecolor('#0f0f1a')
            ax2.set_facecolor('#0f0f1a')

            conf    = r['confidence'] * 100
            is_real = r['label'] == 'REAL'
            c       = '#10b981' if is_real else '#ef4444'

            ax2.barh([0], [100], color='#1e293b', height=0.4)
            ax2.barh([0], [conf], color=c,
                     height=0.4, alpha=0.85)

            ax2.text(conf/2, 0, f'{conf:.1f}%',
                     ha='center', va='center',
                     color='white', fontsize=12,
                     fontweight='bold')
            ax2.axvline(60, color='#f59e0b',
                        linestyle='--',
                        linewidth=1.2, alpha=0.7)
            ax2.text(60.5, 0.35, 'Min\n60%',
                     color='#f59e0b', fontsize=7)

            ax2.set_xlim(0, 110)
            ax2.set_yticks([])
            ax2.set_xlabel("Confidence (%)",
                           color='#475569', fontsize=9)
            ax2.tick_params(colors='#94a3b8', labelsize=9)
            ax2.spines[:].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        with col_b:
            # ── Token analysis ──
            st.markdown(
                '<div class="section-label">Token Analysis</div>',
                unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card" style="padding:1.2rem;">
                <div style="color:#475569;font-size:0.8rem;">
                    Cleaned Tokens</div>
                <div style="color:#e2e8f0;font-size:1.6rem;
                            font-weight:700;">{n_tokens}</div>
            </div>
            """, unsafe_allow_html=True)

            # TF-IDF top features for this article
            try:
                vec       = vectorizer.transform([cleaned])
                feat_names= vectorizer.get_feature_names_out()
                nz_idx    = vec.nonzero()[1]
                scores    = [(feat_names[i],
                              float(vec[0, i]))
                             for i in nz_idx]
                scores.sort(key=lambda x: x[1], reverse=True)
                top_feats = scores[:12]

                if top_feats:
                    df_feats = pd.DataFrame(
                        top_feats,
                        columns=['Token', 'TF-IDF Score'])

                    fig3, ax3 = plt.subplots(figsize=(5, 4.5))
                    fig3.patch.set_facecolor('#0f0f1a')
                    ax3.set_facecolor('#0f0f1a')

                    clr = '#10b981' if is_real else '#ef4444'
                    ax3.barh(df_feats['Token'][::-1],
                             df_feats['TF-IDF Score'][::-1],
                             color=clr, alpha=0.75,
                             edgecolor='none')
                    ax3.set_title(
                        "Top TF-IDF Tokens (this article)",
                        color='#94a3b8', fontsize=9,
                        pad=8)
                    ax3.tick_params(colors='#94a3b8',
                                    labelsize=8)
                    ax3.spines[:].set_visible(False)
                    ax3.set_xlabel("TF-IDF Score",
                                   color='#475569',
                                   fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close()

            except Exception:
                st.info("Token chart unavailable.")

        # ── Cleaned text preview ──
        st.markdown(
            '<div class="section-label" style="margin-top:1rem;">            Preprocessed Text (sent to model)</div>',
            unsafe_allow_html=True)
        st.code(cleaned[:600] + ("..." if len(cleaned) > 600
                                 else ""),
                language='text')


# ════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════
with tab3:

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="card">
            <div class="section-label">About This Project</div>
            <div style="color:#94a3b8; line-height:1.85;
                        font-size:0.9rem; margin-top:0.5rem;">
                <p>
                This Fake News Detection system was built as a
                complete end-to-end Data Science project using
                <strong style="color:#a5b4fc;">
                NLP + Machine Learning</strong>.
                </p>
                <p>
                It uses a <strong style="color:#a5b4fc;">
                Logistic Regression</strong> model trained on
                <strong style="color:#a5b4fc;">TF-IDF features</strong>
                extracted from ~45,000 news articles from the
                Kaggle Fake/Real News dataset.
                </p>
                <p>
                The model achieves <strong style="color:#10b981;">
                ~98.9% accuracy</strong> on the held-out test set.
                </p>
            </div>
        </div>

        <div class="card">
            <div class="section-label">Tech Stack</div>
            <div style="color:#94a3b8; font-size:0.88rem;
                        line-height:2; margin-top:0.5rem;">
                🐍 Python 3.x<br>
                📊 Pandas + NumPy<br>
                🔤 NLTK (tokenization, lemmatization)<br>
                ⚙️ Scikit-learn (TF-IDF + Logistic Regression)<br>
                📈 Matplotlib + Seaborn<br>
                💾 Joblib (model persistence)<br>
                🌐 Streamlit (frontend)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="section-label">ML Pipeline</div>
            <div style="color:#94a3b8; font-size:0.85rem;
                        line-height:2; margin-top:0.5rem;">
                1️⃣  Load True.csv + Fake.csv (Kaggle)<br>
                2️⃣  Label: Real=1, Fake=0<br>
                3️⃣  Merge + clean + deduplicate<br>
                4️⃣  NLP: lowercase → remove URLs/HTML/punct<br>
                5️⃣  Tokenize → remove stopwords → lemmatize<br>
                6️⃣  TF-IDF vectorization (50K features, 1-2 grams)<br>
                7️⃣  Train 4 models + compare<br>
                8️⃣  Evaluate: accuracy, F1, ROC-AUC<br>
                9️⃣  Save best model (LR) as model.pkl<br>
                🔟  Deploy on Streamlit
            </div>
        </div>

        <div class="card">
            <div class="section-label">Limitations & Ethics</div>
            <div style="color:#94a3b8; font-size:0.85rem;
                        line-height:1.85; margin-top:0.5rem;">
                ⚠️ Trained on 2016–2017 era news data.<br>
                ⚠️ May not generalise to all news domains.<br>
                ⚠️ Best for political + world news in English.<br>
                ⚠️ Not suitable for satire or opinion pieces.<br>
                ⚠️ Always verify from multiple trusted sources.<br>
                ⚠️ This tool is for educational use only.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Dataset info ──
    st.markdown("""
    <div class="card">
        <div class="section-label">Dataset</div>
        <div style="color:#94a3b8; font-size:0.88rem;
                    line-height:1.85; margin-top:0.5rem;">
            <strong style="color:#e2e8f0;">Source:</strong>
            Kaggle — Fake and Real News Dataset
            (by Clément Bisaillon)<br>
            <strong style="color:#e2e8f0;">Files:</strong>
            True.csv (21,417 articles) +
            Fake.csv (23,481 articles)<br>
            <strong style="color:#e2e8f0;">Total:</strong>
            44,898 articles after merging<br>
            <strong style="color:#e2e8f0;">Columns:</strong>
            title, text, subject, date<br>
            <strong style="color:#e2e8f0;">Split:</strong>
            80% train / 20% test (stratified)<br>
            <strong style="color:#e2e8f0;">Period:</strong>
            Approximately 2015–2018
        </div>
    </div>
    """, unsafe_allow_html=True)
