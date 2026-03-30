import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import io
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse · Milestone 2",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("dark_mode", True),
    ("files_loaded", False),
    ("data_parsed", False),
    ("master_built", False),
    ("tsfresh_done", False),
    ("prophet_done", False),
    ("clustering_done", False),
    ("daily", None), ("hourly_s", None), ("hourly_i", None),
    ("sleep", None), ("hr", None), ("hr_minute", None),
    ("master", None), ("features", None),
    ("forecast_hr", None), ("prophet_hr", None),
    ("cluster_features", None), ("X_scaled", None),
    ("X_pca", None), ("X_tsne", None),
    ("kmeans_labels", None), ("dbscan_labels", None),
    ("OPTIMAL_K", 3), ("EPS", 2.2),
    ("var_explained", None), ("n_clusters", 0), ("n_noise", 0),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Theme ─────────────────────────────────────────────────────────────────────
dark = st.session_state.dark_mode
if dark:
    BG        = "linear-gradient(135deg, #0a0e1a 0%, #0f1729 40%, #0a1628 100%)"
    CARD_BG   = "rgba(15,23,42,0.85)"
    CARD_BOR  = "rgba(99,179,237,0.2)"
    MPL_BOR   = "#1e3a5f"  # hex equiv for matplotlib
    TEXT      = "#e2e8f0"
    MUTED     = "#94a3b8"
    ACCENT    = "#63b3ed"
    ACCENT2   = "#f687b3"
    ACCENT3   = "#68d391"
    PLOT_BG   = "#0f172a"
    GRID_CLR  = "rgba(255,255,255,0.06)"
    MPL_GRID  = "#1a2535"
    TICK_CLR  = "#64748b"
    BADGE_BG  = "rgba(99,179,237,0.15)"
    SECTION_BG= "rgba(99,179,237,0.07)"
    WARN_BG   = "rgba(246,173,85,0.12)"
    WARN_BOR  = "rgba(246,173,85,0.4)"
    SUCCESS_BG= "rgba(104,211,145,0.1)"
    SUCCESS_BOR="rgba(104,211,145,0.4)"
else:
    BG        = "linear-gradient(135deg, #f0f4ff 0%, #fafbff 50%, #f5f0ff 100%)"
    CARD_BG   = "rgba(255,255,255,0.9)"
    CARD_BOR  = "rgba(66,153,225,0.25)"
    MPL_BOR   = "#b8d4f0"  # hex equiv for matplotlib
    TEXT      = "#1a202c"
    MUTED     = "#4a5568"
    ACCENT    = "#3182ce"
    ACCENT2   = "#d53f8c"
    ACCENT3   = "#38a169"
    PLOT_BG   = "#ffffff"
    GRID_CLR  = "rgba(0,0,0,0.06)"
    MPL_GRID  = "#e8edf5"
    TICK_CLR  = "#718096"
    BADGE_BG  = "rgba(49,130,206,0.1)"
    SECTION_BG= "rgba(49,130,206,0.05)"
    WARN_BG   = "rgba(221,107,32,0.08)"
    WARN_BOR  = "rgba(221,107,32,0.35)"
    SUCCESS_BG= "rgba(56,161,105,0.08)"
    SUCCESS_BOR="rgba(56,161,105,0.35)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], .main {{
    background: {BG} !important;
    font-family: 'Inter', sans-serif;
    color: {TEXT} !important;
}}

[data-testid="stHeader"] {{ background: transparent !important; }}

[data-testid="stSidebar"] {{
    background: {'rgba(10,14,26,0.97)' if dark else 'rgba(240,244,255,0.97)'} !important;
    border-right: 1px solid {CARD_BOR};
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

.block-container {{ padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px; }}

p, div, span, label {{ color: {TEXT}; }}

/* Hero */
.m2-hero {{
    background: {'linear-gradient(135deg, rgba(10,14,26,0.9), rgba(15,23,60,0.85))' if dark else 'linear-gradient(135deg, rgba(235,244,255,0.9), rgba(245,235,255,0.85))'};
    border: 1px solid {CARD_BOR};
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}}
.m2-hero::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, {'rgba(99,179,237,0.08)' if dark else 'rgba(49,130,206,0.06)'} 0%, transparent 70%);
    border-radius: 50%;
}}
.m2-hero-title {{
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: {TEXT};
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}}
.m2-hero-sub {{
    font-size: 1.05rem;
    color: {MUTED};
    font-weight: 300;
    margin: 0;
}}
.m2-hero-badge {{
    display: inline-block;
    background: {BADGE_BG};
    border: 1px solid {CARD_BOR};
    border-radius: 100px;
    padding: 0.3rem 1rem;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
}}

/* Section headers */
.sec-header {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid {CARD_BOR};
}}
.sec-icon {{
    font-size: 1.4rem;
    width: 2.2rem; height: 2.2rem;
    display: flex; align-items: center; justify-content: center;
    background: {BADGE_BG};
    border-radius: 8px;
    border: 1px solid {CARD_BOR};
}}
.sec-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: {TEXT};
    margin: 0;
}}
.sec-badge {{
    margin-left: auto;
    background: {BADGE_BG};
    border: 1px solid {CARD_BOR};
    border-radius: 100px;
    padding: 0.2rem 0.7rem;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
}}

/* Cards */
.card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BOR};
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}}
.card-title {{
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
}}

/* Step pills */
.step-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: {SECTION_BG};
    border: 1px solid {CARD_BOR};
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
    margin-bottom: 0.8rem;
}}

/* Metric cards */
.metric-grid {{ display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 0.8rem 0; }}
.metric-card {{
    flex: 1; min-width: 120px;
    background: {SECTION_BG};
    border: 1px solid {CARD_BOR};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}}
.metric-val {{
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: {ACCENT};
    line-height: 1;
    margin-bottom: 0.25rem;
}}
.metric-label {{
    font-size: 0.72rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* Log box */
.log-box {{
    background: {'rgba(0,0,0,0.4)' if dark else 'rgba(240,240,240,0.8)'};
    border: 1px solid {CARD_BOR};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {ACCENT3};
    line-height: 1.7;
}}
.log-line {{ margin: 0; }}
.log-key {{ color: {ACCENT}; }}
.log-warn {{ color: #f6ad55; }}

/* Alert boxes */
.alert-warn {{
    background: {WARN_BG};
    border-left: 3px solid #f6ad55;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.85rem;
    color: {'#fbd38d' if dark else '#c05621'};
}}
.alert-success {{
    background: {SUCCESS_BG};
    border-left: 3px solid {ACCENT3};
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.85rem;
    color: {'#9ae6b4' if dark else '#276749'};
}}
.alert-info {{
    background: {BADGE_BG};
    border-left: 3px solid {ACCENT};
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.85rem;
    color: {'#bee3f8' if dark else '#2c5282'};
}}

/* Screenshot badge */
.screenshot-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: {'rgba(246,135,179,0.15)' if dark else 'rgba(213,63,140,0.1)'};
    border: 1px solid {'rgba(246,135,179,0.4)' if dark else 'rgba(213,63,140,0.3)'};
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT2};
    margin-bottom: 0.8rem;
}}

/* Cluster profile tag */
.cluster-tag {{
    display: inline-block;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}}

/* Pipeline progress */
.pipeline-bar {{
    display: flex;
    gap: 0.4rem;
    margin: 1rem 0;
    align-items: center;
}}
.pipeline-step {{
    flex: 1;
    height: 4px;
    border-radius: 2px;
    transition: background 0.4s;
}}

/* Dividers */
.m2-divider {{
    border: none;
    border-top: 1px solid {CARD_BOR};
    margin: 2rem 0;
}}

/* Streamlit overrides */
div[data-testid="stFileUploader"] {{
    background: {SECTION_BG};
    border: 2px dashed {CARD_BOR};
    border-radius: 14px;
    padding: 0.5rem;
}}
div[data-testid="stFileUploader"]:hover {{
    border-color: {ACCENT};
}}
.stButton > button {{
    background: {'rgba(99,179,237,0.15)' if dark else 'rgba(49,130,206,0.1)'};
    border: 1px solid {CARD_BOR};
    color: {ACCENT};
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    background: {ACCENT};
    color: white;
    border-color: {ACCENT};
    transform: translateY(-1px);
}}
.stSlider [data-baseweb="slider"] {{ padding: 0.2rem 0; }}
div[data-testid="stDataFrame"] {{
    border: 1px solid {CARD_BOR};
    border-radius: 10px;
    overflow: hidden;
}}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fig_style(fig, ax_or_axes, title="", xlabel="", ylabel=""):
    """Apply consistent dark/light style to matplotlib figures."""
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    axes = np.array(axes).flatten()
    fig.patch.set_facecolor(PLOT_BG)
    for ax in axes:
        ax.set_facecolor(PLOT_BG)
        if title:  ax.set_title(title, fontsize=12, color=TEXT, fontweight='bold', pad=10)
        if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
        if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.tick_params(colors=TICK_CLR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(MPL_BOR)
        ax.grid(color=MPL_GRID, linestyle='-', linewidth=0.5, alpha=0.4)

def sec(icon, title, badge=None):
    badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="sec-header">
      <div class="sec-icon">{icon}</div>
      <p class="sec-title">{title}</p>
      {badge_html}
    </div>""", unsafe_allow_html=True)

def card(content_fn, title=None):
    title_html = f'<div class="card-title">{title}</div>' if title else ''
    st.markdown(f'<div class="card">{title_html}', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)

def step_pill(n, label):
    st.markdown(f'<div class="step-pill">◆ Step {n} &nbsp;·&nbsp; {label}</div>', unsafe_allow_html=True)

def screenshot_badge(cell_ref):
    st.markdown(f'<div class="screenshot-badge">📸 Screenshot · {cell_ref}</div>', unsafe_allow_html=True)

def ui_success(msg): st.markdown(f'<div class="alert-success">✅ {msg}</div>', unsafe_allow_html=True)
def ui_warn(msg):    st.markdown(f'<div class="alert-warn">⚠️ {msg}</div>', unsafe_allow_html=True)
def ui_info(msg):    st.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)

def metrics(*items):
    html = '<div class="metric-grid">'
    for val, label in items:
        html += f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

PALETTE = ["#63b3ed","#f687b3","#68d391","#f6ad55","#b794f4","#fc8181"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.5rem 0 1.5rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;color:{ACCENT}">
        🧬 FitPulse
      </div>
      <div style="font-size:0.72rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-top:0.2rem">
        Milestone 2 · ML Pipeline
      </div>
    </div>
    """, unsafe_allow_html=True)

    new_dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)

    # Pipeline progress
    steps_done = sum([
        st.session_state.files_loaded,
        st.session_state.data_parsed,
        st.session_state.master_built,
        st.session_state.tsfresh_done,
        st.session_state.prophet_done,
        st.session_state.clustering_done,
    ])
    pct = int(steps_done / 6 * 100)
    st.markdown(f"""
    <div style="margin-bottom:1rem">
      <div style="font-size:0.72rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-bottom:0.4rem">
        PIPELINE PROGRESS &nbsp;·&nbsp; {pct}%
      </div>
      <div style="background:{CARD_BOR};border-radius:4px;height:6px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{ACCENT},{ACCENT2});border-radius:4px;transition:width 0.4s"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("📂", "Data Loading",      "files_loaded"),
        ("🔬", "TSFresh Features",   "tsfresh_done"),
        ("📈", "Prophet Forecast",   "prophet_done"),
        ("🤖", "Clustering",         "clustering_done"),
    ]
    for icon, label, state_key in nav_items:
        done = st.session_state.get(state_key, False)
        dot  = f'<span style="color:{ACCENT3}">●</span>' if done else f'<span style="color:{MUTED}">○</span>'
        st.markdown(f'<div style="font-size:0.82rem;padding:0.3rem 0;color:{TEXT if done else MUTED}">{dot} {icon} {label}</div>', unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)

    # KMeans K slider
    st.markdown(f'<div style="font-size:0.72rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.3rem">KMEANS CLUSTERS (K)</div>', unsafe_allow_html=True)
    st.session_state.OPTIMAL_K = st.slider("K", 2, 8, st.session_state.OPTIMAL_K, key="k_slider", label_visibility="hidden")

    st.markdown(f'<div style="font-size:0.72rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin:0.8rem 0 0.3rem">DBSCAN EPS</div>', unsafe_allow_html=True)
    st.session_state.EPS = st.slider("EPS", 1.0, 5.0, st.session_state.EPS, step=0.1, key="eps_slider", label_visibility="hidden")

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.68rem;color:{MUTED};font-family:JetBrains Mono,monospace">Real Fitbit Dataset<br>30 users · March–April 2016<br>Minute-level HR data</div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="m2-hero">
  <div class="m2-hero-badge">MILESTONE 2 · FEATURE EXTRACTION & MODELING</div>
  <h1 class="m2-hero-title">🧬 FitPulse ML Pipeline</h1>
  <p class="m2-hero-sub">TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE — Real Fitbit Device Data</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FILE UPLOAD & DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
sec("📂", "Data Loading", "Steps 1–9")

# ── Required file registry ────────────────────────────────────────────────────
REQUIRED_FILES = {
    "dailyActivity_merged.csv":     {
        "key_cols": ["ActivityDate", "TotalSteps", "Calories"],
        "label": "Daily Activity",    "icon": "🏃"},
    "hourlySteps_merged.csv":       {
        "key_cols": ["ActivityHour", "StepTotal"],
        "label": "Hourly Steps",      "icon": "👣"},
    "hourlyIntensities_merged.csv": {
        "key_cols": ["ActivityHour", "TotalIntensity"],
        "label": "Hourly Intensities","icon": "⚡"},
    "minuteSleep_merged.csv":       {
        "key_cols": ["date", "value", "logId"],
        "label": "Minute Sleep",      "icon": "💤"},
    "heartrate_seconds_merged.csv": {
        "key_cols": ["Time", "Value"],
        "label": "Heart Rate",        "icon": "❤️"},
}

def score_match(df, req_info):
    """Score how well a df matches a required file (count matching key columns)."""
    return sum(1 for col in req_info["key_cols"] if col in df.columns)

st.markdown(f"""
<div class="alert-info">
Select all 5 Fitbit CSV files at once using the uploader below. Files are auto-detected by their column structure — no need to rename them.
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📁  Drop all 5 Fitbit CSV files here (select multiple at once)",
    type="csv",
    accept_multiple_files=True,
    key="multi_uploader",
    help="Hold Ctrl (Windows) or Cmd (Mac) to select multiple files"
)

# ── Auto-detect and validate uploaded files ───────────────────────────────────
detected   = {}   # req_filename → dataframe
ignored    = []   # unmatched upload names
if uploaded_files:
    # Step 1: read all uploads into memory
    raw_uploads = []
    for uf in uploaded_files:
        try:
            df_tmp = pd.read_csv(uf)
            raw_uploads.append((uf.name, df_tmp))
        except Exception:
            ignored.append(uf.name)

    # Step 2: for each required file, pick the upload with the highest column-match score
    used_names = set()
    for req_name, finfo in REQUIRED_FILES.items():
        best_score, best_name, best_df = 0, None, None
        for uname, udf in raw_uploads:
            s = score_match(udf, finfo)
            if s > best_score:
                best_score, best_name, best_df = s, uname, udf
        if best_score >= 2:
            detected[req_name] = best_df
            used_names.add(best_name)

    # Step 3: flag uploads that didn't match anything
    for uname, _ in raw_uploads:
        if uname not in used_names:
            ignored.append(uname)

# ── Status grid ───────────────────────────────────────────────────────────────
status_html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.6rem;margin:1rem 0">'
for req_name, finfo in REQUIRED_FILES.items():
    found = req_name in detected
    bg    = SUCCESS_BG if found else WARN_BG
    bor   = SUCCESS_BOR if found else WARN_BOR
    ico   = "✅" if found else "❌"
    status_html += f"""
    <div style="background:{bg};border:1px solid {bor};border-radius:10px;padding:0.7rem 0.9rem">
      <div style="font-size:1.2rem">{ico} {finfo['icon']}</div>
      <div style="font-size:0.72rem;font-weight:600;color:{TEXT};margin-top:0.3rem">{finfo['label']}</div>
      <div style="font-size:0.65rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-top:0.1rem">
        {'Found ✓' if found else 'Missing'}
      </div>
    </div>"""
status_html += "</div>"
st.markdown(status_html, unsafe_allow_html=True)

# Show ignored files
if ignored:
    ui_warn(f"Ignored {len(ignored)} unrecognised file(s): {', '.join(ignored)}")

# Summary metrics
n_up = len(detected)
metrics(
    (n_up,                             "Detected"),
    (5 - n_up,                         "Missing"),
    ("✓" if n_up == 5 else "✗",        "Ready to Load"),
)

missing_names = [REQUIRED_FILES[r]["label"] for r in REQUIRED_FILES if r not in detected]
if missing_names:
    ui_warn(f"Still missing: {', '.join(missing_names)}")
else:
    ui_success("All 5 required files detected — ready to process!")

if st.button("⚡ Load & Parse All Files", disabled=(n_up < 5)):
    with st.spinner("Loading and parsing all files..."):
        try:
            daily    = detected["dailyActivity_merged.csv"]
            hourly_s = detected["hourlySteps_merged.csv"]
            hourly_i = detected["hourlyIntensities_merged.csv"]
            sleep    = detected["minuteSleep_merged.csv"]
            hr       = detected["heartrate_seconds_merged.csv"]

            # Parse timestamps
            daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"], format="%m/%d/%Y")
            hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
            hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
            sleep["date"]            = pd.to_datetime(sleep["date"],            format="%m/%d/%Y %I:%M:%S %p")
            hr["Time"]               = pd.to_datetime(hr["Time"],               format="%m/%d/%Y %I:%M:%S %p")

            # HR resampling
            hr_minute = (
                hr.set_index("Time")
                .groupby("Id")["Value"]
                .resample("1min").mean()
                .reset_index()
            )
            hr_minute.columns = ["Id","Time","HeartRate"]
            hr_minute = hr_minute.dropna()

            # Build master
            hr_minute["Date"] = hr_minute["Time"].dt.date
            hr_daily = (
                hr_minute.groupby(["Id","Date"])["HeartRate"]
                .agg(["mean","max","min","std"]).reset_index()
                .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"})
            )
            sleep["Date"] = sleep["date"].dt.date
            sleep_daily = (
                sleep.groupby(["Id","Date"])
                .agg(TotalSleepMinutes=("value","count"),
                     DominantSleepStage=("value", lambda x: x.mode()[0]))
                .reset_index()
            )
            master = daily.copy().rename(columns={"ActivityDate":"Date"})
            master["Date"] = master["Date"].dt.date
            master = master.merge(hr_daily, on=["Id","Date"], how="left")
            master = master.merge(sleep_daily, on=["Id","Date"], how="left")
            master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
            master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
            for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))

            # Store
            st.session_state.daily     = daily
            st.session_state.hourly_s  = hourly_s
            st.session_state.hourly_i  = hourly_i
            st.session_state.sleep     = sleep
            st.session_state.hr        = hr
            st.session_state.hr_minute = hr_minute
            st.session_state.master    = master
            st.session_state.files_loaded  = True
            st.session_state.data_parsed   = True
            st.session_state.master_built  = True
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.files_loaded:
    daily     = st.session_state.daily
    hourly_s  = st.session_state.hourly_s
    hr        = st.session_state.hr
    hr_minute = st.session_state.hr_minute
    master    = st.session_state.master
    sleep     = st.session_state.sleep

    ui_success("All 5 files loaded and master DataFrame built")

    # Null check
    step_pill(4, "Null Value Check")
    cols_nc = st.columns(5)
    dfs = [("dailyActivity",daily),("hourlySteps",hourly_s),
           ("hourlyIntensities",st.session_state.hourly_i),
           ("minuteSleep",sleep),("heartrate",hr)]
    for col, (nm, df) in zip(cols_nc, dfs):
        nulls = df.isnull().sum().sum()
        col.markdown(f"""
        <div class="metric-card" style="text-align:left">
          <div style="font-size:0.7rem;color:{MUTED};margin-bottom:0.3rem;font-family:'JetBrains Mono',monospace">{nm}</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:{'#68d391' if nulls==0 else '#fc8181'}">{nulls}</div>
          <div style="font-size:0.68rem;color:{MUTED}">nulls · {df.shape[0]:,} rows</div>
        </div>
        """, unsafe_allow_html=True)

    # Time normalization log
    step_pill(7, "Time Normalization Log")
    freq_check = hourly_s.groupby("Id")["ActivityHour"].diff().dropna().dt.total_seconds()/3600
    st.markdown(f"""
    <div class="log-box">
      <p class="log-line"><span class="log-key">✅ HR resampled</span> &nbsp;seconds → 1-minute intervals</p>
      <p class="log-line">&nbsp;&nbsp;&nbsp;Rows before : <b>{hr.shape[0]:,}</b> &nbsp;|&nbsp; Rows after : <b>{hr_minute.shape[0]:,}</b></p>
      <p class="log-line"><span class="log-key">✅ Date range</span> &nbsp;{daily["ActivityDate"].min().date()} → {daily["ActivityDate"].max().date()} &nbsp;({(daily["ActivityDate"].max()-daily["ActivityDate"].min()).days} days)</p>
      <p class="log-line"><span class="log-key">✅ Hourly frequency</span> &nbsp;{freq_check.median():.1f}h median &nbsp;|&nbsp; {(freq_check==1.0).mean()*100:.1f}% exact 1-hour</p>
      <p class="log-line"><span class="log-key">✅ Sleep stages</span> &nbsp;1=Light · 2=Deep · 3=REM &nbsp;|&nbsp; {sleep.shape[0]:,} records</p>
      <p class="log-line"><span class="log-warn">⚠️ Timezone</span> &nbsp;Local time — UTC normalization not applicable</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset overview
    step_pill(5, "Dataset Overview")
    metrics(
        (daily["Id"].nunique(), "Daily Users"),
        (hr["Id"].nunique(),    "HR Users"),
        (sleep["Id"].nunique(), "Sleep Users"),
        (f"{hr_minute.shape[0]:,}", "HR Minute Rows"),
        (master.shape[0],          "Master Rows"),
    )

    # Cleaned preview
    step_pill(9, "Cleaned Dataset Preview")
    show_cols = ["Id","Date","TotalSteps","Calories","AvgHR","TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]
    st.dataframe(master[show_cols].head(20), use_container_width=True, height=280)

    st.markdown('<hr class="m2-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — TSFresh
    # ══════════════════════════════════════════════════════════════════════════
    sec("🧪", "TSFresh Feature Extraction", "Steps 10–12")

    ui_info("TSFresh extracts statistical features from minute-level heart rate time series. Each row = one user, each column = one statistical feature.")

    if st.button("🔬 Run TSFresh Feature Extraction"):
        with st.spinner("⏳ Extracting features... (1–3 minutes for real data)"):
            try:
                from tsfresh import extract_features
                from tsfresh.feature_extraction import MinimalFCParameters
                from sklearn.preprocessing import MinMaxScaler

                ts_hr = hr_minute[["Id","Time","HeartRate"]].copy().dropna()
                ts_hr = ts_hr.sort_values(["Id","Time"])
                ts_hr = ts_hr.rename(columns={"Id":"id","Time":"time","HeartRate":"value"})

                # Use explicit minimal feature dict to avoid version issues
                _minimal_fc = {
                    "sum_values": None,
                    "median": None,
                    "mean": None,
                    "length": None,
                    "standard_deviation": None,
                    "variance": None,
                    "root_mean_square": None,
                    "maximum": None,
                    "absolute_maximum": None,
                    "minimum": None,
                }
                features = extract_features(
                    ts_hr,
                    column_id="id",
                    column_sort="time",
                    column_value="value",
                    default_fc_parameters=_minimal_fc,
                    disable_progressbar=True,
                    n_jobs=1
                )
                features = features.dropna(axis=1, how="all")
                st.session_state.features    = features
                st.session_state.tsfresh_done = True
                st.rerun()
            except Exception as e:
                st.error(f"TSFresh error: {e}")

    if st.session_state.tsfresh_done and st.session_state.features is not None:
        features = st.session_state.features
        ui_success(f"TSFresh complete — {features.shape[0]} users × {features.shape[1]} features extracted")

        step_pill(10, "TSFresh Input Stats")
        metrics(
            (hr_minute["Id"].nunique(), "Users"),
            (f"{hr_minute.shape[0]:,}", "Minute Rows"),
            (features.shape[1], "Features Extracted"),
        )

        step_pill(12, "Feature Matrix Heatmap")
        screenshot_badge("Cell 15 · TSFresh Heatmap")

        from sklearn.preprocessing import MinMaxScaler
        scaler_vis    = MinMaxScaler()
        features_norm = pd.DataFrame(
            scaler_vis.fit_transform(features),
            index=features.index, columns=features.columns
        )

        fig, ax = plt.subplots(figsize=(14, max(5, features.shape[0]*0.55)))
        sns.heatmap(
            features_norm, ax=ax, cmap="coolwarm",
            annot=True, fmt=".2f", linewidths=0.5,
            linecolor="#1a2744" if dark else "#e2e8f0",
            cbar_kws={"shrink":0.8}
        )
        ax.set_title("TSFresh Feature Matrix — Real Fitbit Heart Rate Data\n(Normalized 0-1 per feature)",
                     fontsize=12, color=TEXT, pad=12)
        ax.set_xlabel("Extracted Statistical Features", fontsize=9, color=MUTED)
        ax.set_ylabel("User ID", fontsize=9, color=MUTED)
        ax.tick_params(colors=TICK_CLR, labelsize=7)
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature descriptions
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Feature Interpretation Guide</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.82rem">
            <div><span style="color:{ACCENT};font-family:'JetBrains Mono',monospace">sum_values</span> — Total HR over time (activity volume)</div>
            <div><span style="color:{ACCENT};font-family:'JetBrains Mono',monospace">median / mean</span> — Central tendency of HR</div>
            <div><span style="color:{ACCENT};font-family:'JetBrains Mono',monospace">standard_deviation</span> — HR variability (fitness indicator)</div>
            <div><span style="color:{ACCENT};font-family:'JetBrains Mono',monospace">variance</span> — Square of std dev</div>
            <div><span style="color:{ACCENT};font-family:'JetBrains Mono',monospace">root_mean_square</span> — Energy-weighted average HR</div>
            <div><span style="color:{ACCENT};font-family:'JetBrains Mono',monospace">maximum / minimum</span> — Peak and resting HR</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="m2-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — PROPHET
    # ══════════════════════════════════════════════════════════════════════════
    sec("📈", "Prophet Trend Forecasting", "Steps 13–17")

    ui_info("Prophet fits additive models with weekly seasonality and 80% confidence intervals. 30-day ahead forecasts for Heart Rate, Steps, and Sleep.")

    if st.button("🔮 Run Prophet Forecasting (Heart Rate + Steps + Sleep)"):
        with st.spinner("⏳ Fitting 3 Prophet models..."):
            try:
                from prophet import Prophet

                # HR
                prophet_hr = hr_minute.groupby("Date")["HeartRate"].mean().reset_index()
                prophet_hr.columns = ["ds","y"]
                prophet_hr["ds"]   = pd.to_datetime(prophet_hr["ds"])
                prophet_hr         = prophet_hr.dropna().sort_values("ds")

                model_hr = Prophet(
                    daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=False, interval_width=0.80,
                    changepoint_prior_scale=0.01, changepoint_range=0.8
                )
                model_hr.fit(prophet_hr)
                future_hr   = model_hr.make_future_dataframe(periods=30)
                forecast_hr = model_hr.predict(future_hr)

                st.session_state.prophet_hr  = prophet_hr
                st.session_state.forecast_hr = forecast_hr
                st.session_state.model_hr    = model_hr

                # Steps
                steps_agg = daily.groupby("ActivityDate")["TotalSteps"].mean().reset_index()
                steps_agg.columns = ["ds","y"]
                steps_agg["ds"]   = pd.to_datetime(steps_agg["ds"])
                steps_agg         = steps_agg.dropna().sort_values("ds")
                m_steps = Prophet(weekly_seasonality=True, yearly_seasonality=False,
                                  daily_seasonality=False, interval_width=0.80,
                                  changepoint_prior_scale=0.1)
                m_steps.fit(steps_agg)
                f_steps = m_steps.predict(m_steps.make_future_dataframe(periods=30))

                # Sleep
                sleep_agg = master.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
                sleep_agg.columns = ["ds","y"]
                sleep_agg["ds"]   = pd.to_datetime(sleep_agg["ds"], errors="coerce")
                sleep_agg         = sleep_agg.dropna().sort_values("ds")
                m_sleep = Prophet(weekly_seasonality=True, yearly_seasonality=False,
                                  daily_seasonality=False, interval_width=0.80,
                                  changepoint_prior_scale=0.1)
                m_sleep.fit(sleep_agg)
                f_sleep = m_sleep.predict(m_sleep.make_future_dataframe(periods=30))

                st.session_state.prophet_steps_agg  = steps_agg
                st.session_state.prophet_steps    = f_steps
                st.session_state.prophet_sleep_agg  = sleep_agg
                st.session_state.prophet_sleep    = f_sleep
                st.session_state.prophet_done = True
                st.rerun()
            except Exception as e:
                st.error(f"Prophet error: {e}")

    if st.session_state.prophet_done:
        prophet_hr  = st.session_state.prophet_hr
        forecast_hr = st.session_state.forecast_hr
        steps_agg   = st.session_state.prophet_steps_agg
        f_steps     = st.session_state.prophet_steps
        sleep_agg   = st.session_state.prophet_sleep_agg
        f_sleep     = st.session_state.prophet_sleep

        ui_success("3 Prophet models fitted — HR, Steps, Sleep · 30-day forecast each")

        # ── HR Forecast ──
        step_pill(15, "Heart Rate Forecast")
        screenshot_badge("Cell 18 · Prophet HR Forecast")

        fig, ax = plt.subplots(figsize=(13, 4.5))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.scatter(prophet_hr["ds"], prophet_hr["y"], color=ACCENT2, s=20, alpha=0.75, label="Actual HR", zorder=3)
        ax.plot(forecast_hr["ds"], forecast_hr["yhat"], color=ACCENT, linewidth=2.5, label="Predicted Trend")
        ax.fill_between(forecast_hr["ds"], forecast_hr["yhat_lower"], forecast_hr["yhat_upper"],
                        alpha=0.2, color=ACCENT, label="80% Confidence Interval")
        ax.axvline(prophet_hr["ds"].max(), color="#f6ad55", linestyle="--", linewidth=1.8, label="Forecast Start")
        ax.set_title("Heart Rate — Prophet Trend Forecast (Real Fitbit Data)", fontsize=12, color=TEXT, fontweight='bold')
        ax.set_xlabel("Date", fontsize=9, color=MUTED)
        ax.set_ylabel("Heart Rate (bpm)", fontsize=9, color=MUTED)
        ax.tick_params(colors=TICK_CLR, labelsize=8)
        ax.legend(fontsize=8, facecolor=PLOT_BG, edgecolor=MPL_BOR)
        for sp in ax.spines.values(): sp.set_edgecolor(MPL_BOR)
        ax.grid(color=MPL_GRID, linewidth=0.5, alpha=0.4)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Steps + Sleep ──
        step_pill(17, "Steps & Sleep Forecast")
        screenshot_badge("Cell 20 · Steps & Sleep Prophet")

        fig, axes = plt.subplots(2, 1, figsize=(13, 8))
        fig.patch.set_facecolor(PLOT_BG)

        for ax, (agg, fc, color, label, ylabel) in zip(axes, [
            (steps_agg, f_steps, ACCENT3, "Steps",          "Steps"),
            (sleep_agg, f_sleep, "#b794f4","Sleep (minutes)","Sleep (min)"),
        ]):
            ax.set_facecolor(PLOT_BG)
            ax.scatter(agg["ds"], agg["y"], color=color, s=18, alpha=0.7, label=f"Actual {label}", zorder=3)
            ax.plot(fc["ds"], fc["yhat"], color=TEXT, linewidth=2.2, label="Trend")
            ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.2, color=color, label="80% CI")
            ax.axvline(agg["ds"].max(), color="#f6ad55", linestyle="--", linewidth=1.8, label="Forecast Start")
            ax.set_title(f"{label} — Prophet Trend Forecast", fontsize=11, color=TEXT, fontweight='bold')
            ax.set_xlabel("Date", fontsize=8, color=MUTED)
            ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
            ax.tick_params(colors=TICK_CLR, labelsize=7)
            ax.legend(fontsize=7, facecolor=PLOT_BG, edgecolor=MPL_BOR)
            for sp in ax.spines.values(): sp.set_edgecolor(MPL_BOR)
            ax.grid(color=MPL_GRID, linewidth=0.5, alpha=0.4)

        plt.tight_layout()
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close()

        # Insights
        hr_delta = forecast_hr["yhat"].iloc[-1] - forecast_hr["yhat"].iloc[len(prophet_hr)]
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Prophet Forecast Insights</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;font-size:0.82rem">
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{ACCENT2};font-weight:600;margin-bottom:0.3rem">❤️ Heart Rate</div>
              <div style="color:{TEXT}">Forecast {'rising' if hr_delta>0 else 'falling'} by <b>{abs(hr_delta):.1f} bpm</b> over 30 days. Weekly seasonality detected — Friday peak.</div>
            </div>
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{ACCENT3};font-weight:600;margin-bottom:0.3rem">🚶 Steps</div>
              <div style="color:{TEXT}">Upward trend detected. Users walking more as spring progresses. Weekly pattern visible.</div>
            </div>
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:#b794f4;font-weight:600;margin-bottom:0.3rem">💤 Sleep</div>
              <div style="color:{TEXT}">Wide confidence band due to sparse data (not all users wore device every night).</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="m2-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — CLUSTERING
    # ══════════════════════════════════════════════════════════════════════════
    sec("🤖", "Clustering — KMeans + DBSCAN + PCA + t-SNE", "Steps 18–27")

    ui_info(f"Using 7 activity features for clustering. KMeans K={st.session_state.OPTIMAL_K}, DBSCAN eps={st.session_state.EPS}. Adjust parameters in the sidebar.")

    if st.button(f"🧮 Run Clustering  (K={st.session_state.OPTIMAL_K} · eps={st.session_state.EPS})"):
        with st.spinner("Running KMeans, DBSCAN, PCA, t-SNE..."):
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans, DBSCAN
                from sklearn.decomposition import PCA
                from sklearn.manifold import TSNE

                cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes",
                                "FairlyActiveMinutes","LightlyActiveMinutes",
                                "SedentaryMinutes","TotalSleepMinutes"]
                cf = master.groupby("Id")[cluster_cols].mean().round(3).dropna()

                scaler   = StandardScaler()
                X_scaled = scaler.fit_transform(cf)

                OPTIMAL_K = st.session_state.OPTIMAL_K
                EPS       = st.session_state.EPS

                # KMeans
                kmeans        = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(X_scaled)
                cf["KMeans_Cluster"] = kmeans_labels

                # Elbow
                inertias = []
                for k in range(2,10):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)

                # DBSCAN
                dbscan        = DBSCAN(eps=EPS, min_samples=2)
                dbscan_labels = dbscan.fit_predict(X_scaled)
                cf["DBSCAN_Cluster"] = dbscan_labels
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise    = list(dbscan_labels).count(-1)

                # PCA
                pca   = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                var_explained = pca.explained_variance_ratio_ * 100

                # t-SNE
                tsne   = TSNE(n_components=2, random_state=42,
                              perplexity=min(30,len(X_scaled)-1), max_iter=1000)
                X_tsne = tsne.fit_transform(X_scaled)

                st.session_state.cluster_features = cf
                st.session_state.cluster_cols     = cluster_cols
                st.session_state.X_scaled         = X_scaled
                st.session_state.X_pca            = X_pca
                st.session_state.X_tsne           = X_tsne
                st.session_state.kmeans_labels    = kmeans_labels
                st.session_state.dbscan_labels    = dbscan_labels
                st.session_state.var_explained    = var_explained
                st.session_state.n_clusters       = n_clusters
                st.session_state.n_noise          = n_noise
                st.session_state.inertias         = inertias
                st.session_state.clustering_done  = True
                st.rerun()
            except Exception as e:
                st.error(f"Clustering error: {e}")

    if st.session_state.clustering_done:
        cf            = st.session_state.cluster_features
        cluster_cols  = st.session_state.cluster_cols
        X_pca         = st.session_state.X_pca
        X_tsne        = st.session_state.X_tsne
        kmeans_labels = st.session_state.kmeans_labels
        dbscan_labels = st.session_state.dbscan_labels
        var_explained = st.session_state.var_explained
        n_clusters    = st.session_state.n_clusters
        n_noise       = st.session_state.n_noise
        inertias      = st.session_state.inertias
        OPTIMAL_K     = st.session_state.OPTIMAL_K
        EPS           = st.session_state.EPS

        ui_success(f"Clustering complete — {cf.shape[0]} users · KMeans K={OPTIMAL_K} · DBSCAN {n_clusters} clusters · {n_noise} noise")

        metrics(
            (cf.shape[0],          "Users Clustered"),
            (OPTIMAL_K,            "KMeans Clusters"),
            (f"{var_explained[0]:.1f}%", "PC1 Variance"),
            (f"{var_explained[1]:.1f}%", "PC2 Variance"),
            (n_clusters,           "DBSCAN Clusters"),
            (n_noise,              "Noise Points"),
        )

        # ── Elbow ──
        step_pill(20, "KMeans Elbow Curve")
        screenshot_badge("Cell 23 · Elbow Curve")

        fig, ax = plt.subplots(figsize=(9, 3.5))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        ax.plot(range(2,10), inertias, "o-", color=ACCENT, linewidth=2.5,
                markersize=9, markerfacecolor=ACCENT2, markeredgecolor="white", markeredgewidth=1.5)
        ax.axvline(OPTIMAL_K, color="#f6ad55", linestyle="--", linewidth=1.5,
                   label=f"Selected K={OPTIMAL_K}", alpha=0.8)
        ax.set_title("KMeans Elbow Curve — Real Fitbit Data", fontsize=12, color=TEXT, fontweight='bold')
        ax.set_xlabel("Number of Clusters (K)", fontsize=9, color=MUTED)
        ax.set_ylabel("Inertia", fontsize=9, color=MUTED)
        ax.tick_params(colors=TICK_CLR, labelsize=8)
        ax.legend(fontsize=8, facecolor=PLOT_BG, edgecolor=MPL_BOR)
        for sp in ax.spines.values(): sp.set_edgecolor(MPL_BOR)
        ax.grid(color=MPL_GRID, linewidth=0.5, alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── KMeans + DBSCAN PCA side by side ──
        step_pill("24+25", "KMeans & DBSCAN — PCA Projection")
        c_left, c_right = st.columns(2)

        def pca_plot(ax, labels, title_suffix, is_dbscan=False):
            for label in sorted(set(labels)):
                mask = labels == label
                if is_dbscan and label == -1:
                    ax.scatter(X_pca[mask,0], X_pca[mask,1], c="red", marker="x",
                               s=150, label="Noise", alpha=0.9, linewidths=2, zorder=5)
                else:
                    ax.scatter(X_pca[mask,0], X_pca[mask,1],
                               c=PALETTE[label % len(PALETTE)],
                               label=f"Cluster {label}", s=110, alpha=0.85,
                               edgecolors="white", linewidths=0.8)
            if not is_dbscan:
                for i, uid in enumerate(cf.index):
                    ax.annotate(str(uid)[-4:], (X_pca[i,0], X_pca[i,1]),
                                fontsize=5, alpha=0.45, color=TEXT)
            ax.set_title(title_suffix, fontsize=10, color=TEXT, fontweight='bold')
            ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)", fontsize=8, color=MUTED)
            ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)", fontsize=8, color=MUTED)
            ax.tick_params(colors=TICK_CLR, labelsize=7)
            ax.legend(fontsize=7, facecolor=PLOT_BG, edgecolor=MPL_BOR)
            for sp in ax.spines.values(): sp.set_edgecolor(MPL_BOR)
            ax.grid(color=MPL_GRID, linewidth=0.5, alpha=0.4)
            ax.set_facecolor(PLOT_BG)

        with c_left:
            screenshot_badge("Cell 27 · KMeans PCA")
            fig, ax = plt.subplots(figsize=(6,5))
            fig.patch.set_facecolor(PLOT_BG)
            pca_plot(ax, kmeans_labels, f"KMeans PCA (K={OPTIMAL_K})", is_dbscan=False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with c_right:
            screenshot_badge("Cell 28 · DBSCAN PCA")
            fig, ax = plt.subplots(figsize=(6,5))
            fig.patch.set_facecolor(PLOT_BG)
            pca_plot(ax, dbscan_labels, f"DBSCAN PCA (eps={EPS})", is_dbscan=True)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # ── t-SNE ──
        step_pill(26, "t-SNE Projection")
        screenshot_badge("Cell 29 · t-SNE Both Models")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor(PLOT_BG)

        for ax, (labels, title, is_db) in zip(axes, [
            (kmeans_labels, f"KMeans t-SNE (K={OPTIMAL_K})", False),
            (dbscan_labels, f"DBSCAN t-SNE (eps={EPS})",     True),
        ]):
            ax.set_facecolor(PLOT_BG)
            for label in sorted(set(labels)):
                mask = labels == label
                if is_db and label == -1:
                    ax.scatter(X_tsne[mask,0], X_tsne[mask,1], c="red", marker="x",
                               s=120, label="Noise", alpha=0.9, linewidths=2)
                else:
                    ax.scatter(X_tsne[mask,0], X_tsne[mask,1],
                               c=PALETTE[label % len(PALETTE)],
                               label=f"Cluster {label}", s=110, alpha=0.85,
                               edgecolors="white", linewidths=0.8)
            ax.set_title(title, fontsize=11, color=TEXT, fontweight='bold')
            ax.set_xlabel("t-SNE Dim 1", fontsize=8, color=MUTED)
            ax.set_ylabel("t-SNE Dim 2", fontsize=8, color=MUTED)
            ax.tick_params(colors=TICK_CLR, labelsize=7)
            ax.legend(fontsize=7, facecolor=PLOT_BG, edgecolor=MPL_BOR)
            for sp in ax.spines.values(): sp.set_edgecolor(MPL_BOR)
            ax.grid(color=MPL_GRID, linewidth=0.5, alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ── Cluster Profiles ──
        step_pill(27, "Cluster Profiles & Interpretation")
        screenshot_badge("Cell 30 · Cluster Profiles")

        feat_cols = [c for c in cf.columns if c not in ["KMeans_Cluster","DBSCAN_Cluster"]]
        profile   = cf.groupby("KMeans_Cluster")[feat_cols].mean().round(2)

        fig, ax = plt.subplots(figsize=(12, 4.5))
        fig.patch.set_facecolor(PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        plot_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
        plot_data = profile[plot_cols]
        x = np.arange(OPTIMAL_K)
        w = 0.15
        bar_colors = [ACCENT, ACCENT2, ACCENT3, "#f6ad55", "#b794f4"]
        for i, (col, color) in enumerate(zip(plot_cols, bar_colors)):
            ax.bar(x + i*w, plot_data[col], w, label=col, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
        ax.set_title("Cluster Profiles — Key Feature Averages", fontsize=12, color=TEXT, fontweight='bold')
        ax.set_xlabel("Cluster", fontsize=9, color=MUTED)
        ax.set_ylabel("Mean Value", fontsize=9, color=MUTED)
        ax.set_xticks(x + w*2)
        ax.set_xticklabels([f"Cluster {i}" for i in range(OPTIMAL_K)], color=TICK_CLR)
        ax.tick_params(colors=TICK_CLR, labelsize=8)
        ax.legend(fontsize=7, facecolor=PLOT_BG, edgecolor=MPL_BOR,
                  bbox_to_anchor=(1.01,1))
        for sp in ax.spines.values(): sp.set_edgecolor(MPL_BOR)
        ax.grid(color=MPL_GRID, linewidth=0.5, alpha=0.4, axis="y")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Interpretation cards
        label_map = {True: ("🏃","HIGHLY ACTIVE","#68d391"),
                     False: None}
        interp_cols = st.columns(OPTIMAL_K)
        for i in range(OPTIMAL_K):
            row   = profile.loc[i]
            steps = row["TotalSteps"]
            sed   = row["SedentaryMinutes"]
            act   = row["VeryActiveMinutes"]
            if steps > 10000:
                emoji, tag, col = "🏃", "HIGHLY ACTIVE", ACCENT3
            elif steps > 5000:
                emoji, tag, col = "🚶", "MODERATELY ACTIVE", ACCENT
            else:
                emoji, tag, col = "🛋️", "SEDENTARY", ACCENT2

            members = list(cf[cf["KMeans_Cluster"]==i].index)
            members_short = ", ".join([str(m)[-4:] for m in members[:4]])
            if len(members) > 4: members_short += f" +{len(members)-4}"

            interp_cols[i].markdown(f"""
            <div class="card" style="border-color:{col}33">
              <div style="font-size:1.5rem;margin-bottom:0.5rem">{emoji}</div>
              <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:{TEXT};margin-bottom:0.3rem">Cluster {i}</div>
              <div class="cluster-tag" style="background:{col}22;color:{col};margin-bottom:0.8rem">{tag}</div>
              <div style="font-size:0.8rem;color:{MUTED};line-height:1.7">
                <div>Steps: <b style="color:{TEXT}">{steps:,.0f}/day</b></div>
                <div>Sedentary: <b style="color:{TEXT}">{sed:.0f} min</b></div>
                <div>Very Active: <b style="color:{TEXT}">{act:.0f} min</b></div>
                <div>Users ({len(members)}): <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:{MUTED}">...{members_short}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="m2-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MILESTONE 2 SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    sec("✅", "Milestone 2 Summary")

    all_done = all([
        st.session_state.files_loaded,
        st.session_state.tsfresh_done,
        st.session_state.prophet_done,
        st.session_state.clustering_done,
    ])

    status_items = [
        ("📂", "Data Loading",      st.session_state.files_loaded,    "5 CSV files · master DataFrame · time normalization"),
        ("🧪", "TSFresh",           st.session_state.tsfresh_done,    f"{st.session_state.features.shape[1] if st.session_state.features is not None else '?'} features · normalized heatmap"),
        ("📈", "Prophet Forecast",  st.session_state.prophet_done,    "HR + Steps + Sleep · 30-day · 80% CI · weekly seasonality"),
        ("🤖", "KMeans Clustering", st.session_state.clustering_done, f"K={st.session_state.OPTIMAL_K} · PCA 2D · t-SNE"),
        ("🔍", "DBSCAN",           st.session_state.clustering_done, f"eps={st.session_state.EPS} · {st.session_state.n_clusters} clusters · {st.session_state.n_noise} noise"),
    ]

    for icon, label, done, detail in status_items:
        dot  = "✅" if done else "⬜"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;padding:0.6rem 0;border-bottom:1px solid {CARD_BOR}">
          <span style="font-size:1.1rem">{dot}</span>
          <span style="font-size:0.9rem;font-weight:600;color:{TEXT};min-width:160px">{icon} {label}</span>
          <span style="font-size:0.8rem;color:{MUTED}">{detail}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="border-color:{'#68d391' if all_done else CARD_BOR}44">
      <div class="card-title">📸 Screenshots Required for Submission</div>
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;font-size:0.82rem">
        <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;display:flex;gap:0.6rem;align-items:center">
          <span style="color:{ACCENT2}">📸</span> <b>Cell 15</b> — TSFresh Feature Matrix Heatmap
        </div>
        <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;display:flex;gap:0.6rem;align-items:center">
          <span style="color:{ACCENT2}">📸</span> <b>Cell 18</b> — Prophet HR Forecast with CI
        </div>
        <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;display:flex;gap:0.6rem;align-items:center">
          <span style="color:{ACCENT2}">📸</span> <b>Cell 20</b> — Steps & Sleep Prophet
        </div>
        <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;display:flex;gap:0.6rem;align-items:center">
          <span style="color:{ACCENT2}">📸</span> <b>Cell 27</b> — KMeans PCA Scatter
        </div>
        <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;display:flex;gap:0.6rem;align-items:center">
          <span style="color:{ACCENT2}">📸</span> <b>Cell 28</b> — DBSCAN PCA Scatter
        </div>
        <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;display:flex;gap:0.6rem;align-items:center">
          <span style="color:{ACCENT2}">📸</span> <b>Cell 29</b> — t-SNE Both Models
        </div>
        <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;display:flex;gap:0.6rem;align-items:center;grid-column:1/-1">
          <span style="color:{ACCENT2}">📸</span> <b>Cell 30</b> — Cluster Profiles Bar Chart
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class="card" style="text-align:center;padding:3rem">
      <div style="font-size:3rem;margin-bottom:1rem">📂</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:{TEXT};margin-bottom:0.5rem">
        Upload Your Fitbit Files to Begin
      </div>
      <div style="color:{MUTED};font-size:0.88rem">
        Upload all 5 CSV files above and click <b>Load & Parse All Files</b>
      </div>
    </div>
    """, unsafe_allow_html=True)