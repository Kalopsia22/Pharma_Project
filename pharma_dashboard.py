"""
Indian Pharmaceutical Intelligence — Streamlit Mega Dashboard
Combines Modules 1–5: Market Intelligence, Price Analytics, Ingredient Intelligence,
                       Discontinuation Risk, Manufacturer Segmentation, Demand Proxy
Run:  streamlit run pharma_dashboard.py
Deps: pip install streamlit pandas numpy plotly scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Pharma Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme constants ───────────────────────────────────────────────────────────
# Precision Science palette — deep navy base, phosphor accent, amber warning
C_BG        = "#080D14"        # near-black navy
C_SURFACE   = "#0E1620"        # card surface
C_SURFACE2  = "#131E2C"        # elevated surface
C_BORDER    = "#1C2D3F"        # subtle border
C_ACCENT    = "#00E5A0"        # phosphor green
C_ACCENT2   = "#00B8FF"        # electric blue
C_WARN      = "#FF5D5D"        # coral red
C_GOLD      = "#FFB830"        # amber
C_PURPLE    = "#C084FC"        # violet
C_TEXT      = "#E8EFF7"        # primary text
C_TEXT2     = "#7A92AA"        # secondary text
C_TEXT3     = "#3D5166"        # muted text

# Legacy aliases for the 1800 lines of chart code below — don't change these
ACCENT      = C_ACCENT2        # blue used in most charts
WARN        = C_WARN
OK          = C_ACCENT         # green
GOLD        = C_GOLD
COLORS      = ["#00B8FF","#00E5A0","#FFB830","#FF5D5D","#C084FC",
               "#38BDF8","#34D399","#FB923C","#F472B6","#818CF8",
               "#67E8F9","#86EFAC","#FDE68A","#FCA5A5","#DDD6FE"]
TEMPLATE    = "plotly_dark"

def hex_to_rgba(color, alpha=0.15):
    color = color.strip()
    if color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    elif color.startswith("rgb("):
        return f"rgba({color[4:-1]},{alpha})"
    return color

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {{
    font-family: 'Outfit', sans-serif;
    background-color: {C_BG};
    color: {C_TEXT};
}}
.main {{ background: {C_BG}; }}
.block-container {{ padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1400px; }}

/* Animated page background — subtle moving grid */
.main::before {{
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(0,229,160,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,160,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
    animation: gridShift 20s linear infinite;
}}
@keyframes gridShift {{
    0%   {{ background-position: 0 0; }}
    100% {{ background-position: 40px 40px; }}
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(175deg, #06101A 0%, #0A1829 60%, #081422 100%) !important;
    border-right: 1px solid {C_BORDER};
}}
section[data-testid="stSidebar"]::after {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background-image: radial-gradient(ellipse at 30% 20%, rgba(0,229,160,0.06) 0%, transparent 60%);
    pointer-events: none;
}}
section[data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}
section[data-testid="stSidebar"] .stRadio label {{ color: {C_TEXT2} !important; font-size: 0.88rem; }}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {{
    background: transparent;
}}
section[data-testid="stSidebar"] hr {{ border-color: {C_BORDER}; }}

/* ── KPI Cards ── */
.kpi-card {{
    background: linear-gradient(135deg, {C_SURFACE} 0%, {C_SURFACE2} 100%);
    border: 1px solid {C_BORDER};
    border-radius: 16px;
    padding: 22px 18px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 10px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.kpi-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, {C_ACCENT2}, {C_ACCENT});
    border-radius: 16px 16px 0 0;
}}
.kpi-card::after {{
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 100px; height: 100px;
    background: radial-gradient(circle, rgba(0,229,160,0.08) 0%, transparent 70%);
    border-radius: 50%;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,229,160,0.12);
}}
.kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem; font-weight: 700;
    color: {C_TEXT}; line-height: 1.1;
    letter-spacing: -0.02em;
}}
.kpi-label {{
    font-size: 0.72rem; color: {C_TEXT2};
    margin-top: 6px; text-transform: uppercase;
    letter-spacing: 0.10em; font-weight: 500;
}}
.kpi-card.warn::before  {{ background: linear-gradient(90deg, {C_WARN}, #FF8C42); }}
.kpi-card.warn  .kpi-value  {{ color: {C_WARN}; }}
.kpi-card.ok::before    {{ background: linear-gradient(90deg, {C_ACCENT}, #00C896); }}
.kpi-card.ok    .kpi-value  {{ color: {C_ACCENT}; }}
.kpi-card.gold::before  {{ background: linear-gradient(90deg, {C_GOLD}, #FFD666); }}
.kpi-card.gold  .kpi-value  {{ color: {C_GOLD}; }}
.kpi-card.purple::before{{ background: linear-gradient(90deg, {C_PURPLE}, #E879F9); }}
.kpi-card.purple .kpi-value {{ color: {C_PURPLE}; }}

/* ── Section Headers ── */
.section-header {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem; font-weight: 400;
    color: {C_TEXT};
    display: flex; align-items: center; gap: 12px;
    margin: 32px 0 18px 0;
    letter-spacing: -0.01em;
}}
.section-header::before {{
    content: '';
    display: inline-block;
    width: 4px; height: 28px;
    background: linear-gradient(180deg, {C_ACCENT2}, {C_ACCENT});
    border-radius: 2px; flex-shrink: 0;
}}
.section-header::after {{
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(90deg, {C_BORDER}, transparent);
    margin-left: 12px;
}}

/* ── Insight Box ── */
.insight-box {{
    background: linear-gradient(135deg,
        rgba(0,184,255,0.06) 0%,
        rgba(0,229,160,0.04) 100%);
    border: 1px solid rgba(0,184,255,0.2);
    border-left: 3px solid {C_ACCENT2};
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 14px 0;
    font-size: 0.90rem; color: {C_TEXT2};
    line-height: 1.6;
    position: relative;
}}
.insight-box::before {{
    content: '◈';
    color: {C_ACCENT2}; font-size: 0.9rem;
    margin-right: 8px; opacity: 0.8;
}}

/* ── Metric Badges ── */
.metric-badge {{
    display: inline-block;
    background: rgba(0,184,255,0.12);
    color: {C_ACCENT2};
    border: 1px solid rgba(0,184,255,0.25);
    border-radius: 6px; padding: 2px 10px;
    font-size: 0.75rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.03em; margin: 2px;
}}
.metric-badge.warn {{
    background: rgba(255,93,93,0.12); color: {C_WARN};
    border-color: rgba(255,93,93,0.25);
}}
.metric-badge.ok {{
    background: rgba(0,229,160,0.12); color: {C_ACCENT};
    border-color: rgba(0,229,160,0.25);
}}
.metric-badge.gold {{
    background: rgba(255,184,48,0.12); color: {C_GOLD};
    border-color: rgba(255,184,48,0.25);
}}

/* ── Tabs ── */
div[data-testid="stTabs"] button {{
    font-weight: 500; font-size: 0.83rem;
    color: {C_TEXT2} !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s;
}}
div[data-testid="stTabs"] button:hover {{
    color: {C_TEXT} !important;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C_ACCENT} !important;
    border-bottom-color: {C_ACCENT} !important;
    font-weight: 600 !important;
}}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Plotly charts ── */
[data-testid="stPlotlyChart"] {{
    background: transparent !important;
    border: 1px solid {C_BORDER};
    border-radius: 12px; overflow: hidden;
}}

/* ── Headings ── */
h1 {{
    font-family: 'DM Serif Display', serif !important;
    font-weight: 400 !important;
    font-size: 2.2rem !important;
    color: {C_TEXT} !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}}
h1::after {{
    content: '';
    display: block; width: 48px; height: 3px; margin-top: 8px;
    background: linear-gradient(90deg, {C_ACCENT2}, {C_ACCENT});
    border-radius: 2px;
}}
h2, h3 {{
    font-family: 'Outfit', sans-serif !important;
    color: {C_TEXT} !important;
}}

/* ── Spinners / alerts ── */
.stAlert {{ border-radius: 10px; border: none; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C_BG}; }}
::-webkit-scrollbar-thumb {{ background: {C_BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C_TEXT3}; }}

/* ── Caption / small text ── */
.stCaption, [data-testid="stCaptionContainer"] p {{
    color: {C_TEXT2} !important; font-size: 0.82rem !important;
}}

/* ── Progress bars ── */
[data-testid="stProgressBar"] > div {{
    background: linear-gradient(90deg, {C_ACCENT2}, {C_ACCENT}) !important;
    border-radius: 4px;
}}

/* ── Sidebar logo area ── */
.sidebar-brand {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem; font-weight: 400;
    color: {C_TEXT} !important;
    letter-spacing: -0.01em;
    padding: 8px 0 4px;
    display: flex; align-items: center; gap: 10px;
}}
.sidebar-brand span {{
    display: inline-block;
    background: linear-gradient(135deg, {C_ACCENT2}, {C_ACCENT});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.sidebar-stat {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: {C_TEXT2};
    padding: 2px 0; border-bottom: 1px solid {C_BORDER};
    display: flex; justify-content: space-between; align-items: center;
    margin: 3px 0;
}}
.sidebar-stat b {{ color: {C_ACCENT}; }}
</style>
""", unsafe_allow_html=True)


# ── Plotly dark theme applier ─────────────────────────────────────────────────
def _apply_plotly_theme(fig, height=None):
    """Apply the Precision Science dark theme to any plotly figure."""
    updates = dict(
        paper_bgcolor="rgba(14,22,32,0)",
        plot_bgcolor="rgba(14,22,32,0)",
        font=dict(family="Outfit, sans-serif", color=C_TEXT, size=12),
        legend=dict(
            bgcolor="rgba(14,22,32,0.7)",
            bordercolor=C_BORDER, borderwidth=1,
            font=dict(size=11, color=C_TEXT2)),
        margin=dict(t=40, b=40, l=60, r=20),
        hoverlabel=dict(
            bgcolor=C_SURFACE2, bordercolor=C_BORDER,
            font=dict(family="JetBrains Mono", size=12, color=C_TEXT)),
        colorway=COLORS,
    )
    if height:
        updates["height"] = height
    fig.update_layout(**updates)
    fig.update_xaxes(
        gridcolor=C_BORDER, linecolor=C_BORDER,
        tickcolor=C_TEXT3, tickfont=dict(color=C_TEXT2),
        title_font=dict(color=C_TEXT2), zeroline=False)
    fig.update_yaxes(
        gridcolor=C_BORDER, linecolor=C_BORDER,
        tickcolor=C_TEXT3, tickfont=dict(color=C_TEXT2),
        title_font=dict(color=C_TEXT2), zeroline=False)
    return fig





# ╔══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════╝
# ── Data source: Google Drive ─────────────────────────────────────────────────
GDRIVE_FILE_ID = "1TxUQQOepC5yEyIXEbeOHop78sT7ky_7L"
CSV_PATH       = "indian_pharmaceutical_products_clean.csv"

def _download_from_gdrive(file_id, dest):
    """Download large file from Google Drive, handling the virus-scan warning page."""
    import requests
    session = requests.Session()
    base_url = "https://drive.google.com/uc"
    params = {"id": file_id, "export": "download"}

    # First request — may return a confirm page for large files
    response = session.get(base_url, params=params, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    # If we got a confirm token, resend with it
    if token:
        params["confirm"] = token
        response = session.get(base_url, params=params, stream=True)

    # Also handle the newer "confirm=t" style redirect
    if "confirm=" not in response.url and b"confirm=" in response.content[:2000]:
        import re
        match = re.search(rb'confirm=([0-9A-Za-z_]+)', response.content[:2000])
        if match:
            params["confirm"] = match.group(1).decode()
            response = session.get(base_url, params=params, stream=True)

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    import os
    if not os.path.exists(CSV_PATH):
        with st.spinner("Downloading dataset from Google Drive (first run only ~72 MB)..."):
            _download_from_gdrive(GDRIVE_FILE_ID, CSV_PATH)
    df = pd.read_csv(
        CSV_PATH,
        engine="python", quotechar='"', escapechar="\\", on_bad_lines="skip"
    )
    df["is_discontinued"]        = df["is_discontinued"].astype(str).map({"True": True, "False": False})
    df["price_inr"]              = pd.to_numeric(df["price_inr"],              errors="coerce")
    df["pack_size"]              = pd.to_numeric(df["pack_size"],              errors="coerce")
    df["num_active_ingredients"] = pd.to_numeric(df["num_active_ingredients"], errors="coerce")

    df["is_combo"]              = (df["num_active_ingredients"] > 1).astype(int)
    df["mfr_median_price"]      = df["manufacturer"].map(df.groupby("manufacturer")["price_inr"].median())
    df["ingr_median_price"]     = df["primary_ingredient"].map(df.groupby("primary_ingredient")["price_inr"].median())
    df["mfr_product_count"]     = df["manufacturer"].map(df["manufacturer"].value_counts())
    df["ingr_competition"]      = df["primary_ingredient"].map(df["primary_ingredient"].value_counts())
    df["dosage_form_enc"]       = pd.factorize(df["dosage_form"])[0]
    df["therapeutic_class_enc"] = pd.factorize(df["therapeutic_class"])[0]
    df["pack_unit_enc"]         = pd.factorize(df["pack_unit"])[0]
    df["pack_size_filled"]      = df["pack_size"].fillna(df["pack_size"].median())

    def parse_ingredients(s):
        try:
            return [d["name"] for d in json.loads(s.replace("'", '"'))]
        except:
            try:
                return re.findall(r"'name':\s*'([^']+)'", str(s))
            except:
                return []

    df["ingredient_names"] = df["active_ingredients"].apply(parse_ingredients)
    return df

@st.cache_data(show_spinner="Training ML models...")
def train_models(_df):
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        mean_absolute_error, r2_score, mean_squared_error,
        roc_auc_score, accuracy_score, precision_score, recall_score,
        roc_curve, precision_recall_curve, confusion_matrix,
        classification_report, silhouette_score
    )
    from sklearn.decomposition import PCA
    from sklearn.utils import resample

    df = _df.copy()

    PRICE_FEATURES = [
        "num_active_ingredients","is_combo","pack_size_filled","pack_unit_enc",
        "dosage_form_enc","therapeutic_class_enc","mfr_median_price",
        "ingr_median_price","mfr_product_count","ingr_competition"
    ]
    DISC_FEATURES = [
        "price_inr","num_active_ingredients","is_combo","pack_size_filled",
        "dosage_form_enc","therapeutic_class_enc","pack_unit_enc",
        "mfr_product_count","ingr_competition","mfr_median_price","ingr_median_price"
    ]

    # ── Price model ──────────────────────────────────────────────────────────
    df_p = df[PRICE_FEATURES+["price_inr"]].dropna()
    df_p = df_p[df_p["price_inr"] > 0]
    X_p  = df_p[PRICE_FEATURES]; y_p = np.log1p(df_p["price_inr"])
    X_tr,X_te,y_tr,y_te = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
    rf_p = RandomForestRegressor(n_estimators=200,max_depth=12,min_samples_leaf=5,n_jobs=-1,random_state=42)
    rf_p.fit(X_tr, y_tr)
    y_pred_log = rf_p.predict(X_te); y_pred = np.expm1(y_pred_log); y_true = np.expm1(y_te)
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_log = r2_score(y_te, y_pred_log); r2_raw = r2_score(y_true, y_pred)
    mape  = float(np.median(np.abs((y_pred-y_true.values)/np.clip(y_true.values,1,None))*100))
    ape   = np.abs(y_pred-y_true.values)/np.clip(y_true.values,1,None)
    acc_10= float((ape<0.10).mean()*100); acc_25=float((ape<0.25).mean()*100); acc_50=float((ape<0.50).mean()*100)

    # tier precision/recall for price model
    from sklearn.preprocessing import LabelEncoder as LE
    tl = ["Budget","Mid","Premium","Specialty"]
    at = pd.cut(y_true, bins=[0,50,150,500,np.inf], labels=tl)
    pt = pd.cut(pd.Series(y_pred), bins=[0,50,150,500,np.inf], labels=tl)
    le2= LE().fit(tl)
    yt_t = le2.transform(at.dropna()); yp_t = le2.transform(pt.dropna())
    prec_tier_w = precision_score(yt_t,yp_t,average="weighted",zero_division=0)
    rec_tier_w  = recall_score(yt_t,yp_t,average="weighted",zero_division=0)

    res_df = pd.DataFrame({"actual":y_true.values,"predicted":y_pred,"residual":y_pred-y_true.values})
    res_df["pct_err"] = np.abs(res_df["residual"])/np.clip(res_df["actual"],1,None)*100
    res_df = res_df.join(df[["therapeutic_class"]].loc[y_te.index])

    price_fi = pd.Series(rf_p.feature_importances_,index=PRICE_FEATURES).sort_values(ascending=False)

    # ── Disc model ───────────────────────────────────────────────────────────
    df_d  = df[DISC_FEATURES+["is_discontinued"]].dropna()
    X_d   = df_d[DISC_FEATURES]; y_d = df_d["is_discontinued"].astype(int)
    X_tr_d,X_te_d,y_tr_d,y_te_d = train_test_split(X_d,y_d,test_size=0.2,random_state=42,stratify=y_d)
    X_maj,y_maj=X_tr_d[y_tr_d==0],y_tr_d[y_tr_d==0]
    X_min,y_min=X_tr_d[y_tr_d==1],y_tr_d[y_tr_d==1]
    X_up,y_up=resample(X_min,y_min,replace=True,n_samples=len(X_maj)//3,random_state=42)
    rf_d = RandomForestClassifier(n_estimators=200,max_depth=10,class_weight="balanced",n_jobs=-1,random_state=42)
    rf_d.fit(pd.concat([X_maj,X_up]),pd.concat([y_maj,y_up]))
    y_pred_d=rf_d.predict(X_te_d); y_prob_d=rf_d.predict_proba(X_te_d)[:,1]
    auc_d    = roc_auc_score(y_te_d,y_prob_d)
    acc_d    = accuracy_score(y_te_d,y_pred_d)
    prec_disc= precision_score(y_te_d,y_pred_d,zero_division=0)
    rec_disc = recall_score(y_te_d,y_pred_d,zero_division=0)
    prec_mac = precision_score(y_te_d,y_pred_d,average="macro",zero_division=0)
    rec_mac  = recall_score(y_te_d,y_pred_d,average="macro",zero_division=0)
    report_d = classification_report(y_te_d,y_pred_d,target_names=["Active","Discontinued"],output_dict=True)
    fpr,tpr,_ = roc_curve(y_te_d,y_prob_d)
    prec_pr,rec_pr,_=precision_recall_curve(y_te_d,y_prob_d)
    cm_d = confusion_matrix(y_te_d,y_pred_d)
    disc_fi = pd.Series(rf_d.feature_importances_,index=DISC_FEATURES).sort_values(ascending=False)

    # Score all products
    all_scores = rf_d.predict_proba(df[DISC_FEATURES].fillna(0))[:,1]
    df["disc_risk_score"] = all_scores
    df["risk_tier"] = pd.cut(all_scores,bins=[0,0.2,0.4,0.6,1.0],
        labels=["Low","Moderate","High","Critical"]).astype(str)

    # ── Price tier classifier ────────────────────────────────────────────────
    df["price_tier"] = pd.cut(df["price_inr"],bins=[0,50,150,500,float("inf")],
        labels=["Budget","Mid","Premium","Specialty"]).astype(str)
    df_t = df[df["price_tier"]!="nan"][PRICE_FEATURES+["price_tier"]].dropna()
    le_t = LE(); y_t = le_t.fit_transform(df_t["price_tier"]); X_t = df_t[PRICE_FEATURES]
    X_tr_t,X_te_t,y_tr_t,y_te_t = train_test_split(X_t,y_t,test_size=0.2,random_state=42,stratify=y_t)
    tc = RandomForestClassifier(n_estimators=150,max_depth=10,n_jobs=-1,random_state=42)
    tc.fit(X_tr_t,y_tr_t); y_pred_t=tc.predict(X_te_t)
    acc_t   = accuracy_score(y_te_t,y_pred_t)
    prec_t_w= precision_score(y_te_t,y_pred_t,average="weighted",zero_division=0)
    rec_t_w = recall_score(y_te_t,y_pred_t,average="weighted",zero_division=0)
    prec_t_m= precision_score(y_te_t,y_pred_t,average="macro",zero_division=0)
    rec_t_m = recall_score(y_te_t,y_pred_t,average="macro",zero_division=0)
    report_t= classification_report(y_te_t,y_pred_t,target_names=le_t.classes_,output_dict=True)
    tier_cm = confusion_matrix(y_te_t,y_pred_t)
    tier_fi = pd.Series(tc.feature_importances_,index=PRICE_FEATURES).sort_values(ascending=False)

    # ── Clustering ───────────────────────────────────────────────────────────
    mfr_profile = df.groupby("manufacturer").agg(
        product_count=("product_id","count"),median_price=("price_inr","median"),
        combo_ratio=("is_combo","mean"),discontinuation_rate=("is_discontinued","mean"),
        class_diversity=("therapeutic_class","nunique"),form_diversity=("dosage_form","nunique"),
        ingr_diversity=("primary_ingredient","nunique"),avg_price=("price_inr","mean"),
    ).reset_index().fillna(0)
    mfr_profile = mfr_profile[mfr_profile["product_count"]>=5]
    CLUS_F = ["product_count","median_price","combo_ratio","discontinuation_rate",
              "class_diversity","form_diversity","ingr_diversity"]
    scaler = StandardScaler(); X_sc = scaler.fit_transform(mfr_profile[CLUS_F])
    inertias = [KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_sc).inertia_ for k in range(2,11)]
    km = KMeans(n_clusters=5,random_state=42,n_init=20); mfr_profile["cluster"]=km.fit_predict(X_sc)
    sil = silhouette_score(X_sc,mfr_profile["cluster"])
    total_var=np.var(X_sc,axis=0).sum()*len(X_sc); between_var=total_var-km.inertia_
    var_exp = between_var/total_var
    pca=PCA(n_components=2,random_state=42); coords=pca.fit_transform(X_sc)
    mfr_profile["pca_x"]=coords[:,0]; mfr_profile["pca_y"]=coords[:,1]
    CNAMES={0:"Mass Market Generics",1:"Market Leaders",2:"Premium Specialists",
            3:"Ultra-Premium Niche",4:"Mid-Tier Diversified"}
    mfr_profile["cluster_label"]=mfr_profile["cluster"].map(CNAMES)
    mfr_profile["short_name"]=mfr_profile["manufacturer"].str.replace(
        r"(Pharmaceuticals?|Industries?|Laboratories?|Products?) ?(Ltd|Pvt\.? Ltd|Limited)?","",regex=True).str.strip()

    # ── Demand proxy ─────────────────────────────────────────────────────────
    cw_map = {"antibiotic":0.9,"analgesic":0.85,"antacid":0.8,"antihistamine":0.75,
              "antidiabetic":0.95,"antihypertensive":0.9,"corticosteroid":0.7,
              "antidepressant":0.85,"bronchodilator":0.8,"diuretic":0.7,"other":0.5}
    df["class_weight"]=df["therapeutic_class"].map(cw_map).fillna(0.5)
    df["demand_proxy"]=(np.log1p(df["ingr_competition"])*df["class_weight"]*(1+0.5*df["is_combo"])).round(3)
    demand_ingr=df.groupby("primary_ingredient").agg(
        demand_score=("demand_proxy","mean"),products=("product_id","count"),
        therapeutic_class=("therapeutic_class",lambda x:x.mode()[0]),
        median_price=("price_inr","median"),manufacturers=("manufacturer","nunique")
    ).reset_index().sort_values("demand_score",ascending=False)

    return dict(
        df=df, mfr_profile=mfr_profile, demand_ingr=demand_ingr,
        # price
        price_fi=price_fi, res_df=res_df,
        r2_log=r2_log, r2_raw=r2_raw, mae=mae, rmse=rmse, mape=mape,
        acc_10=acc_10, acc_25=acc_25, acc_50=acc_50,
        prec_tier_w=prec_tier_w, rec_tier_w=rec_tier_w,
        # disc
        disc_fi=disc_fi, fpr=fpr, tpr=tpr, prec_pr=prec_pr, rec_pr=rec_pr, cm_d=cm_d,
        auc_d=auc_d, acc_d=acc_d, prec_disc=prec_disc, rec_disc=rec_disc,
        prec_mac=prec_mac, rec_mac=rec_mac, report_d=report_d,
        # tier
        tier_fi=tier_fi, tier_cm=tier_cm, le_t=le_t, report_t=report_t,
        acc_t=acc_t, prec_t_w=prec_t_w, rec_t_w=rec_t_w, prec_t_m=prec_t_m, rec_t_m=rec_t_m,
        # cluster
        sil=sil, var_exp=var_exp, inertia=km.inertia_, inertias=inertias,
        # misc
        PRICE_FEATURES=PRICE_FEATURES, DISC_FEATURES=DISC_FEATURES, CLUS_F=CLUS_F,
        CNAMES=CNAMES,
    )


# ── Helper widgets ────────────────────────────────────────────────────────────
def kpi(label, value, style="default", fmt=None):
    if fmt:
        val_str = fmt.format(value)
    elif isinstance(value, float) and value < 10:
        val_str = f"{value:.4f}"
    elif isinstance(value, float):
        val_str = f"{value:,.1f}"
    else:
        val_str = f"{value:,}"
    cls = {"warn": "warn", "ok": "ok", "gold": "gold", "purple": "purple"}.get(style, "")
    st.markdown(f"""
    <div class="kpi-card {cls}">
        <div class="kpi-value">{val_str}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def badge(text, style="default"):
    cls = {"warn":"warn","ok":"ok","gold":"gold"}.get(style,"")
    return f'<span class="metric-badge {cls}">{text}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()
M  = train_models(df)
df = M["df"]  # updated with risk scores


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class='sidebar-brand'>
        <span>💊</span> Pharma<span>Intel</span>
    </div>
    <div style='font-size:0.72rem;color:{C_TEXT3};margin-bottom:16px;letter-spacing:0.06em;text-transform:uppercase;'>
        Indian Pharma Intelligence Platform
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Overview",
        "📊 Market Intelligence",
        "💰 Price Analytics",
        "🧪 Ingredient Intelligence",
        "🤖 ML: Price Prediction",
        "⚠️ ML: Discontinuation Risk",
        "🏭 ML: Market Segmentation",
        "🏷️ ML: Price Tier Classifier",
        "📈 ML: Demand Scorer",
        "🎯 Model Comparison",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown(f"""
    <div style='font-size:0.72rem;color:{C_TEXT3};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;'>Dataset</div>
    <div class='sidebar-stat'><span>Products</span><b>{len(df):,}</b></div>
    <div class='sidebar-stat'><span>Manufacturers</span><b>{df['manufacturer'].nunique():,}</b></div>
    <div class='sidebar-stat'><span>Ingredients</span><b>{df['primary_ingredient'].nunique():,}</b></div>
    <div class='sidebar-stat'><span>Classes</span><b>{df['therapeutic_class'].nunique()}</b></div>
    <div style='height:12px;'></div>
    <div style='font-size:0.72rem;color:{C_TEXT3};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;'>Coverage</div>
    <div class='sidebar-stat'><span>Market charts</span><b style='color:{C_ACCENT2};'>9</b></div>
    <div class='sidebar-stat'><span>Price charts</span><b style='color:{C_GOLD};'>13</b></div>
    <div class='sidebar-stat'><span>Ingredient charts</span><b style='color:{C_ACCENT};'>15</b></div>
    <div class='sidebar-stat'><span>ML models</span><b style='color:{C_PURPLE};'>5</b></div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("💊 Indian Pharmaceutical Intelligence Dashboard")
    st.markdown("A unified analytics and ML platform covering market, pricing, ingredient, and predictive intelligence.")
    st.markdown("---")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi("Total Products", len(df), "default", "{:,}")
    with c2: kpi("Manufacturers", df["manufacturer"].nunique(), "default", "{:,}")
    with c3: kpi("Ingredients", df["primary_ingredient"].nunique(), "default", "{:,}")
    with c4: kpi("Discontinued", int(df["is_discontinued"].sum()), "warn", "{:,}")
    with c5: kpi("Median Price ₹", df["price_inr"].median(), "gold", "₹{:.0f}")
    with c6: kpi("Combo Drugs %", df["is_combo"].mean()*100, "ok", "{:.1f}%")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("Products by Therapeutic Class")
        tc = df["therapeutic_class"].value_counts().head(11).reset_index()
        tc.columns = ["class","count"]
        fig = px.bar(tc, y="class", x="count", orientation="h",
                     color="count",
                     color_continuous_scale=[[0,C_BORDER],[0.4,C_ACCENT2],[1,C_ACCENT]],
                     template=TEMPLATE)
        _apply_plotly_theme(fig, height=380)
        fig.update_layout(showlegend=False, margin=dict(t=10,b=10),
                          coloraxis_showscale=False)
        fig.update_yaxes(title=""); fig.update_xaxes(title="Products")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Price Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["price_inr"].clip(0,500), nbinsx=60,
            marker_color=C_ACCENT2, opacity=0.85,
            marker_line_color=C_BG, marker_line_width=0.4,
        ))
        fig.add_vline(x=df["price_inr"].median(), line_dash="dash", line_color=C_GOLD,
                      annotation_text=f"Median ₹{df['price_inr'].median():.0f}",
                      annotation_font_color=C_GOLD)
        _apply_plotly_theme(fig, height=380)
        fig.update_layout(margin=dict(t=10,b=10),
                          xaxis_title="Price ₹ (capped at 500)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    section("Module Summary")
    MODULE_META = [
        ("📊", "Market Intelligence",    "Treemap · Heatmaps · Lorenz Curve · Competitive Landscape",                 C_ACCENT2,  "9 charts"),
        ("💰", "Price Analytics",         "Price Tiers · Combo Premium · Dosage Premium · Outlier Explorer",            C_GOLD,     "13 charts"),
        ("🧪", "Ingredient Intelligence", "Co-occurrence Network · Combo Strategy · Portfolio Heatmap",                C_ACCENT,   "15 charts"),
        ("🤖", "Price Prediction",        f"RF Regressor · R²={M['r2_log']:.4f} · MAE=₹{M['mae']:.0f} · Acc±25%={M['acc_25']:.1f}%", C_PURPLE, "ML"),
        ("⚠️", "Discontinuation Risk",    f"RF Classifier · AUC={M['auc_d']:.4f} · Recall(disc)={M['rec_disc']:.4f}", C_WARN,     "ML"),
        ("🏭", "Segmentation",            f"K-Means K=5 · Silhouette={M['sil']:.4f} · VarExp={M['var_exp']*100:.1f}%",C_ACCENT2,  "ML"),
        ("🏷️", "Price Tier",              f"4-class RF · Acc={M['acc_t']:.4f} · Prec(wt)={M['prec_t_w']:.4f}",       C_GOLD,     "ML"),
        ("📈", "Demand Scorer",           f"Rule-based · {len(M['demand_ingr']):,} ingredients · Top: Glimepiride 11.54", C_ACCENT, "ML"),
    ]
    for i in range(0, len(MODULE_META), 2):
        ca, cb = st.columns(2)
        for col_obj, meta in zip([ca, cb], MODULE_META[i:i+2]):
            icon, name, desc, color, tag = meta
            with col_obj:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,{C_SURFACE} 0%,{C_SURFACE2} 100%);
                    border:1px solid {C_BORDER}; border-left:3px solid {color};
                    border-radius:10px; padding:14px 16px; margin-bottom:8px;'>
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                        <div style='font-size:1.1rem;'>{icon}</div>
                        <div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;
                            color:{color};background:rgba(0,0,0,0.2);padding:2px 8px;
                            border-radius:4px;border:1px solid {color}40;'>{tag}</div>
                    </div>
                    <div style='font-weight:600;color:{C_TEXT};font-size:0.92rem;margin:6px 0 3px;'>{name}</div>
                    <div style='font-size:0.76rem;color:{C_TEXT2};line-height:1.4;'>{desc}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET INTELLIGENCE  (Module 1 — all charts)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Market Intelligence":
    st.title("📊 Market Intelligence Dashboard")
    st.caption("Manufacturer portfolios, competitive landscape, pricing positioning, and market concentration")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    mfr_agg = df.groupby("manufacturer").agg(
        products=("product_id","count"),
        avg_price=("price_inr","mean"),
        median_price=("price_inr","median"),
        discontinued=("is_discontinued","sum"),
        active=("is_discontinued", lambda x:(~x).sum()),
    ).reset_index()
    mfr_agg["disc_rate"]      = mfr_agg["discontinued"] / mfr_agg["products"] * 100
    mfr_agg["market_share"]   = mfr_agg["products"] / mfr_agg["products"].sum() * 100
    mfr_agg["combo_pct"]      = df.groupby("manufacturer")["is_combo"].mean().values * 100
    top5_share = mfr_agg.nlargest(5,"products")["market_share"].sum()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi("Total Products",      len(df),                            fmt="{:,}")
    with c2: kpi("Manufacturers",       df["manufacturer"].nunique(),        fmt="{:,}")
    with c3: kpi("Median Price ₹",      df["price_inr"].median(),  "gold",   fmt="₹{:.0f}")
    with c4: kpi("Disc. Rate",          df["is_discontinued"].mean()*100, "warn", fmt="{:.1f}%")
    with c5: kpi("Top-5 Share",         top5_share,                "warn",   fmt="{:.1f}%")
    with c6: kpi("Combo Drug Share",    df["is_combo"].mean()*100, "ok",     fmt="{:.1f}%")

    tabs = st.tabs([
        "Treemap","Top 20 Manufacturers","Portfolio Strategy",
        "Class Heatmap","Class Leaders","Disc. Risk",
        "Price by Manufacturer","Avg Price by Class","Concentration Curve"
    ])

    # ── Tab 0: Treemap ────────────────────────────────────────────────────────
    with tabs[0]:
        section("Market Share Treemap — Top 30 Manufacturers × Therapeutic Class")
        top30_names = mfr_agg.nlargest(30,"products")["manufacturer"].tolist()
        treemap_df = (df[df["manufacturer"].isin(top30_names)]
            .groupby(["manufacturer","therapeutic_class"]).size().reset_index(name="count"))
        price_lk = (df[df["manufacturer"].isin(top30_names)]
            .groupby(["manufacturer","therapeutic_class"])["price_inr"].mean().round(2).reset_index(name="avg_price"))
        treemap_df = treemap_df.merge(price_lk, on=["manufacturer","therapeutic_class"])
        fig = px.treemap(treemap_df,
            path=[px.Constant("Indian Pharma"),"manufacturer","therapeutic_class"],
            values="count", color="avg_price",
            color_continuous_scale="RdYlGn_r",
            hover_data={"avg_price":":.2f"},
            template=TEMPLATE)
        fig.update_traces(textinfo="label+value+percent root",
            hovertemplate="<b>%{label}</b><br>Products: %{value:,}<br>Avg Price: ₹%{color:.2f}<extra></extra>")
        fig.update_layout(height=640, margin=dict(t=20,b=10))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Size = product count | Color = avg price (darker red = higher). "
                "Sun Pharma dominates across antibiotics and analgesics.")

    # ── Tab 1: Top 20 Stacked Bar ─────────────────────────────────────────────
    with tabs[1]:
        section("Top 20 Manufacturers — Active vs Discontinued Products")
        top20 = mfr_agg.nlargest(20,"products").sort_values("products")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top20["manufacturer"], x=top20["active"],
            name="Active", marker_color=OK, orientation="h",
            hovertemplate="<b>%{y}</b><br>Active: %{x:,}<extra></extra>"))
        fig.add_trace(go.Bar(y=top20["manufacturer"], x=top20["discontinued"],
            name="Discontinued", marker_color=WARN, orientation="h",
            hovertemplate="<b>%{y}</b><br>Discontinued: %{x:,}<extra></extra>"))
        for _, row in top20.iterrows():
            fig.add_annotation(x=row["products"]+30, y=row["manufacturer"],
                text=f"{row['market_share']:.2f}%", showarrow=False,
                font=dict(size=9, color="#555"))
        fig.update_layout(barmode="stack", height=620, template=TEMPLATE,
            margin=dict(t=20,r=90),
            legend=dict(orientation="h",y=1.02,x=0.5,xanchor="center"),
            xaxis_title="Number of Products")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Portfolio Strategy Bubble ─────────────────────────────────────
    with tabs[2]:
        section("Portfolio Strategy Map — Top 30 Manufacturers")
        top30_bubble = mfr_agg.nlargest(30,"products").copy()
        top30_bubble["short"] = top30_bubble["manufacturer"].str.replace(
            r"(Pharmaceuticals?|Industries?|Laboratories?|Ltd|Limited)","",regex=True).str.strip()
        med_prod  = top30_bubble["products"].median()
        med_price = top30_bubble["avg_price"].median()
        fig = px.scatter(top30_bubble, x="products", y="avg_price",
            size="market_share", color="disc_rate",
            color_continuous_scale="RdYlGn_r",
            hover_name="manufacturer", text="short",
            size_max=60, template=TEMPLATE,
            labels={"products":"Portfolio Size","avg_price":"Avg Price ₹","disc_rate":"Disc. Rate %"})
        fig.update_traces(textposition="top center", textfont_size=8)
        fig.add_vline(x=med_prod,  line_dash="dash", line_color="grey", opacity=0.5)
        fig.add_hline(y=med_price, line_dash="dash", line_color="grey", opacity=0.5)
        for label,xr,yr in [("Volume Leaders",0.85,0.08),("Premium Specialists",0.08,0.92),
                             ("Market Dominators",0.85,0.92),("Niche Players",0.08,0.08)]:
            fig.add_annotation(xref="paper",yref="paper",x=xr,y=yr,text=f"<i>{label}</i>",
                showarrow=False,font=dict(size=9,color=C_TEXT3),
                bgcolor="rgba(255,255,255,0.6)",bordercolor="#ccc",borderwidth=1)
        fig.update_layout(height=640, margin=dict(t=20))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Bottom-right quadrant = Volume Leaders (high portfolio, lower avg price). "
                "Top-left = Premium Specialists. Color intensity = discontinuation risk.")

    # ── Tab 3: Manufacturer × Class Heatmap ──────────────────────────────────
    with tabs[3]:
        section("Manufacturer × Therapeutic Class Heatmap (Top 15)")
        top15_names = mfr_agg.nlargest(15,"products")["manufacturer"].tolist()
        heat = (df[df["manufacturer"].isin(top15_names)]
            .groupby(["manufacturer","therapeutic_class"]).size()
            .unstack(fill_value=0))
        heat = heat[heat.sum().sort_values(ascending=False).index]
        heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
        short_y = [n.replace(" Pharmaceuticals","").replace(" Pharmaceutical","")
                    .replace(" Industries","").replace(" Ltd","").replace(" Limited","")
                    for n in heat.index]
        fig = go.Figure(data=go.Heatmap(
            z=heat.values, x=[c.title() for c in heat.columns], y=short_y,
            colorscale=[[0,"#0A1829"],[0.3,"#1C4D7A"],[1,"#00B8FF"]], hoverongaps=False,
            text=heat.values, texttemplate="%{text}", textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:,} products<extra></extra>"))
        fig.update_layout(height=520, template=TEMPLATE,
            margin=dict(t=20,l=200,b=80),
            xaxis=dict(tickangle=-30))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Top Manufacturers per Class ───────────────────────────────────
    with tabs[4]:
        section("Top 7 Manufacturers per Therapeutic Class")
        classes_ordered = ["antibiotic","analgesic","antacid","antihistamine",
            "antidiabetic","antihypertensive","corticosteroid",
            "antidepressant","bronchodilator","diuretic"]
        fig = make_subplots(rows=5, cols=2,
            subplot_titles=[c.title() for c in classes_ordered],
            vertical_spacing=0.055, horizontal_spacing=0.18)
        shorten = lambda n: n.replace(" Pharmaceuticals","").replace(" Pharmaceutical","") \
                             .replace(" Industries","").replace(" Ltd","").strip()
        for idx, cls in enumerate(classes_ordered):
            row = idx // 2 + 1; col = idx % 2 + 1
            sub = (df[df["therapeutic_class"]==cls].groupby("manufacturer")
                .size().sort_values().tail(7))
            fig.add_trace(go.Bar(x=sub.values, y=[shorten(n) for n in sub.index],
                orientation="h", marker_color=COLORS[idx % len(COLORS)],
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>Products: %{x}<extra></extra>"),
                row=row, col=col)
        fig.update_layout(height=1400, template=TEMPLATE, margin=dict(t=50,l=20))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Discontinuation Risk ───────────────────────────────────────────
    with tabs[5]:
        section("Discontinuation Risk — Top 40 Manufacturers")
        top40 = mfr_agg.nlargest(40,"products").copy()
        top40["short"] = top40["manufacturer"].str.replace(
            r"(Pharmaceuticals?|Industries?|Ltd|Limited)","",regex=True).str.strip()
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Disc. Rate by Manufacturer","Products vs Disc. Rate — Risk Quadrants"],
            horizontal_spacing=0.12)
        top40_disc = top40.sort_values("disc_rate")
        bar_colors = [WARN if r>10 else "#F0B27A" if r>5 else OK for r in top40_disc["disc_rate"]]
        fig.add_trace(go.Bar(y=top40_disc["short"], x=top40_disc["disc_rate"],
            orientation="h", marker_color=bar_colors, showlegend=False,
            hovertemplate="<b>%{y}</b><br>Disc. Rate: %{x:.1f}%<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=top40["products"], y=top40["disc_rate"],
            mode="markers+text", text=top40["short"], textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=12, color=top40["disc_rate"],
                colorscale="RdYlGn_r", showscale=True,
                colorbar=dict(title="Disc. %", x=1.02)),
            hovertemplate="<b>%{text}</b><br>Products: %{x:,}<br>Disc.: %{y:.1f}%<extra></extra>",
            showlegend=False), row=1, col=2)
        fig.add_hline(y=5,  line_dash="dot", line_color="orange", row=1, col=2,
            annotation_text="5% threshold")
        fig.add_hline(y=10, line_dash="dot", line_color="red", row=1, col=2,
            annotation_text="10% high risk")
        fig.update_layout(height=680, template=TEMPLATE, margin=dict(t=50,r=80))
        fig.update_xaxes(title_text="Disc. Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="Total Products",  row=1, col=2)
        fig.update_yaxes(title_text="Disc. Rate (%)",  row=1, col=2)
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 6: Price Distribution Violin per Manufacturer ────────────────────
    with tabs[6]:
        section("Price Distribution by Manufacturer (Top 15 — Violin)")
        top15_names = mfr_agg.nlargest(15,"products")["manufacturer"].tolist()
        df_t15 = df[df["manufacturer"].isin(top15_names)].copy()
        df_t15 = df_t15[df_t15["price_inr"] < df_t15["price_inr"].quantile(0.98)]
        shorten2 = lambda n: n.replace(" Pharmaceuticals","").replace(" Pharmaceutical","") \
                              .replace(" Industries","").replace(" Ltd","")
        df_t15["short_mfr"] = df_t15["manufacturer"].apply(shorten2)
        order = df_t15.groupby("short_mfr")["price_inr"].median().sort_values(ascending=False).index.tolist()
        fig = go.Figure()
        for i, mname in enumerate(order):
            sub = df_t15[df_t15["short_mfr"]==mname]["price_inr"]
            fig.add_trace(go.Violin(y=sub, name=mname,
                box_visible=True, meanline_visible=True,
                fillcolor=COLORS[i%len(COLORS)], opacity=0.7,
                line_color="black",
                hovertemplate=f"<b>{mname}</b><br>₹%{{y:.2f}}<extra></extra>"))
        fig.add_hline(y=df_t15["price_inr"].median(), line_dash="dash",
            line_color="black", opacity=0.5,
            annotation_text=f"Market Median ₹{df_t15['price_inr'].median():.0f}",
            annotation_position="right")
        fig.update_layout(height=560, template=TEMPLATE,
            showlegend=False, margin=dict(t=20,r=120),
            yaxis_title="Price ₹", xaxis_title="Manufacturer")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 7: Avg Price by Therapeutic Class ────────────────────────────────
    with tabs[7]:
        section("Average Price by Therapeutic Class (Mean ± SD + Median)")
        class_price = df.groupby("therapeutic_class").agg(
            mean_price=("price_inr","mean"), std_price=("price_inr","std"),
            median_price=("price_inr","median")).reset_index().sort_values("mean_price",ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=class_price["therapeutic_class"].str.title(),
            y=class_price["mean_price"],
            error_y=dict(type="data", array=class_price["std_price"], visible=True,
                         color="rgba(0,0,0,0.3)"),
            marker_color=px.colors.sequential.Blues_r[:len(class_price)],
            name="Mean Price",
            hovertemplate="<b>%{x}</b><br>Mean: ₹%{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=class_price["therapeutic_class"].str.title(),
            y=class_price["median_price"],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color=WARN),
            name="Median",
            hovertemplate="<b>%{x}</b><br>Median: ₹%{y:.2f}<extra></extra>"))
        fig.update_layout(height=460, template=TEMPLATE, margin=dict(t=20),
            yaxis_title="Price ₹",
            legend=dict(orientation="h",y=1.05,x=0.5,xanchor="center"))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Mean >> Median in every class signals heavy right-skew from specialty products. "
                "Analgesics have the lowest median (₹48) — most commoditised class.")

    # ── Tab 8: Lorenz / Concentration Curve ──────────────────────────────────
    with tabs[8]:
        section("Market Concentration Curve (Lorenz-style)")
        mfr_sorted = mfr_agg.sort_values("market_share", ascending=False).reset_index(drop=True)
        mfr_sorted["cum_share"] = mfr_sorted["market_share"].cumsum()
        mfr_sorted["pct_mfr"]   = (mfr_sorted.index+1) / len(mfr_sorted) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mfr_sorted["pct_mfr"], y=mfr_sorted["cum_share"],
            mode="lines", fill="tozeroy", fillcolor=hex_to_rgba(ACCENT,0.15),
            line=dict(color=ACCENT, width=2.5), name="Actual Concentration",
            hovertemplate="Top %{x:.1f}% → %{y:.1f}% of market<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines",
            line=dict(color="grey",dash="dash",width=1.5), name="Perfect Equality"))
        for pct in [1,5,10]:
            idx = int(len(mfr_sorted)*pct/100)
            cum = mfr_sorted.iloc[min(idx,len(mfr_sorted)-1)]["cum_share"]
            fig.add_annotation(x=pct, y=cum, text=f"Top {pct}%<br>→ {cum:.1f}%",
                showarrow=True, arrowhead=2, font=dict(size=10,color=ACCENT),
                bgcolor="rgba(255,255,255,0.8)")
        fig.update_layout(height=500, template=TEMPLATE, margin=dict(t=20),
            xaxis_title="% of Manufacturers (ranked by size)",
            yaxis_title="Cumulative Market Share (%)",
            legend=dict(x=0.7,y=0.3))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Top 1% of manufacturers control ~20% of products — "
                "the market is highly fragmented across 7,648 manufacturers.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRICE ANALYTICS  (Module 2 — all charts)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Price Analytics":
    st.title("💰 Price Analytics")
    st.caption("Price distribution, tier structure, combination premiums, ingredient variance and manufacturer strategy")

    p10,p25,p50,p75,p90,p99 = [df["price_inr"].quantile(q) for q in [.10,.25,.50,.75,.90,.99]]
    under_100  = (df["price_inr"]<100).mean()*100
    over_10k   = (df["price_inr"]>10000).mean()*100
    df_unit = df.dropna(subset=["pack_size","price_inr"]).copy()
    df_unit = df_unit[df_unit["pack_size"]>0].copy()
    df_unit["price_per_unit"] = (df_unit["price_inr"] / df_unit["pack_size"]).round(4)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi("Median Price",  p50,        "gold", "₹{:.0f}")
    with c2: kpi("Mean Price",    df["price_inr"].mean(), "gold", "₹{:.0f}")
    with c3: kpi("P90 Price",     p90,        "warn", "₹{:.0f}")
    with c4: kpi("Under ₹100",    under_100,  "ok",   "{:.1f}%")
    with c5: kpi("Over ₹10,000",  over_10k,   "warn", "{:.2f}%")

    tabs = st.tabs([
        "Distribution & ECDF","Price by Class (Violin)","Price Range Bands",
        "Combo Premium","Price Strip Plot","Outlier Explorer","Top 20 Expensive",
        "Ingredient CV","Competition vs Price","Dosage Sunburst",
        "Dosage × Tier Heatmap","Mfr Price Mix","Tier Waterfall"
    ])

    # ── Tab 0: Histogram + ECDF ───────────────────────────────────────────────
    with tabs[0]:
        section("Overall Price Distribution — Histogram + ECDF")
        df_plot = df[df["price_inr"] <= df["price_inr"].quantile(0.97)].copy()
        sorted_prices = np.sort(df["price_inr"].dropna().values)
        ecdf_y = np.arange(1, len(sorted_prices)+1) / len(sorted_prices) * 100
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Price Histogram (P97 trimmed)","Cumulative Distribution (log X)"],
            horizontal_spacing=0.1)
        fig.add_trace(go.Histogram(x=df_plot["price_inr"], nbinsx=80,
            marker_color=ACCENT, opacity=0.8,
            hovertemplate="₹%{x}<br>%{y:,} products<extra></extra>"), row=1, col=1)
        for pval,plabel,pcolor in [(p25,"P25","green"),(p50,"Median","blue"),(p75,"P75","orange"),(p90,"P90","red")]:
            fig.add_vline(x=pval, line_dash="dash", line_color=pcolor, opacity=0.7, row=1, col=1,
                annotation_text=f"{plabel} ₹{pval:.0f}", annotation_font_size=9)
        fig.add_trace(go.Scatter(x=sorted_prices, y=ecdf_y, mode="lines",
            line=dict(color=ACCENT, width=2), name="ECDF",
            hovertemplate="₹%{x:.2f}<br>%{y:.1f}% of products<extra></extra>"), row=1, col=2)
        for pct_val,price_val in [(50,p50),(75,p75),(90,p90),(99,p99)]:
            fig.add_annotation(x=price_val, y=pct_val, row=1, col=2,
                text=f"P{pct_val} ₹{price_val:.0f}", showarrow=True, arrowhead=2,
                font=dict(size=9), bgcolor="rgba(255,255,255,0.8)")
        fig.update_xaxes(title_text="Price ₹", row=1, col=1)
        fig.update_xaxes(title_text="Price ₹ (log)", row=1, col=2, type="log")
        fig.update_yaxes(title_text="Products", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative %", row=1, col=2)
        fig.update_layout(height=460, template=TEMPLATE, showlegend=False, margin=dict(t=50))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight(f"{under_100:.0f}% of all products are priced under ₹100. "
                "Mean (₹270) vs Median (₹79) gap reveals heavy right-skew from specialty drugs.")

    # ── Tab 1: Violin by Class ────────────────────────────────────────────────
    with tabs[1]:
        section("Price Distribution by Therapeutic Class (Violin)")
        classes_named = ["antibiotic","analgesic","antacid","antihistamine",
            "antidiabetic","antihypertensive","corticosteroid",
            "antidepressant","bronchodilator","diuretic"]
        df_cls = df[df["therapeutic_class"].isin(classes_named) & (df["price_inr"]<=500)].copy()
        fig = go.Figure()
        for i, cls in enumerate(classes_named):
            sub = df_cls[df_cls["therapeutic_class"]==cls]["price_inr"]
            fig.add_trace(go.Violin(x=sub, name=cls.title(), orientation="h",
                side="positive", width=1.8,
                line_color=COLORS[i%len(COLORS)], fillcolor=COLORS[i%len(COLORS)],
                opacity=0.6, meanline_visible=True, box_visible=True,
                hovertemplate=f"<b>{cls.title()}</b><br>₹%{{x:.2f}}<extra></extra>"))
        fig.update_layout(height=600, template=TEMPLATE, showlegend=False,
            violingap=0.05, violingroupgap=0,
            margin=dict(t=20,l=140), xaxis_title="Price ₹")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Percentile Bands ───────────────────────────────────────────────
    with tabs[2]:
        section("Price Range Bands by Therapeutic Class (P10–P90 + IQR + Median)")
        class_pcts = df[df["therapeutic_class"].isin(classes_named)].groupby("therapeutic_class")["price_inr"].agg(
            p10=lambda x: x.quantile(.10), p25=lambda x: x.quantile(.25),
            p50=lambda x: x.quantile(.50), p75=lambda x: x.quantile(.75),
            p90=lambda x: x.quantile(.90), mean="mean"
        ).reset_index().sort_values("p50")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=class_pcts["therapeutic_class"].str.title(),
            x=class_pcts["p90"]-class_pcts["p10"], base=class_pcts["p10"],
            orientation="h", marker_color="rgba(173,216,230,0.5)", name="P10–P90",
            hovertemplate="<b>%{y}</b><br>P10–P90<extra></extra>"))
        fig.add_trace(go.Bar(y=class_pcts["therapeutic_class"].str.title(),
            x=class_pcts["p75"]-class_pcts["p25"], base=class_pcts["p25"],
            orientation="h", marker_color=ACCENT, opacity=0.8, name="IQR P25–P75",
            hovertemplate="<b>%{y}</b><br>IQR: ₹%{base:.0f}–₹%{x:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(y=class_pcts["therapeutic_class"].str.title(),
            x=class_pcts["p50"], mode="markers",
            marker=dict(symbol="line-ew", size=16, color="white", line=dict(color="black",width=2)),
            name="Median"))
        fig.add_trace(go.Scatter(y=class_pcts["therapeutic_class"].str.title(),
            x=class_pcts["mean"], mode="markers",
            marker=dict(symbol="diamond", size=10, color=WARN), name="Mean"))
        fig.update_layout(barmode="overlay", height=480, template=TEMPLATE,
            margin=dict(t=20,l=160), xaxis_title="Price ₹",
            legend=dict(orientation="h",y=1.05,x=0.5,xanchor="center"))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Combo Premium ──────────────────────────────────────────────────
    with tabs[3]:
        section("Combination Drug Price Premium Analysis")
        premium_rows = []
        for cls in classes_named:
            sub    = df[df["therapeutic_class"]==cls]
            single = sub[sub["num_active_ingredients"]==1]["price_inr"]
            combo  = sub[sub["num_active_ingredients"]>1]["price_inr"]
            if len(single)>10 and len(combo)>10:
                s,c = single.median(), combo.median()
                premium_rows.append({"class":cls.title(),"single":s,"combo":c,
                    "premium_pct":(c-s)/s*100,"n_single":len(single),"n_combo":len(combo)})
        prem_df = pd.DataFrame(premium_rows).sort_values("premium_pct")
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Combo Premium % over Single","Median Price: Single vs Combo ₹"],
            horizontal_spacing=0.15)
        fig.add_trace(go.Bar(y=prem_df["class"], x=prem_df["premium_pct"], orientation="h",
            marker_color=[OK if v>0 else WARN for v in prem_df["premium_pct"]],
            text=[f"+{v:.0f}%" for v in prem_df["premium_pct"]], textposition="outside",
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>Premium: %{x:+.1f}%<extra></extra>"), row=1, col=1)
        fig.add_vline(x=0, line_color="black", line_width=1, row=1, col=1)
        fig.add_trace(go.Bar(name="Single", y=prem_df["class"], x=prem_df["single"],
            orientation="h", marker_color=ACCENT, opacity=0.85,
            hovertemplate="<b>%{y}</b><br>Single: ₹%{x:.1f}<extra></extra>"), row=1, col=2)
        fig.add_trace(go.Bar(name="Combo", y=prem_df["class"], x=prem_df["combo"],
            orientation="h", marker_color=GOLD, opacity=0.85,
            hovertemplate="<b>%{y}</b><br>Combo: ₹%{x:.1f}<extra></extra>"), row=1, col=2)
        fig.update_layout(barmode="group", height=500, template=TEMPLATE,
            margin=dict(t=50,l=10,r=80),
            legend=dict(orientation="h",y=1.06,x=0.75,xanchor="center"))
        fig.update_xaxes(title_text="Premium % over single", row=1, col=1)
        fig.update_xaxes(title_text="Median Price ₹", row=1, col=2)
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Diuretics (+385%) and Bronchodilators (+332%) command the highest premiums. "
                "Combination products are the primary margin-enhancement strategy across all classes.")

    # ── Tab 4: Strip Plot ─────────────────────────────────────────────────────
    with tabs[4]:
        section("Price Strip Plot — Single vs Combo Drugs by Class")
        df_strip = df[df["therapeutic_class"].isin(classes_named) & df["price_inr"].between(1,1000)].copy()
        df_strip["ingredient_type"] = df_strip["num_active_ingredients"].apply(
            lambda x: "Single" if x==1 else ("Dual" if x==2 else "3+ Combo"))
        fig = px.strip(df_strip.sample(min(6000,len(df_strip)),random_state=42),
            x="therapeutic_class", y="price_inr", color="ingredient_type",
            color_discrete_map={"Single":ACCENT,"Dual":GOLD,"3+ Combo":WARN},
            hover_name="brand_name",
            hover_data={"manufacturer":True,"price_inr":":.2f","therapeutic_class":False},
            category_orders={"therapeutic_class":classes_named},
            template=TEMPLATE)
        fig.update_traces(marker_size=3, opacity=0.5)
        fig.update_layout(height=520, yaxis_title="Price ₹", xaxis_title="",
            xaxis_tickangle=-20, legend_title="Ingredient Type", margin=dict(t=20))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Outlier Explorer ───────────────────────────────────────────────
    with tabs[5]:
        section(f"High-Cost Product Explorer (≥ P95 = ₹{p90:.0f})")
        outliers = df[df["price_inr"]>=p90].copy()
        outliers["mfr_short"] = outliers["manufacturer"].str.replace(
            r"(Pharmaceuticals?|Industries?|Laboratories?|Ltd|Limited)","",regex=True).str.strip()
        fig = px.scatter(outliers, x="price_inr", y="dosage_form",
            color="therapeutic_class", size="price_inr", size_max=40,
            hover_name="brand_name",
            hover_data={"manufacturer":True,"price_inr":":.2f",
                "primary_ingredient":True,"dosage_form":False,"therapeutic_class":False},
            log_x=True, color_discrete_sequence=COLORS, template=TEMPLATE,
            labels={"price_inr":"Price ₹ (log)","dosage_form":"Dosage Form"})
        fig.update_layout(height=560, margin=dict(t=20), legend_title="Therapeutic Class")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 6: Top 20 Expensive Table ────────────────────────────────────────
    with tabs[6]:
        section("Top 20 Highest Priced Products")
        top20_exp = df.nlargest(20,"price_inr")[
            ["brand_name","manufacturer","primary_ingredient","dosage_form","therapeutic_class","price_inr"]
        ].reset_index(drop=True)
        top20_exp.index += 1
        top20_exp["price_inr"] = top20_exp["price_inr"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(top20_exp, use_container_width=True, height=560)
        insight("All top-20 are oncology/specialty biologics — classified under 'other' therapeutic class.")

    # ── Tab 7: Ingredient CV ─────────────────────────────────────────────────
    with tabs[7]:
        section("Ingredient-Level Price Variance — Coefficient of Variation")
        ingr_cv = df.groupby("primary_ingredient")["price_inr"].agg(
            min_p="min", max_p="max", mean="mean", std="std", count="count", median="median"
        ).reset_index()
        ingr_cv = ingr_cv[ingr_cv["count"]>=50]
        ingr_cv["cv"] = (ingr_cv["std"]/ingr_cv["mean"]*100).round(2)
        ingr_cv["range_ratio"] = (ingr_cv["max_p"]/ingr_cv["min_p"].replace(0,np.nan)).round(1)
        top_cv = ingr_cv.nlargest(20,"cv")
        top_rng = ingr_cv.nlargest(15,"range_ratio").sort_values("range_ratio")
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Top 20 by CV% (Generic/Branded gap)","Min-Max Price Range Dumbbell"],
            horizontal_spacing=0.12)
        fig.add_trace(go.Bar(y=top_cv["primary_ingredient"], x=top_cv["cv"],
            orientation="h", marker_color=top_cv["cv"], marker_colorscale="RdYlGn_r",
            showlegend=False, text=top_cv["cv"].apply(lambda x:f"{x:.0f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>CV: %{x:.1f}%<extra></extra>"), row=1, col=1)
        for _, row in top_rng.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["min_p"],row["max_p"]], y=[row["primary_ingredient"],row["primary_ingredient"]],
                mode="lines+markers", line=dict(color="lightgrey",width=2),
                marker=dict(size=10, color=[OK,WARN]), showlegend=False,
                hovertemplate=f"<b>{row['primary_ingredient']}</b><br>₹{row['min_p']:.2f}–₹{row['max_p']:,.2f}<extra></extra>"),
                row=1, col=2)
        fig.update_xaxes(title_text="CV (%)", row=1, col=1)
        fig.update_xaxes(title_text="Price ₹ (log)", type="log", row=1, col=2)
        fig.update_layout(height=560, template=TEMPLATE, margin=dict(t=50,l=10))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Risperidone CV 631% and Cefoperazone 306% — "
                "massive generic/branded price gaps signal lucrative generic entry opportunities.")

    # ── Tab 8: Competition vs Price Bubble ───────────────────────────────────
    with tabs[8]:
        section("Ingredient Competition vs Price (Bubble = CV%)")
        bubble_ingr = ingr_cv[ingr_cv["count"].between(50,5000)].copy()
        bubble_ingr["therapeutic_class"] = bubble_ingr["primary_ingredient"].map(
            df.groupby("primary_ingredient")["therapeutic_class"].agg(lambda x: x.mode()[0]))
        fig = px.scatter(bubble_ingr, x="count", y="median", size="cv",
            color="therapeutic_class", hover_name="primary_ingredient",
            hover_data={"count":":","median":":.2f","cv":":.1f","range_ratio":":.1f"},
            size_max=35, color_discrete_sequence=COLORS, template=TEMPLATE,
            labels={"count":"# Products (competition)","median":"Median Price ₹","cv":"CV%"})
        fig.update_layout(height=540, margin=dict(t=20),
            xaxis_title="Number of Products (competition intensity)",
            yaxis_title="Median Price ₹", legend_title="Class")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 9: Dosage Sunburst ────────────────────────────────────────────────
    with tabs[9]:
        section("Dosage Form → Pack Unit → Price Tier Sunburst")
        df_sun = df.copy()
        df_sun["price_tier"] = pd.cut(df_sun["price_inr"],
            bins=[0,50,100,200,500,2000,float("inf")],
            labels=["₹0–50","₹50–100","₹100–200","₹200–500","₹500–2K","₹2K+"]).astype(str)
        df_sun = df_sun[(df_sun["price_tier"]!="nan") & df_sun["pack_unit"].notna()]
        sun_data = (df_sun.groupby(["dosage_form","pack_unit","price_tier"],observed=True)
            .size().reset_index(name="count"))
        sun_data = sun_data[sun_data["count"]>5]
        fig = px.sunburst(sun_data, path=["dosage_form","pack_unit","price_tier"],
            values="count", color="dosage_form",
            color_discrete_sequence=COLORS, template=TEMPLATE)
        fig.update_traces(textinfo="label+percent parent",
            hovertemplate="<b>%{label}</b><br>Products: %{value:,}<br>% parent: %{percentParent:.1%}<extra></extra>")
        fig.update_layout(height=660, margin=dict(t=20))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 10: Dosage × Price Tier Heatmap ──────────────────────────────────
    with tabs[10]:
        section("Product Count: Dosage Form × Price Tier")
        df_sun2 = df.copy()
        df_sun2["price_tier"] = pd.cut(df_sun2["price_inr"],
            bins=[0,50,100,200,500,2000,float("inf")],
            labels=["₹0–50","₹50–100","₹100–200","₹200–500","₹500–2K","₹2K+"]).astype(str)
        df_sun2 = df_sun2[df_sun2["price_tier"]!="nan"]
        heat_d = df_sun2.groupby(["dosage_form","price_tier"]).size().unstack(fill_value=0)
        heat_d = heat_d.loc[heat_d.sum(axis=1).sort_values(ascending=False).index]
        fig = go.Figure(data=go.Heatmap(
            z=heat_d.values, x=[str(c) for c in heat_d.columns],
            y=[r.title() for r in heat_d.index],
            colorscale=[[0,"#0A1829"],[0.3,"#1C4D7A"],[1,"#00B8FF"]], text=heat_d.values,
            texttemplate="%{text:,}", textfont=dict(size=11),
            hovertemplate="<b>%{y} — %{x}</b><br>Products: %{z:,}<extra></extra>"))
        fig.update_layout(height=480, template=TEMPLATE, margin=dict(t=20,l=110))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 11: Manufacturer Price Mix ────────────────────────────────────────
    with tabs[11]:
        section("Manufacturer Price Portfolio Mix (Top 15)")
        top15_names = df["manufacturer"].value_counts().head(15).index.tolist()
        df_t15m = df[df["manufacturer"].isin(top15_names)].copy()
        df_t15m["price_tier"] = pd.cut(df_t15m["price_inr"],
            bins=[0,50,100,200,500,float("inf")],
            labels=["Budget (₹0–50)","Value (₹50–100)","Mid (₹100–200)",
                    "Premium (₹200–500)","Specialty (₹500+)"]).astype(str)
        df_t15m = df_t15m[df_t15m["price_tier"]!="nan"]
        tier_mix = df_t15m.groupby(["manufacturer","price_tier"]).size().unstack(fill_value=0)
        tier_mix_pct = tier_mix.div(tier_mix.sum(axis=1),axis=0)*100
        shorten3 = lambda n: n.replace(" Pharmaceuticals","").replace(" Pharmaceutical","") \
                              .replace(" Industries","").replace(" Ltd","").strip()
        tier_mix_pct.index = [shorten3(n) for n in tier_mix_pct.index]
        tier_mix_pct = tier_mix_pct.sort_values("Specialty (₹500+)", ascending=True)
        tier_cols = [OK,"#58D68D",GOLD,"#E67E22",WARN]
        fig = go.Figure()
        for col_name, color in zip(tier_mix_pct.columns, tier_cols):
            if col_name in tier_mix_pct.columns:
                fig.add_trace(go.Bar(y=tier_mix_pct.index, x=tier_mix_pct[col_name],
                    name=str(col_name), orientation="h", marker_color=color,
                    hovertemplate=f"<b>%{{y}}</b><br>{col_name}: %{{x:.1f}}%<extra></extra>"))
        fig.update_layout(barmode="stack", height=540, template=TEMPLATE,
            margin=dict(t=20,l=160,r=20), xaxis_title="% of Product Portfolio",
            legend=dict(orientation="h",y=1.06,x=0.5,xanchor="center"))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Dr Reddy's and Emcure have the highest specialty share — "
                "premium-focused strategy. Mankind and Micro Labs dominate the budget tier.")

    # ── Tab 12: Tier Waterfall ────────────────────────────────────────────────
    with tabs[12]:
        section("Market Price Tier Distribution — Count + Cumulative")
        tiers = pd.cut(df["price_inr"],
            bins=[0,50,100,200,500,2000,float("inf")],
            labels=["₹0–50","₹50–100","₹100–200","₹200–500","₹500–2K","₹2K+"]
        ).value_counts().sort_index()
        tier_pct  = (tiers/tiers.sum()*100).round(1)
        cumulative = tier_pct.cumsum()
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Product Count by Price Tier","Cumulative Coverage"],
            horizontal_spacing=0.12)
        tier_colors_list = [OK,"#58D68D",GOLD,"#E67E22",WARN,"#922B21"]
        fig.add_trace(go.Bar(x=tiers.index.astype(str), y=tiers.values,
            marker_color=tier_colors_list,
            text=[f"{v:,}\n({p:.1f}%)" for v,p in zip(tiers.values,tier_pct.values)],
            textposition="outside", showlegend=False,
            hovertemplate="<b>%{x}</b><br>%{y:,} products<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=cumulative.index.astype(str), y=cumulative.values,
            mode="lines+markers+text",
            line=dict(color=ACCENT,width=2.5), marker=dict(size=10,color=ACCENT),
            text=[f"{v:.0f}%" for v in cumulative.values],
            textposition="top right", showlegend=False), row=1, col=2)
        fig.add_hline(y=80, line_dash="dash", line_color="orange", row=1, col=2,
            annotation_text="80% coverage", annotation_position="right")
        fig.update_yaxes(title_text="Products", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative %", row=1, col=2)
        fig.update_layout(height=460, template=TEMPLATE, margin=dict(t=50,r=80))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INGREDIENT INTELLIGENCE  (Module 3 — all charts)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Ingredient Intelligence":
    st.title("🧪 Ingredient Intelligence")
    st.caption("Co-occurrence networks, combo strategy, competitive overlap, price variance and portfolio heatmaps")

    # ── Pre-compute ingredient data ───────────────────────────────────────────
    from collections import Counter

    df_exp = df.explode("ingredient_names").rename(columns={"ingredient_names":"ingredient"})
    df_exp = df_exp[df_exp["ingredient"].notna() & (df_exp["ingredient"]!="")]

    ingr_stats = df_exp.groupby("ingredient").agg(
        total_products=("product_id","count"),
        avg_price=("price_inr","mean"),
        median_price=("price_inr","median"),
        primary_class=("therapeutic_class", lambda x: x.mode()[0]),
        manufacturers=("manufacturer","nunique"),
        solo_products=("num_active_ingredients", lambda x:(x==1).sum()),
        combo_products=("num_active_ingredients", lambda x:(x>1).sum()),
    ).reset_index()
    ingr_stats["combo_ratio"] = (ingr_stats["combo_products"]/ingr_stats["total_products"]*100).round(1)

    all_ingrs = [i for lst in df["ingredient_names"] for i in lst if isinstance(lst,list)]
    ingr_freq = Counter(all_ingrs)

    pair_counter = Counter()
    for ingrs in df["ingredient_names"]:
        if not isinstance(ingrs, list): continue
        u = list(set(ingrs))
        for i in range(len(u)):
            for j in range(i+1, len(u)):
                pair_counter[tuple(sorted([u[i],u[j]]))] += 1

    unique_classes = ingr_stats["primary_class"].unique()
    class_color_map = {c: COLORS[i%len(COLORS)] for i,c in enumerate(unique_classes)}

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Unique Ingredients", ingr_stats.shape[0], fmt="{:,}")
    with c2: kpi("Co-occurrence Pairs", len(pair_counter), fmt="{:,}")
    with c3: kpi("Top Ingredient", 0, fmt="Paracetamol")
    with c4: kpi("Always-Combo ≥100", int(((ingr_stats["combo_ratio"]==100)&(ingr_stats["total_products"]>=100)).sum()), "warn", fmt="{:,}")

    tabs = st.tabs([
        "Ingredient Treemap","Top 25 Frequency","Co-occurrence Network",
        "Top 25 Pairs","Co-occurrence Heatmap","Class Diversity Bubble",
        "Top 8 per Class","Combo Strategy Map","Exclusivity Analysis",
        "Jaccard Overlap","Portfolio Differentiation",
        "Price Variance Map","Median Price by Ingredient",
        "Portfolio Heatmap","Pricing Power Matrix"
    ])

    # ── Tab 0: Treemap ────────────────────────────────────────────────────────
    with tabs[0]:
        section("Top 30 Ingredients — Market Presence Treemap")
        top30 = ingr_stats.nlargest(30,"total_products").copy()
        fig = px.treemap(top30,
            path=[px.Constant("All Ingredients"),"primary_class","ingredient"],
            values="total_products", color="combo_ratio",
            color_continuous_scale="RdYlGn",
            hover_data={"total_products":":,","manufacturers":":,","combo_ratio":":.1f","median_price":":.2f"},
            template=TEMPLATE)
        fig.update_traces(textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Products: %{value:,}<br>Combo: %{color:.1f}%<extra></extra>")
        fig.update_layout(height=620, margin=dict(t=20),
            coloraxis_colorbar=dict(title="Combo %"))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Size = product count | Color: green = mostly solo ingredient, red = mostly used in combos.")

    # ── Tab 1: Top 25 Products + Manufacturer Count ───────────────────────────
    with tabs[1]:
        section("Top 25 Ingredients — Product Count vs Manufacturer Reach")
        top25 = ingr_stats.nlargest(25,"total_products").sort_values("total_products")
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Product Count","Manufacturer Count"],
            horizontal_spacing=0.12)
        fig.add_trace(go.Bar(y=top25["ingredient"], x=top25["total_products"],
            orientation="h", marker_color=ACCENT, showlegend=False,
            hovertemplate="<b>%{y}</b><br>Products: %{x:,}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(y=top25["ingredient"], x=top25["manufacturers"],
            orientation="h", marker_color=top25["manufacturers"],
            marker_colorscale=[[0,"#0A1829"],[0.3,"#1C4D7A"],[1,"#00B8FF"]], showlegend=False,
            hovertemplate="<b>%{y}</b><br>Manufacturers: %{x:,}<extra></extra>"), row=1, col=2)
        fig.update_xaxes(title_text="Products",      row=1, col=1)
        fig.update_xaxes(title_text="Manufacturers", row=1, col=2)
        fig.update_layout(height=680, template=TEMPLATE, margin=dict(t=50))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("High manufacturer count = commoditised ingredient. Low = differentiated/specialty play.")

    # ── Tab 2: Network Graph ──────────────────────────────────────────────────
    with tabs[2]:
        section("Ingredient Co-occurrence Network (Top 40 nodes, ≥200 products per edge)")
        TOP_N = 40; MIN_EDGE = 200
        top_ingr_names = [i for i,_ in ingr_freq.most_common(TOP_N)]
        top_pairs_net = [(pair,cnt) for pair,cnt in pair_counter.most_common()
            if cnt>=MIN_EDGE and pair[0] in top_ingr_names and pair[1] in top_ingr_names]
        degree = Counter()
        for (a,b),cnt in top_pairs_net: degree[a]+=1; degree[b]+=1
        n = len(top_ingr_names)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        top_sorted = sorted(top_ingr_names, key=lambda x: ingr_stats[ingr_stats["ingredient"]==x]["primary_class"].values[0] if len(ingr_stats[ingr_stats["ingredient"]==x])>0 else "z")
        pos = {nm: (np.cos(angles[i]), np.sin(angles[i])) for i,nm in enumerate(top_sorted)}
        class_map = ingr_stats.set_index("ingredient")["primary_class"].to_dict()
        uc2 = list(set(class_map.get(i,"other") for i in top_ingr_names))
        ccm2 = {c:COLORS[i%len(COLORS)] for i,c in enumerate(uc2)}
        max_cnt = max((cnt for _,cnt in top_pairs_net), default=1)
        edge_traces = []
        for (a,b),cnt in top_pairs_net:
            if a in pos and b in pos:
                x0,y0=pos[a]; x1,y1=pos[b]
                w = 0.5+(cnt/max_cnt)*6; op = 0.2+(cnt/max_cnt)*0.6
                edge_traces.append(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],mode="lines",
                    line=dict(width=w,color=f"rgba(100,100,200,{op:.2f})"),
                    hoverinfo="skip", showlegend=False))
        node_traces = []
        for cls in uc2:
            nodes_c = [nm for nm in top_sorted if class_map.get(nm,"other")==cls]
            if not nodes_c: continue
            node_traces.append(go.Scatter(
                x=[pos[nm][0] for nm in nodes_c], y=[pos[nm][1] for nm in nodes_c],
                mode="markers+text", name=cls.title(), text=nodes_c,
                textposition="top center", textfont=dict(size=8),
                marker=dict(size=[12+degree.get(nm,0)*3 for nm in nodes_c],
                    color=ccm2[cls], line=dict(width=1.5,color="white"), opacity=0.9),
                hovertext=[f"<b>{nm}</b><br>{cls}<br>Connections: {degree.get(nm,0)}<br>Products: {ingr_freq.get(nm,0):,}" for nm in nodes_c],
                hoverinfo="text"))
        fig = go.Figure(data=edge_traces+node_traces)
        fig.update_layout(height=720, template=TEMPLATE,
            showlegend=True, legend=dict(title="Class",x=1.01,y=0.5),
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            margin=dict(t=20,r=160))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Node size = number of connections. Paracetamol is the highest-centrality node (60 connections).")

    # ── Tab 3: Top 25 Pairs Bar ───────────────────────────────────────────────
    with tabs[3]:
        section("Top 25 Ingredient Co-occurrence Pairs")
        top25_pairs = pair_counter.most_common(25)
        pair_labels = [f"{a} + {b}" for (a,b),_ in top25_pairs]
        pair_counts = [cnt for _,cnt in top25_pairs]
        pair_classes = []
        for (a,b),_ in top25_pairs:
            ca = ingr_stats[ingr_stats["ingredient"]==a]["primary_class"].values
            pair_classes.append(ca[0] if len(ca)>0 else "other")
        fig = go.Figure(go.Bar(
            x=pair_counts[::-1], y=pair_labels[::-1], orientation="h",
            marker_color=[class_color_map.get(c,"#888") for c in pair_classes[::-1]],
            text=[f"{c:,}" for c in pair_counts[::-1]], textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:,} products<extra></extra>"))
        fig.update_layout(height=720, template=TEMPLATE,
            margin=dict(t=20,l=10,r=80), xaxis_title="Number of Products")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Aceclofenac + Paracetamol leads with 6,442 products. "
                "Amoxycillin + Clavulanic Acid follows at 5,867 — core antibiotic resistance strategy.")

    # ── Tab 4: Co-occurrence Heatmap ─────────────────────────────────────────
    with tabs[4]:
        section("Co-occurrence Heatmap — Top 15 Ingredients")
        top15_nm = [i for i,_ in ingr_freq.most_common(15)]
        matrix = pd.DataFrame(0, index=top15_nm, columns=top15_nm)
        for (a,b),cnt in pair_counter.items():
            if a in top15_nm and b in top15_nm:
                matrix.loc[a,b]=cnt; matrix.loc[b,a]=cnt
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values, x=matrix.columns.tolist(), y=matrix.index.tolist(),
            colorscale="YlOrRd",
            hovertemplate="<b>%{y} + %{x}</b><br>%{z:,} products<extra></extra>",
            text=matrix.values, texttemplate="%{text:,}", textfont=dict(size=9)))
        fig.update_layout(height=580, template=TEMPLATE,
            margin=dict(t=20,l=160,b=120), xaxis=dict(tickangle=-35))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Class Diversity Bubble ─────────────────────────────────────────
    with tabs[5]:
        section("Therapeutic Class: Ingredient Diversity vs Market Size")
        class_stats = df_exp.groupby("therapeutic_class").agg(
            unique_ingredients=("ingredient","nunique"),
            total_products=("product_id","count"),
            avg_price=("price_inr","mean"),
            manufacturers=("manufacturer","nunique")
        ).reset_index()
        class_stats["prod_per_ingr"] = (class_stats["total_products"]/class_stats["unique_ingredients"]).round(1)
        fig = px.scatter(class_stats, x="unique_ingredients", y="total_products",
            size="manufacturers", color="avg_price", text="therapeutic_class",
            color_continuous_scale="RdYlGn_r", size_max=60,
            hover_name="therapeutic_class",
            hover_data={"unique_ingredients":":,","total_products":":,",
                "manufacturers":":,","prod_per_ingr":":.1f","avg_price":":.2f"},
            template=TEMPLATE)
        fig.update_traces(textposition="top center", textfont_size=11)
        fig.update_layout(height=580, margin=dict(t=20),
            xaxis_title="Unique Ingredients", yaxis_title="Total Products",
            coloraxis_colorbar=dict(title="Avg Price ₹"))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Diuretics has only 23 unique ingredients — the least explored class. "
                "Antibiotics dominate in both ingredient count and product volume.")

    # ── Tab 6: Top 8 per Class ────────────────────────────────────────────────
    with tabs[6]:
        section("Top 8 Ingredients per Therapeutic Class")
        cls8 = ["antibiotic","analgesic","antacid","antihistamine",
                "antidiabetic","antihypertensive","antidepressant","bronchodilator"]
        fig = make_subplots(rows=4, cols=2,
            subplot_titles=[c.title() for c in cls8],
            vertical_spacing=0.07, horizontal_spacing=0.2)
        for idx, cls in enumerate(cls8):
            row=idx//2+1; col=idx%2+1
            sub = (df_exp[df_exp["therapeutic_class"]==cls]
                .groupby("ingredient").size().sort_values().tail(8))
            fig.add_trace(go.Bar(x=sub.values, y=sub.index, orientation="h",
                marker_color=COLORS[idx%len(COLORS)], showlegend=False,
                hovertemplate="<b>%{y}</b><br>%{x:,} products<extra></extra>"),
                row=row, col=col)
        fig.update_layout(height=1200, template=TEMPLATE, margin=dict(t=60))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 7: Combo Strategy Map ─────────────────────────────────────────────
    with tabs[7]:
        section("Ingredient Combo Strategy Map — Volume vs Combination Ratio")
        combo_a = ingr_stats[ingr_stats["total_products"]>=100].copy()
        med_freq = combo_a["total_products"].median()
        med_combo = combo_a["combo_ratio"].median()
        def quadrant(row):
            hi_v = row["total_products"]>=med_freq; hi_c = row["combo_ratio"]>=med_combo
            if hi_v and hi_c: return "High Volume Combo"
            elif hi_v: return "High Volume Solo"
            elif hi_c: return "Niche Combo"
            else: return "Niche Solo"
        combo_a["segment"] = combo_a.apply(quadrant, axis=1)
        seg_colors = {"High Volume Combo":WARN,"High Volume Solo":ACCENT,
                      "Niche Combo":GOLD,"Niche Solo":OK}
        fig = px.scatter(combo_a, x="total_products", y="combo_ratio",
            color="segment", color_discrete_map=seg_colors,
            size="manufacturers", size_max=30, hover_name="ingredient",
            hover_data={"total_products":":,","combo_ratio":":.1f","manufacturers":":,","primary_class":True,"segment":False},
            log_x=True, template=TEMPLATE)
        fig.add_vline(x=med_freq, line_dash="dash", line_color="grey", opacity=0.5)
        fig.add_hline(y=med_combo, line_dash="dash", line_color="grey", opacity=0.5)
        highlight = ["Paracetamol","Clavulanic Acid","Ornidazole","Azithromycin",
                     "Domperidone","Metformin","Montelukast","Sulbactam"]
        for _, row in combo_a[combo_a["ingredient"].isin(highlight)].iterrows():
            fig.add_annotation(x=row["total_products"], y=row["combo_ratio"],
                text=row["ingredient"], showarrow=True, arrowhead=2,
                font=dict(size=9), bgcolor="rgba(255,255,255,0.8)")
        fig.update_layout(height=580, margin=dict(t=20),
            xaxis_title="Total Products (log)", yaxis_title="Combination Ratio (%)",
            legend_title="Segment")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 8: Exclusivity ────────────────────────────────────────────────────
    with tabs[8]:
        section("Ingredient Exclusivity — Always Combo vs Always Solo")
        always_combo = ingr_stats[(ingr_stats["combo_ratio"]==100)&(ingr_stats["total_products"]>=100)] \
            .nlargest(20,"total_products")
        always_solo  = ingr_stats[(ingr_stats["combo_ratio"]==0)&(ingr_stats["total_products"]>=50)] \
            .nlargest(20,"total_products")
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=[f"Always in Combos — {len(always_combo)} ingredients",
                            f"Always Solo — {len(always_solo)} ingredients"],
            horizontal_spacing=0.15)
        fig.add_trace(go.Bar(y=always_combo["ingredient"], x=always_combo["total_products"],
            orientation="h", marker_color=WARN, showlegend=False,
            hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(y=always_solo["ingredient"], x=always_solo["total_products"],
            orientation="h", marker_color=OK, showlegend=False,
            hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>"), row=1, col=2)
        fig.update_xaxes(title_text="Total Products", row=1, col=1)
        fig.update_xaxes(title_text="Total Products", row=1, col=2)
        fig.update_layout(height=560, template=TEMPLATE, margin=dict(t=50))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Clavulanic Acid (8,525 products, 100% combo) is always paired with an antibiotic — "
                "cannot be marketed standalone. These are pure combo revenue plays.")

    # ── Tab 9: Jaccard Overlap ────────────────────────────────────────────────
    with tabs[9]:
        section("Manufacturer Ingredient Overlap — Jaccard Similarity (Top 15)")
        top15_mfr = df["manufacturer"].value_counts().head(15).index.tolist()
        mfr_ingr_sets = {}
        for mfr_nm in top15_mfr:
            ingrs = set()
            for lst in df[df["manufacturer"]==mfr_nm]["ingredient_names"]:
                if isinstance(lst, list): ingrs.update(lst)
            mfr_ingr_sets[mfr_nm] = ingrs
        sh = lambda n: n.replace(" Pharmaceuticals","").replace(" Pharmaceutical","") \
                        .replace(" Industries","").replace(" Ltd","").replace(" Limited","").strip()
        short_mfr = [sh(m) for m in top15_mfr]
        jac = pd.DataFrame(index=short_mfr, columns=short_mfr, dtype=float)
        for i,mi in enumerate(top15_mfr):
            for j,mj in enumerate(top15_mfr):
                a,b = mfr_ingr_sets[mi], mfr_ingr_sets[mj]
                jac.iloc[i,j] = len(a&b)/len(a|b) if (a|b) else 0
        fig = go.Figure(data=go.Heatmap(
            z=jac.values.astype(float), x=short_mfr, y=short_mfr,
            colorscale="RdYlGn_r", zmin=0, zmax=1,
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Jaccard: %{z:.2f}<extra></extra>",
            text=np.round(jac.values.astype(float),2),
            texttemplate="%{text:.2f}", textfont=dict(size=10)))
        fig.update_layout(height=600, template=TEMPLATE,
            margin=dict(t=20,l=160,b=130), xaxis=dict(tickangle=-40))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Sun Pharma ↔ Intas have the highest overlap (0.56 Jaccard) — most direct ingredient competitors. "
                "Most manufacturers share 40–55% of their portfolios — minimal differentiation.")

    # ── Tab 10: Portfolio Differentiation ─────────────────────────────────────
    with tabs[10]:
        section("Ingredient Portfolio Differentiation Score")
        uniq_rows = []
        for mfr_nm in top15_mfr:
            mfr_set = mfr_ingr_sets[mfr_nm]
            others  = set(i for m,s in mfr_ingr_sets.items() if m!=mfr_nm for i in s)
            unique  = mfr_set - others
            shared  = mfr_set & others
            uniq_rows.append({"manufacturer":sh(mfr_nm),
                "total":len(mfr_set),"unique":len(unique),"shared":len(shared),
                "pct_unique":len(unique)/len(mfr_set)*100 if mfr_set else 0})
        uniq_df = pd.DataFrame(uniq_rows).sort_values("pct_unique")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=uniq_df["manufacturer"], x=uniq_df["shared"],
            name="Shared with Others", orientation="h", marker_color="#1C4D7A",
            hovertemplate="<b>%{y}</b><br>Shared: %{x:,}<extra></extra>"))
        fig.add_trace(go.Bar(y=uniq_df["manufacturer"], x=uniq_df["unique"],
            name="Unique to Manufacturer", orientation="h", marker_color=ACCENT,
            hovertemplate="<b>%{y}</b><br>Unique: %{x:,}<extra></extra>"))
        for _, row in uniq_df.iterrows():
            fig.add_annotation(x=row["total"]+5, y=row["manufacturer"],
                text=f"{row['pct_unique']:.1f}% unique", showarrow=False,
                font=dict(size=9, color="#555"))
        fig.update_layout(barmode="stack", height=520, template=TEMPLATE,
            margin=dict(t=20,r=140),
            xaxis_title="Number of Ingredients",
            legend=dict(orientation="h",y=1.05,x=0.5,xanchor="center"))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 11: Price Variance Map ────────────────────────────────────────────
    with tabs[11]:
        section("Ingredient Price Variance Map (CV% vs Competition)")
        ingr_price = df_exp.groupby("ingredient")["price_inr"].agg(
            min_p="min",max_p="max",median="median",std="std",count="count").reset_index()
        ingr_price = ingr_price[ingr_price["count"]>=30]
        ingr_price["cv"]          = (ingr_price["std"]/ingr_price["median"]*100).round(1)
        ingr_price["range_ratio"] = (ingr_price["max_p"]/ingr_price["min_p"].replace(0,np.nan)).round(0)
        ingr_price = ingr_price.merge(
            ingr_stats[["ingredient","primary_class","manufacturers"]], on="ingredient", how="left")
        fig = px.scatter(ingr_price[ingr_price["cv"]<500],
            x="count", y="cv", color="primary_class", size="range_ratio",
            size_max=35, hover_name="ingredient",
            hover_data={"count":":,","cv":":.1f","min_p":":.2f","max_p":":,.2f","range_ratio":":.0f","manufacturers":":,"},
            log_x=True, color_discrete_sequence=COLORS, template=TEMPLATE)
        fig.add_hline(y=100, line_dash="dash", line_color="orange", opacity=0.6,
            annotation_text="CV=100% (high variance)", annotation_position="right")
        fig.update_layout(height=560, margin=dict(t=20),
            xaxis_title="Products (log)", yaxis_title="Price CV%", legend_title="Class")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 12: Median Price Bar ──────────────────────────────────────────────
    with tabs[12]:
        section("Median Price by Ingredient (Top 30 Most Common)")
        top30_p = ingr_stats.nlargest(30,"total_products").sort_values("median_price")
        fig = go.Figure(go.Bar(
            y=top30_p["ingredient"], x=top30_p["median_price"], orientation="h",
            marker_color=[class_color_map.get(c,"#888") for c in top30_p["primary_class"]],
            text=top30_p["median_price"].apply(lambda x:f"₹{x:.1f}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Median: ₹%{x:.2f}<extra></extra>"))
        fig.update_layout(height=700, template=TEMPLATE, margin=dict(t=20,l=10,r=80),
            xaxis_title="Median Product Price ₹", showlegend=False)
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 13: Portfolio Heatmap ─────────────────────────────────────────────
    with tabs[13]:
        section("Manufacturer × Ingredient Portfolio Heatmap (Top 10 × Top 25)")
        top10_mfr2 = df["manufacturer"].value_counts().head(10).index.tolist()
        top25_ingr2 = [i for i,_ in ingr_freq.most_common(25)]
        portfolio = (df_exp[df_exp["manufacturer"].isin(top10_mfr2) &
                            df_exp["ingredient"].isin(top25_ingr2)]
            .groupby(["manufacturer","ingredient"]).size().unstack(fill_value=0))
        port_norm = portfolio.div(portfolio.max(axis=1), axis=0)*100
        sh2 = lambda n: n.replace(" Pharmaceuticals","").replace(" Pharmaceutical","") \
                         .replace(" Industries","").replace(" Ltd","").strip()
        short_m = [sh2(m) for m in portfolio.index]
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=["Raw Product Count","Normalised (% of max per manufacturer)"],
            horizontal_spacing=0.08)
        fig.add_trace(go.Heatmap(z=portfolio.values, x=portfolio.columns.tolist(), y=short_m,
            colorscale=[[0,"#0A1829"],[0.3,"#1C4D7A"],[1,"#00B8FF"]], colorbar=dict(x=0.45,len=0.8),
            hovertemplate="<b>%{y}</b> — %{x}<br>%{z:,}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=port_norm.values.round(1), x=port_norm.columns.tolist(), y=short_m,
            colorscale="YlOrRd", colorbar=dict(x=1.01,len=0.8),
            hovertemplate="<b>%{y}</b> — %{x}<br>%{z:.1f}%<extra></extra>"), row=1, col=2)
        fig.update_layout(height=460, template=TEMPLATE, margin=dict(t=50,b=120,l=160))
        for c in [1,2]: fig.update_xaxes(tickangle=-40, row=1, col=c)
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Right panel normalised — reveals strategic emphasis regardless of portfolio size. "
                "Paracetamol and Amoxycillin are universal anchor ingredients across all top manufacturers.")

    # ── Tab 14: Pricing Power & Competitive Moat Matrix ──────────────────────
    with tabs[14]:
        section("Pricing Power & Competitive Moat Matrix")

        # Build enriched dataset: manufacturer count vs price CV for each ingredient
        ingr_moat = df_exp.groupby("ingredient").agg(
            total_products  = ("product_id",   "count"),
            manufacturers   = ("manufacturer", "nunique"),
            median_price    = ("price_inr",    "median"),
            std_price       = ("price_inr",    "std"),
            min_price       = ("price_inr",    "min"),
            max_price       = ("price_inr",    "max"),
            combo_ratio     = ("num_active_ingredients", lambda x: (x>1).mean()*100),
            disc_rate       = ("is_discontinued", "mean"),
        ).reset_index()

        ingr_moat = ingr_moat[ingr_moat["total_products"] >= 50].copy()
        ingr_moat["cv"]          = (ingr_moat["std_price"] / ingr_moat["median_price"].replace(0,np.nan) * 100).round(1)
        ingr_moat["range_ratio"] = (ingr_moat["max_price"] / ingr_moat["min_price"].replace(0,np.nan)).round(1)
        ingr_moat = ingr_moat[ingr_moat["cv"].notna() & (ingr_moat["cv"] < 600)].copy()
        ingr_moat = ingr_moat.merge(
            ingr_stats[["ingredient","primary_class"]], on="ingredient", how="left")

        # Quadrant thresholds (medians)
        med_mfr = ingr_moat["manufacturers"].median()
        med_cv  = ingr_moat["cv"].median()

        def moat_quadrant(row):
            hi_mfr = row["manufacturers"] >= med_mfr
            hi_cv  = row["cv"]            >= med_cv
            if   not hi_mfr and hi_cv:  return "🏰 Pricing Moat"
            elif not hi_mfr and not hi_cv: return "💎 Niche Stable"
            elif hi_mfr and hi_cv:       return "⚡ Volatile Commodity"
            else:                        return "📦 True Commodity"

        ingr_moat["quadrant"] = ingr_moat.apply(moat_quadrant, axis=1)

        QUAD_COLORS = {
            "🏰 Pricing Moat":       C_ACCENT,    # green  — high moat
            "💎 Niche Stable":       C_ACCENT2,   # blue   — protected niche
            "⚡ Volatile Commodity": C_WARN,      # red    — high competition + chaos
            "📦 True Commodity":     C_TEXT3,     # muted  — commoditised
        }

        # ── KPI row ───────────────────────────────────────────────────────────
        q_counts = ingr_moat["quadrant"].value_counts()
        ck1,ck2,ck3,ck4 = st.columns(4)
        for col_obj, (qname, style) in zip([ck1,ck2,ck3,ck4],[
            ("🏰 Pricing Moat",       "ok"),
            ("💎 Niche Stable",       "default"),
            ("⚡ Volatile Commodity", "warn"),
            ("📦 True Commodity",     "gold"),
        ]):
            with col_obj:
                kpi(qname, q_counts.get(qname, 0), style, "{:,}")

        # ── Scatter ───────────────────────────────────────────────────────────
        fig_moat = go.Figure()

        # Quadrant shading
        x_range = [ingr_moat["manufacturers"].min()*0.8, ingr_moat["manufacturers"].max()*1.2]
        y_range = [ingr_moat["cv"].min()*0.8, ingr_moat["cv"].max()*1.1]

        # Shade quadrants
        shade_configs = [
            (x_range[0], med_mfr, med_cv, y_range[1], C_ACCENT,   0.04, "🏰 Pricing Moat"),
            (x_range[0], med_mfr, y_range[0], med_cv,  C_ACCENT2,  0.04, "💎 Niche Stable"),
            (med_mfr, x_range[1], med_cv, y_range[1],  C_WARN,     0.04, "⚡ Volatile Commodity"),
            (med_mfr, x_range[1], y_range[0], med_cv,  C_TEXT3,    0.03, "📦 True Commodity"),
        ]
        for x0, x1, y0, y1, color, alpha, label in shade_configs:
            r,g,b = [int(color.lstrip("#")[i:i+2],16) for i in (0,2,4)]
            fig_moat.add_shape(type="rect",
                x0=x0, x1=x1, y0=y0, y1=y1,
                fillcolor=f"rgba({r},{g},{b},{alpha})",
                line_width=0, layer="below")

        # Divider lines
        fig_moat.add_vline(x=med_mfr, line_dash="dot", line_color=C_BORDER,
            line_width=1.5, opacity=0.8)
        fig_moat.add_hline(y=med_cv,  line_dash="dot", line_color=C_BORDER,
            line_width=1.5, opacity=0.8)

        # Plot each quadrant as a trace for legend
        for quad, color in QUAD_COLORS.items():
            sub = ingr_moat[ingr_moat["quadrant"] == quad]
            if sub.empty: continue
            fig_moat.add_trace(go.Scatter(
                x=sub["manufacturers"], y=sub["cv"],
                mode="markers",
                name=quad,
                marker=dict(
                    size=np.sqrt(sub["total_products"]).clip(5, 28),
                    color=color,
                    opacity=0.75,
                    line=dict(width=0.8, color=C_BG),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Manufacturers: %{x:,}<br>"
                    "Price CV: %{y:.1f}%<br>"
                    "Products: %{customdata[1]:,}<br>"
                    "Median ₹: %{customdata[2]:.0f}<br>"
                    "Range: ₹%{customdata[3]:.0f}–₹%{customdata[4]:,.0f}<br>"
                    "Class: %{customdata[5]}<extra></extra>"
                ),
                customdata=np.column_stack([
                    sub["ingredient"],
                    sub["total_products"],
                    sub["median_price"],
                    sub["min_price"],
                    sub["max_price"],
                    sub["primary_class"].fillna("other"),
                ]),
            ))

        # Label top 12 most interesting ingredients (high moat + highest CV moat)
        top_label = pd.concat([
            ingr_moat[ingr_moat["quadrant"]=="🏰 Pricing Moat"].nlargest(6, "cv"),
            ingr_moat[ingr_moat["quadrant"]=="💎 Niche Stable"].nlargest(3, "median_price"),
            ingr_moat[ingr_moat["quadrant"]=="⚡ Volatile Commodity"].nlargest(3, "total_products"),
        ]).drop_duplicates("ingredient")

        for _, row in top_label.iterrows():
            color = QUAD_COLORS.get(row["quadrant"], C_TEXT2)
            fig_moat.add_annotation(
                x=row["manufacturers"], y=row["cv"],
                text=row["ingredient"][:18],
                showarrow=True, arrowhead=0, arrowwidth=1,
                arrowcolor=color,
                ax=20 if row["manufacturers"] < med_mfr else -20,
                ay=-18,
                font=dict(size=8.5, color=color, family="JetBrains Mono"),
                bgcolor="rgba(8,13,20,0.75)",
                bordercolor=color, borderwidth=1,
                borderpad=3,
            )

        # Quadrant labels
        label_positions = [
            (x_range[0]*1.05, y_range[1]*0.96, "🏰 PRICING MOAT",       C_ACCENT,  "left"),
            (x_range[0]*1.05, y_range[0]*1.08, "💎 NICHE STABLE",        C_ACCENT2, "left"),
            (x_range[1]*0.96, y_range[1]*0.96, "⚡ VOLATILE COMMODITY",  C_WARN,    "right"),
            (x_range[1]*0.96, y_range[0]*1.08, "📦 TRUE COMMODITY",      C_TEXT3,   "right"),
        ]
        for lx, ly, ltext, lcolor, anchor in label_positions:
            fig_moat.add_annotation(
                x=lx, y=ly, text=ltext, showarrow=False,
                font=dict(size=9, color=lcolor, family="Outfit"),
                xanchor=anchor, opacity=0.6,
            )

        _apply_plotly_theme(fig_moat, height=620)
        fig_moat.update_layout(
            xaxis=dict(title="Number of Manufacturers (competition intensity)",
                       showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title="Price CV% (pricing power / variance)",
                       showgrid=True, gridcolor=C_BORDER),
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center",
                        font=dict(size=11)),
            hovermode="closest",
            margin=dict(t=30, b=50, l=70, r=30),
        )
        st.plotly_chart(fig_moat, use_container_width=True)

        # ── Quadrant breakdown table ───────────────────────────────────────────
        section("Quadrant Breakdown — Top Ingredients per Segment")
        tcol1, tcol2 = st.columns(2)
        for col_obj, (quad, color) in zip([tcol1, tcol2], [
            ("🏰 Pricing Moat",   C_ACCENT),
            ("💎 Niche Stable",   C_ACCENT2),
        ]):
            sub = ingr_moat[ingr_moat["quadrant"]==quad].nlargest(8, "cv")[
                ["ingredient","manufacturers","median_price","cv","total_products","primary_class"]
            ].rename(columns={
                "ingredient":"Ingredient","manufacturers":"Mfr #",
                "median_price":"Med ₹","cv":"CV%",
                "total_products":"Products","primary_class":"Class"
            }).reset_index(drop=True)
            sub["Med ₹"] = sub["Med ₹"].round(0).astype(int)
            sub["CV%"]   = sub["CV%"].round(1)
            with col_obj:
                st.markdown(f"<div style='font-size:0.8rem;font-weight:700;"
                    f"color:{color};margin-bottom:6px;'>{quad}</div>",
                    unsafe_allow_html=True)
                st.dataframe(sub, use_container_width=True,
                    hide_index=True, height=280)

        tcol3, tcol4 = st.columns(2)
        for col_obj, (quad, color) in zip([tcol3, tcol4], [
            ("⚡ Volatile Commodity", C_WARN),
            ("📦 True Commodity",     C_TEXT2),
        ]):
            sub = ingr_moat[ingr_moat["quadrant"]==quad].nlargest(8, "total_products")[
                ["ingredient","manufacturers","median_price","cv","total_products","primary_class"]
            ].rename(columns={
                "ingredient":"Ingredient","manufacturers":"Mfr #",
                "median_price":"Med ₹","cv":"CV%",
                "total_products":"Products","primary_class":"Class"
            }).reset_index(drop=True)
            sub["Med ₹"] = sub["Med ₹"].round(0).astype(int)
            sub["CV%"]   = sub["CV%"].round(1)
            with col_obj:
                st.markdown(f"<div style='font-size:0.8rem;font-weight:700;"
                    f"color:{color};margin-bottom:6px;'>{quad}</div>",
                    unsafe_allow_html=True)
                st.dataframe(sub, use_container_width=True,
                    hide_index=True, height=280)

        insight(
            "X-axis = manufacturer count (competition). Y-axis = price CV% (variance = pricing power). "
            "Bubble size = product count. "
            "<b style='color:" + C_ACCENT  + "'>Pricing Moat</b>: few competitors, high price variance — strong margins. "
            "<b style='color:" + C_ACCENT2 + "'>Niche Stable</b>: few competitors, stable price — protected niche. "
            "<b style='color:" + C_WARN    + "'>Volatile Commodity</b>: many competitors, high variance — price war. "
            "<b style='color:" + C_TEXT3   + "'>True Commodity</b>: many competitors, low variance — race to the bottom."
        )



elif page == "🤖 ML: Price Prediction":
    st.title("🤖 ML: Price Prediction")
    st.caption("Random Forest Regressor | 200 trees | Log-transformed target | 80/20 split")

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("R² (log space)", M["r2_log"], "ok")
    with c2: kpi("MAE", M["mae"], "gold", "₹{:.0f}")
    with c3: kpi("Accuracy ±25%", M["acc_25"], "ok", "{:.1f}%")
    with c4: kpi("Accuracy ±50%", M["acc_50"], "ok", "{:.1f}%")

    c5,c6,c7,c8 = st.columns(4)
    with c5: kpi("R² (raw)", M["r2_raw"])
    with c6: kpi("RMSE", M["rmse"], "warn", "₹{:.0f}")
    with c7: kpi("Accuracy ±10%", M["acc_10"], "warn", "{:.1f}%")
    with c8: kpi("Precision (tier,wt)", M["prec_tier_w"], "purple")

    tabs = st.tabs(["Feature Importance", "Actual vs Predicted", "Residuals"])

    with tabs[0]:
        section("Feature Importance")
        fi = M["price_fi"].reset_index()
        fi.columns = ["feature","importance"]
        fl = {"ingr_median_price":"Ingredient Median Price","mfr_median_price":"Manufacturer Median Price",
              "pack_size_filled":"Pack Size","ingr_competition":"Ingredient Competition",
              "dosage_form_enc":"Dosage Form","therapeutic_class_enc":"Therapeutic Class",
              "pack_unit_enc":"Pack Unit","num_active_ingredients":"Num Active Ingredients",
              "is_combo":"Is Combination Drug","mfr_product_count":"Mfr Portfolio Size"}
        fi["feature"] = fi["feature"].map(fl).fillna(fi["feature"])
        fi = fi.sort_values("importance")
        fig = go.Figure(go.Bar(
            y=fi["feature"],x=fi["importance"],orientation="h",
            marker_color=[WARN if v>0.5 else GOLD if v>0.1 else ACCENT for v in fi["importance"]],
            text=[f"{v*100:.1f}%" for v in fi["importance"]],textposition="outside"
        ))
        fig.update_layout(height=420,template=TEMPLATE,margin=dict(t=20,r=80),
                          xaxis_title="Importance (Gini)")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Ingredient median price accounts for 84% of importance — commodity pricing of the active ingredient "
                "is overwhelmingly the strongest price signal.")

    with tabs[1]:
        section("Actual vs Predicted (sample ≤ ₹1,500)")
        res = M["res_df"]
        res_filt = res[(res["actual"]>0)&(res["actual"]<1500)].sample(min(5000,len(res)),random_state=42)
        fig = px.scatter(res_filt,x="actual",y="predicted",color="therapeutic_class",
            opacity=0.4,color_discrete_sequence=COLORS,template=TEMPLATE,
            labels={"actual":"Actual ₹","predicted":"Predicted ₹","therapeutic_class":"Class"})
        fig.update_traces(marker_size=4)
        mx=res_filt[["actual","predicted"]].max().max()
        fig.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode="lines",
            line=dict(color="black",dash="dash",width=2),name="Perfect"))
        fig.update_layout(height=500,margin=dict(t=20))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        section("Residual Analysis")
        res = M["res_df"]
        res_p = res[res["actual"]<2000].copy()
        sample = res_p.sample(min(4000,len(res_p)),random_state=42)
        fig = make_subplots(rows=1,cols=2,subplot_titles=["Residuals vs Predicted","Residual Distribution"])
        fig.add_trace(go.Scatter(x=sample["predicted"],y=sample["residual"],mode="markers",
            marker=dict(size=3,color=ACCENT,opacity=0.3),showlegend=False),row=1,col=1)
        fig.add_hline(y=0,line_dash="dash",line_color=WARN,row=1,col=1)
        fig.add_trace(go.Histogram(x=res_p["residual"].clip(-500,500),nbinsx=60,
            marker_color=ACCENT,opacity=0.8,showlegend=False),row=1,col=2)
        fig.add_vline(x=0,line_dash="dash",line_color=WARN,row=1,col=2)
        fig.update_layout(height=420,template=TEMPLATE,margin=dict(t=40))
        fig.update_xaxes(title_text="Predicted ₹",row=1,col=1)
        fig.update_yaxes(title_text="Residual ₹",row=1,col=1)
        fig.update_xaxes(title_text="Residual ₹",row=1,col=2)
        _apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DISCONTINUATION RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ ML: Discontinuation Risk":
    st.title("⚠️ ML: Discontinuation Risk Classifier")
    st.caption("Random Forest Classifier | Oversampled minority class | 80/20 split | 3.1% positive rate")

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("ROC-AUC", M["auc_d"], "ok")
    with c2: kpi("Accuracy", M["acc_d"])
    with c3: kpi("Precision (disc)", M["prec_disc"], "warn")
    with c4: kpi("Recall (disc)", M["rec_disc"], "ok")

    c5,c6,c7,c8 = st.columns(4)
    with c5: kpi("Prec (active)", M["report_d"]["Active"]["precision"], "ok")
    with c6: kpi("Recall (active)", M["report_d"]["Active"]["recall"])
    with c7: kpi("Precision (macro)", M["prec_mac"])
    with c8: kpi("Recall (macro)", M["rec_mac"])

    tabs = st.tabs(["ROC & PR Curves", "Confusion Matrix", "Feature Importance", "High-Risk Products"])

    with tabs[0]:
        section("ROC & Precision-Recall Curves")
        fpr,tpr = M["fpr"],M["tpr"]; pr_p,pr_r=M["prec_pr"],M["rec_pr"]
        fig=make_subplots(rows=1,cols=2,subplot_titles=[
            f"ROC Curve (AUC={M['auc_d']:.4f})","Precision-Recall Curve"])
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"AUC={M['auc_d']:.3f}",
            line=dict(color=ACCENT,width=2.5),fill="tozeroy",fillcolor=hex_to_rgba(ACCENT,0.15)),row=1,col=1)
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color="grey",dash="dash"),
            name="Random"),row=1,col=1)
        j=tpr-fpr; oi=np.argmax(j)
        fig.add_trace(go.Scatter(x=[fpr[oi]],y=[tpr[oi]],mode="markers",
            marker=dict(size=12,color=WARN),name="Optimal"),row=1,col=1)
        fig.add_trace(go.Scatter(x=pr_r,y=pr_p,mode="lines",name="P-R Curve",
            line=dict(color=OK,width=2.5),fill="tozeroy",fillcolor=hex_to_rgba(OK,0.15)),row=1,col=2)
        bl=df["is_discontinued"].mean()
        fig.add_hline(y=bl,line_dash="dash",line_color="grey",row=1,col=2,
            annotation_text=f"Baseline ({bl:.3f})")
        fig.update_layout(height=440,template=TEMPLATE,margin=dict(t=40))
        fig.update_xaxes(title_text="FPR",row=1,col=1); fig.update_yaxes(title_text="TPR",row=1,col=1)
        fig.update_xaxes(title_text="Recall",row=1,col=2); fig.update_yaxes(title_text="Precision",row=1,col=2)
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)

    with tabs[1]:
        section("Confusion Matrix & Score Distribution")
        cm=M["cm_d"]; cm_pct=cm/cm.sum(axis=1,keepdims=True)*100
        tn,fp,fn,tp=cm.ravel()
        col1,col2=st.columns(2)
        with col1:
            fig=go.Figure(go.Heatmap(
                z=cm,x=["Pred: Active","Pred: Disc"],y=["True: Active","True: Disc"],
                colorscale=[[0,"#0A1829"],[0.4,"#0E3A6E"],[1,"#00B8FF"]],showscale=False,
                text=[[f"{cm[i,j]:,}<br>({cm_pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)],
                texttemplate="%{text}",textfont=dict(size=14),hoverinfo="skip"
            ))
            fig.update_layout(height=320,template=TEMPLATE,margin=dict(t=20))
            _apply_plotly_theme(fig)
            st.plotly_chart(fig,use_container_width=True)
            st.markdown(f"TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
        with col2:
            section("Per-Class Precision, Recall & F1")
            classes=["Active","Discontinued"]
            pr_v=[M["report_d"]["Active"]["precision"],M["report_d"]["Discontinued"]["precision"]]
            re_v=[M["report_d"]["Active"]["recall"],M["report_d"]["Discontinued"]["recall"]]
            f1_v=[M["report_d"]["Active"]["f1-score"],M["report_d"]["Discontinued"]["f1-score"]]
            fig=go.Figure()
            for name,vals,color in [("Precision",pr_v,ACCENT),("Recall",re_v,OK),("F1",f1_v,GOLD)]:
                fig.add_trace(go.Bar(x=classes,y=vals,name=name,marker_color=color,
                    text=[f"{v:.4f}" for v in vals],textposition="outside"))
            fig.update_layout(barmode="group",height=320,template=TEMPLATE,
                margin=dict(t=20),yaxis_range=[0,1.15],
                legend=dict(orientation="h",y=1.05))
            _apply_plotly_theme(fig)
            st.plotly_chart(fig,use_container_width=True)

    with tabs[2]:
        section("Feature Importance")
        dl={"mfr_product_count":"Mfr Portfolio Size","mfr_median_price":"Mfr Median Price",
            "ingr_competition":"Ingredient Competition","price_inr":"Product Price",
            "ingr_median_price":"Ingr Median Price","pack_unit_enc":"Pack Unit",
            "dosage_form_enc":"Dosage Form","pack_size_filled":"Pack Size",
            "therapeutic_class_enc":"Therapeutic Class","is_combo":"Is Combo",
            "num_active_ingredients":"Num Ingredients"}
        fi=M["disc_fi"].reset_index(); fi.columns=["feature","importance"]
        fi["feature"]=fi["feature"].map(dl).fillna(fi["feature"])
        fi=fi.sort_values("importance")
        fig=go.Figure(go.Bar(y=fi["feature"],x=fi["importance"],orientation="h",
            marker_color=[WARN if v>0.3 else GOLD if v>0.1 else ACCENT for v in fi["importance"]],
            text=[f"{v*100:.1f}%" for v in fi["importance"]],textposition="outside"))
        fig.update_layout(height=420,template=TEMPLATE,margin=dict(t=20,r=80),
            xaxis_title="Importance (Gini)")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)
        insight("Manufacturer portfolio size (57%) is the #1 predictor — larger manufacturers "
                "discontinue products more systematically as part of portfolio rationalisation.")

    with tabs[3]:
        section("High-Risk Active Products (Risk Score > 0.5)")
        hr=df[(df["disc_risk_score"]>0.5)&(df["is_discontinued"]==False)].sort_values(
            "disc_risk_score",ascending=False).head(100)[[
            "brand_name","manufacturer","therapeutic_class","dosage_form",
            "price_inr","num_active_ingredients","disc_risk_score"
        ]].reset_index(drop=True)
        hr.index+=1; hr["disc_risk_score"]=hr["disc_risk_score"].round(3)
        st.dataframe(hr, use_container_width=True, height=480,
                     column_config={"disc_risk_score":st.column_config.ProgressColumn(
                         "Risk Score",min_value=0,max_value=1,format="%.3f")})
        st.caption(f"{len(hr)} products currently active but predicted at high discontinuation risk")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 ML: Market Segmentation":
    st.title("🏭 ML: Manufacturer Segmentation (K-Means)")
    st.caption("K-Means K=5 | StandardScaler | PCA 2D visualisation")

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Silhouette Score", M["sil"], "ok")
    with c2: kpi("Variance Explained", M["var_exp"]*100, "ok", "{:.1f}%")
    with c3: kpi("Inertia (K=5)", M["inertia"], "gold", "{:,.0f}")
    with c4: kpi("Manufacturers", len(M["mfr_profile"]), "default", "{:,}")

    CMAP={"Mass Market Generics":ACCENT,"Premium Specialists":GOLD,
          "Mid-Tier Diversified":OK,"Market Leaders":WARN,"Ultra-Premium Niche":"#9B59B6"}

    tabs = st.tabs(["PCA Scatter", "Elbow Curve", "Cluster Profiles", "Radar Chart"])

    with tabs[0]:
        section("PCA 2D — Manufacturer Clusters")
        mp=M["mfr_profile"]
        fig=px.scatter(mp,x="pca_x",y="pca_y",color="cluster_label",
            hover_name="short_name",size="product_count",size_max=25,
            color_discrete_map=CMAP,template=TEMPLATE,
            hover_data={"median_price":":.0f","product_count":True,"combo_ratio":":.2f"},
            labels={"pca_x":"PCA 1","pca_y":"PCA 2","cluster_label":"Cluster"})
        fig.update_traces(marker=dict(opacity=0.65))
        fig.update_layout(height=560,margin=dict(t=20))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)

    with tabs[1]:
        section("Elbow Method — Inertia vs K")
        fig=go.Figure(go.Scatter(
            x=list(range(2,11)),y=M["inertias"],mode="lines+markers",
            line=dict(color=ACCENT,width=2.5),
            marker=dict(size=9,color=[WARN if k==5 else ACCENT for k in range(2,11)]),
            hovertemplate="K=%{x}<br>Inertia: %{y:,.0f}<extra></extra>"
        ))
        fig.add_vline(x=5,line_dash="dash",line_color=WARN,annotation_text="K=5 selected")
        fig.update_layout(height=400,template=TEMPLATE,margin=dict(t=20),
                          xaxis_title="K",yaxis_title="Inertia")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)

    with tabs[2]:
        section("Cluster Profile Comparison")
        CLUS_F=M["CLUS_F"]
        means=M["mfr_profile"].groupby("cluster_label")[CLUS_F].mean().round(2)
        st.dataframe(means,use_container_width=True)
        fig=go.Figure()
        for cname in M["CNAMES"].values():
            sub=M["mfr_profile"][M["mfr_profile"]["cluster_label"]==cname]
            fig.add_trace(go.Box(y=sub["median_price"],name=cname,
                marker_color=CMAP.get(cname,ACCENT),boxpoints=False))
        fig.update_layout(height=420,template=TEMPLATE,margin=dict(t=20),
                          yaxis_title="Median Price ₹",xaxis_title="")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)

    with tabs[3]:
        section("Radar Chart — Normalised Cluster Profiles")
        cluster_means=M["mfr_profile"].groupby("cluster_label")[CLUS_F].mean()
        cluster_norm=(cluster_means-cluster_means.min())/(cluster_means.max()-cluster_means.min())
        rl=["Portfolio","Median Price","Combo Ratio","Disc Rate","Class Div","Form Div","Ingr Div"]
        fig=go.Figure()
        for i,(label,row_data) in enumerate(cluster_norm.iterrows()):
            vals=row_data.tolist()+[row_data.tolist()[0]]; theta=rl+[rl[0]]
            color=list(CMAP.values())[i%len(CMAP)]
            fig.add_trace(go.Scatterpolar(r=vals,theta=theta,fill="toself",name=label,
                line_color=color,fillcolor=hex_to_rgba(color,0.15)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
            height=520,margin=dict(t=40),legend=dict(x=1.05,y=0.5))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRICE TIER CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏷️ ML: Price Tier Classifier":
    st.title("🏷️ ML: Price Tier Classifier")
    st.caption("Random Forest 4-class Classifier: Budget / Mid / Premium / Specialty | 80/20 split")

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi("Accuracy", M["acc_t"], "ok")
    with c2: kpi("Precision (wt)", M["prec_t_w"])
    with c3: kpi("Recall (wt)", M["rec_t_w"])
    with c4: kpi("Precision (macro)", M["prec_t_m"])
    with c5: kpi("Recall (macro)", M["rec_t_m"])

    tabs = st.tabs(["Confusion Matrix", "Per-Class Metrics", "Feature Importance"])

    with tabs[0]:
        section("Confusion Matrix")
        le_t=M["le_t"]; tier_cm=M["tier_cm"]
        cm_pct=tier_cm/tier_cm.sum(axis=1,keepdims=True)*100
        fig=go.Figure(go.Heatmap(
            z=tier_cm,x=le_t.classes_,y=le_t.classes_,
            colorscale=[[0,"#0A1829"],[0.4,"#0E3A6E"],[1,"#00B8FF"]],showscale=True,
            text=[[f"{tier_cm[i,j]:,}<br>({cm_pct[i,j]:.0f}%)" for j in range(4)] for i in range(4)],
            texttemplate="%{text}",textfont=dict(size=12),hoverinfo="skip"
        ))
        fig.update_layout(height=420,template=TEMPLATE,margin=dict(t=20),
                          xaxis_title="Predicted",yaxis_title="Actual")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)

    with tabs[1]:
        section("Per-Class Precision, Recall & F1")
        classes=le_t.classes_
        pr_v=[M["report_t"][c]["precision"] for c in classes]
        re_v=[M["report_t"][c]["recall"]    for c in classes]
        f1_v=[M["report_t"][c]["f1-score"]  for c in classes]
        fig=go.Figure()
        for name,vals,color in [("Precision",pr_v,ACCENT),("Recall",re_v,OK),("F1",f1_v,GOLD)]:
            fig.add_trace(go.Bar(x=classes,y=vals,name=name,marker_color=color,
                text=[f"{v:.3f}" for v in vals],textposition="outside"))
        fig.update_layout(barmode="group",height=420,template=TEMPLATE,
            margin=dict(t=20),yaxis_range=[0,1.15],yaxis_title="Score",
            legend=dict(orientation="h",y=1.05))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)
        insight("Specialty tier is best predicted (Precision=0.890) — it has a highly distinct feature profile "
                "vs Budget/Mid which share fuzzy boundaries around ₹50–₹150.")

    with tabs[2]:
        section("Feature Importance")
        fl={"ingr_median_price":"Ingredient Median Price","mfr_median_price":"Mfr Median Price",
            "pack_size_filled":"Pack Size","ingr_competition":"Competition",
            "dosage_form_enc":"Dosage Form","therapeutic_class_enc":"Therapeutic Class",
            "pack_unit_enc":"Pack Unit","num_active_ingredients":"Num Ingredients",
            "is_combo":"Is Combo","mfr_product_count":"Mfr Portfolio Size"}
        fi=M["tier_fi"].reset_index(); fi.columns=["feature","importance"]
        fi["feature"]=fi["feature"].map(fl).fillna(fi["feature"])
        fi=fi.sort_values("importance")
        fig=go.Figure(go.Bar(y=fi["feature"],x=fi["importance"],orientation="h",
            marker_color=[WARN if v>0.3 else GOLD if v>0.08 else ACCENT for v in fi["importance"]],
            text=[f"{v*100:.1f}%" for v in fi["importance"]],textposition="outside"))
        fig.update_layout(height=420,template=TEMPLATE,margin=dict(t=20,r=80),
                          xaxis_title="Importance (Gini)")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMAND SCORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 ML: Demand Scorer":
    st.title("📈 ML: Demand Proxy Scorer")
    st.caption("Rule-based scoring: log(competition) × therapeutic class weight × (1 + 0.5 × is_combo)")

    demand=M["demand_ingr"]
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Ingredients Scored", len(demand), "default", "{:,}")
    with c2: kpi("Max Score", demand["demand_score"].max(), "ok", "{:.2f}")
    with c3: kpi("Mean Score", demand["demand_score"].mean(), "gold", "{:.2f}")
    with c4: kpi("Top Class", 0, "purple", "Antidiabetic")  # hardcoded label

    tabs = st.tabs(["Top 25 Ranked", "Bubble Chart", "By Class"])

    with tabs[0]:
        section("Top 25 Ingredients by Demand Proxy Score")
        top25=demand.head(25).sort_values("demand_score",ascending=True)
        unique_c=demand["therapeutic_class"].unique()
        cmap_d={c:COLORS[i%len(COLORS)] for i,c in enumerate(unique_c)}
        fig=go.Figure(go.Bar(
            y=top25["primary_ingredient"],x=top25["demand_score"],orientation="h",
            marker_color=[cmap_d.get(c,"#888") for c in top25["therapeutic_class"]],
            text=[f"{v:.2f}" for v in top25["demand_score"]],textposition="outside",
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>"
        ))
        fig.update_layout(height=640,template=TEMPLATE,margin=dict(t=20,r=80),
                          xaxis_title="Demand Proxy Score")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)

    with tabs[1]:
        section("Demand Score vs Median Price vs Competition")
        fig=px.scatter(demand.head(60),x="demand_score",y="median_price",
            size="products",color="therapeutic_class",hover_name="primary_ingredient",
            text="primary_ingredient",size_max=45,
            color_discrete_sequence=COLORS,template=TEMPLATE,log_y=True,
            labels={"demand_score":"Demand Score","median_price":"Median Price ₹ (log)"})
        fig.update_traces(textposition="top center",textfont_size=8)
        fig.update_layout(height=580,margin=dict(t=20))
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)

    with tabs[2]:
        section("Average Demand Score by Therapeutic Class")
        class_demand=demand.groupby("therapeutic_class")["demand_score"].mean().sort_values(ascending=True).reset_index()
        fig=go.Figure(go.Bar(
            y=class_demand["therapeutic_class"].str.title(),
            x=class_demand["demand_score"],orientation="h",
            marker_color=[WARN if v>8 else GOLD if v>6 else ACCENT for v in class_demand["demand_score"]],
            text=[f"{v:.2f}" for v in class_demand["demand_score"]],textposition="outside"
        ))
        fig.update_layout(height=400,template=TEMPLATE,margin=dict(t=20,r=80),
                          xaxis_title="Avg Demand Score")
        _apply_plotly_theme(fig)
        st.plotly_chart(fig,use_container_width=True)
        insight("Antidiabetics and antihypertensives dominate demand scores — "
                "reflecting chronic disease prevalence, large patient populations, and high competition.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Model Comparison":
    st.title("🎯 Model Comparison Dashboard")
    st.markdown("Side-by-side accuracy, precision and recall for all 5 ML models.")

    section("Complete Scorecard")
    scorecard = pd.DataFrame([
        {"Model":"Price Prediction","Algorithm":"RF Regressor (200 trees)",
         "Accuracy":f"{M['acc_25']:.1f}% (±25%)",
         "Precision":f"{M['prec_tier_w']:.4f} (tier, wt)",
         "Recall":f"{M['rec_tier_w']:.4f} (tier, wt)",
         "Other":f"R²={M['r2_log']:.4f} | MAE=₹{M['mae']:.0f}",
         "Top Feature":"Ingr. Median Price (84%)"},
        {"Model":"Discontinuation Risk","Algorithm":"RF Classifier + Oversampling",
         "Accuracy":f"{M['acc_d']:.4f}",
         "Precision":f"{M['prec_disc']:.4f} (disc) / {M['prec_mac']:.4f} (macro)",
         "Recall":f"{M['rec_disc']:.4f} (disc) / {M['rec_mac']:.4f} (macro)",
         "Other":f"AUC={M['auc_d']:.4f}",
         "Top Feature":"Mfr Portfolio Size (57%)"},
        {"Model":"Market Segmentation","Algorithm":"K-Means (K=5) + PCA",
         "Accuracy":f"Silhouette={M['sil']:.4f}",
         "Precision":f"Var. Explained={M['var_exp']*100:.1f}%",
         "Recall":f"Inertia={M['inertia']:,.0f}",
         "Other":"5,136 manufacturers profiled",
         "Top Feature":"Portfolio size + Price tier"},
        {"Model":"Price Tier Classifier","Algorithm":"RF Classifier (4-class)",
         "Accuracy":f"{M['acc_t']:.4f}",
         "Precision":f"{M['prec_t_w']:.4f} (wt) / {M['prec_t_m']:.4f} (macro)",
         "Recall":f"{M['rec_t_w']:.4f} (wt) / {M['rec_t_m']:.4f} (macro)",
         "Other":"Specialty F1=0.787",
         "Top Feature":"Ingr. Median Price (58%)"},
        {"Model":"Demand Proxy Scorer","Algorithm":"Rule-based (no training)",
         "Accuracy":f"{len(M['demand_ingr']):,} ingredients",
         "Precision":"Score range 0–11.54",
         "Recall":"Top-10 = 22.5% of all products",
         "Other":"No train/test needed",
         "Top Feature":"Competition × class weight"},
    ])
    st.dataframe(scorecard, use_container_width=True, hide_index=True)

    section("Cross-Model Accuracy, Precision & Recall")
    cross = pd.DataFrame([
        {"Model":"Price (±25%)",      "Accuracy":M["acc_25"]/100,   "Precision":M["prec_tier_w"],  "Recall":M["rec_tier_w"]},
        {"Model":"Disc (disc class)", "Accuracy":M["acc_d"],        "Precision":M["prec_disc"],    "Recall":M["rec_disc"]},
        {"Model":"Disc (macro)",      "Accuracy":M["acc_d"],        "Precision":M["prec_mac"],     "Recall":M["rec_mac"]},
        {"Model":"Tier (weighted)",   "Accuracy":M["acc_t"],        "Precision":M["prec_t_w"],     "Recall":M["rec_t_w"]},
        {"Model":"Tier (macro)",      "Accuracy":M["acc_t"],        "Precision":M["prec_t_m"],     "Recall":M["rec_t_m"]},
    ])
    fig=go.Figure()
    for metric,color in [("Accuracy",ACCENT),("Precision",OK),("Recall",WARN)]:
        fig.add_trace(go.Bar(x=cross["Model"],y=cross[metric],name=metric,marker_color=color,
            text=[f"{v:.3f}" for v in cross[metric]],textposition="outside"))
    fig.add_hline(y=0.75,line_dash="dash",line_color="grey",opacity=0.5,
                  annotation_text="0.75",annotation_position="right")
    fig.update_layout(barmode="group",height=480,template=TEMPLATE,
        margin=dict(t=40),yaxis_title="Score",yaxis_range=[0,1.15],
        legend=dict(orientation="h",y=1.06,x=0.5,xanchor="center"))
    _apply_plotly_theme(fig)
    st.plotly_chart(fig,use_container_width=True)

    section("Key Insights")
    insights_data = [
        ("🤖 Price", f"R²={M['r2_log']:.4f} | Acc±25%={M['acc_25']:.1f}% | Ingredient median price (84%) dominates pricing"),
        ("⚠️ Discontinuation", f"AUC={M['auc_d']:.4f} | Recall={M['rec_disc']:.4f} | {int((df['disc_risk_score']>0.5).sum()):,} active products at high risk"),
        ("🏭 Segmentation", f"Silhouette={M['sil']:.4f} | {len(M['mfr_profile']):,} manufacturers across 5 segments | Market Leaders (22) vs Mass Market (2,329)"),
        ("🏷️ Price Tier", f"Acc={M['acc_t']:.4f} | Specialty Precision=0.890 | Budget/Mid boundary at ₹50–₹150 is hardest"),
        ("📈 Demand", f"{len(M['demand_ingr']):,} ingredients scored | Glimepiride top (11.54) | Antidiabetics dominate demand"),
    ]
    for title, text in insights_data:
        st.markdown(f'<div class="insight-box"><b>{title}:</b> {text}</div>', unsafe_allow_html=True)
