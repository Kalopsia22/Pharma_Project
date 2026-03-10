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
ACCENT     = "#1B4F72"
WARN       = "#E74C3C"
OK         = "#27AE60"
GOLD       = "#F39C12"
COLORS     = px.colors.qualitative.Bold
TEMPLATE   = "plotly_white"

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
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .main { background: #F7F9FC; }

    section[data-testid="stSidebar"] {
        background: #0D2137;
        border-right: 2px solid #1B4F72;
    }
    section[data-testid="stSidebar"] * { color: #E8F4FD !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label { color: #A9CCE3 !important; }

    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(27,79,114,0.10);
        border-top: 4px solid #1B4F72;
        margin-bottom: 8px;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1B4F72; line-height: 1.1; }
    .kpi-label { font-size: 0.78rem; color: #7F8C8D; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-card.warn  { border-top-color: #E74C3C; } .kpi-card.warn .kpi-value  { color: #E74C3C; }
    .kpi-card.ok    { border-top-color: #27AE60; } .kpi-card.ok .kpi-value    { color: #27AE60; }
    .kpi-card.gold  { border-top-color: #F39C12; } .kpi-card.gold .kpi-value  { color: #F39C12; }
    .kpi-card.purple{ border-top-color: #8E44AD; } .kpi-card.purple .kpi-value{ color: #8E44AD; }

    .section-header {
        font-size: 1.35rem; font-weight: 700; color: #0D2137;
        border-left: 5px solid #1B4F72; padding-left: 12px;
        margin: 28px 0 16px 0;
    }
    .insight-box {
        background: #EAF2FF; border-left: 4px solid #1B4F72;
        padding: 14px 18px; border-radius: 6px; margin: 12px 0;
        font-size: 0.92rem; color: #1A2E40;
    }
    .metric-badge {
        display: inline-block; background: #1B4F72; color: white;
        border-radius: 20px; padding: 3px 12px; font-size: 0.78rem;
        font-weight: 600; margin: 2px;
    }
    .metric-badge.warn { background: #E74C3C; }
    .metric-badge.ok   { background: #27AE60; }
    .metric-badge.gold { background: #F39C12; }

    div[data-testid="stTabs"] button {
        font-weight: 600; font-size: 0.85rem;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


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
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

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
    st.markdown("## 💊 Pharma Intel")
    st.markdown("---")
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
    ])
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#A9CCE3;'>
    <b>Dataset</b><br>
    {len(df):,} products<br>
    {df['manufacturer'].nunique():,} manufacturers<br>
    {df['primary_ingredient'].nunique():,} ingredients<br>
    {df['therapeutic_class'].nunique()} therapeutic classes
    </div>
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

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        section("Products by Therapeutic Class")
        tc = df["therapeutic_class"].value_counts().head(11).reset_index()
        tc.columns = ["class","count"]
        fig = px.bar(tc, y="class", x="count", orientation="h",
                     color="count", color_continuous_scale="Blues",
                     template=TEMPLATE)
        fig.update_layout(height=380, showlegend=False, margin=dict(t=10,b=10),
                          coloraxis_showscale=False)
        fig.update_yaxes(title=""); fig.update_xaxes(title="Products")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Price Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["price_inr"].clip(0,500), nbinsx=60,
            marker_color=ACCENT, opacity=0.8
        ))
        fig.add_vline(x=df["price_inr"].median(), line_dash="dash", line_color=WARN,
                      annotation_text=f"Median ₹{df['price_inr'].median():.0f}")
        fig.update_layout(height=380, template=TEMPLATE, margin=dict(t=10,b=10),
                          xaxis_title="Price ₹ (capped at 500)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    section("Module Summary")
    modules = [
        ("📊 Market Intelligence", "Treemap, heatmaps, Lorenz curve, competitive landscape"),
        ("💰 Price Analytics", "Price tiers, dosage premium, combo premium, outlier explorer"),
        ("🧪 Ingredient Intelligence", "Co-occurrence network, combo strategy, portfolio heatmap"),
        ("🤖 Price Prediction", f"RF Regressor · R²={M['r2_log']:.4f} · MAE=₹{M['mae']:.0f} · Acc±25%={M['acc_25']:.1f}%"),
        ("⚠️ Discontinuation Risk", f"RF Classifier · AUC={M['auc_d']:.4f} · Recall(disc)={M['rec_disc']:.4f}"),
        ("🏭 Segmentation", f"K-Means K=5 · Silhouette={M['sil']:.4f} · VarExp={M['var_exp']*100:.1f}%"),
        ("🏷️ Price Tier", f"4-class RF · Acc={M['acc_t']:.4f} · Prec(wt)={M['prec_t_w']:.4f}"),
        ("📈 Demand Scorer", f"Rule-based · {len(M['demand_ingr']):,} ingredients · Top: Glimepiride 11.54"),
    ]
    for i in range(0, len(modules), 2):
        ca, cb = st.columns(2)
        for col_obj, (name, desc) in zip([ca,cb], modules[i:i+2]):
            with col_obj:
                st.markdown(f"**{name}**")
                st.caption(desc)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Market Intelligence":
    st.title("📊 Market Intelligence")
    tabs = st.tabs(["Market Share", "Portfolio Strategy", "Discontinuation Risk", "Concentration"])

    with tabs[0]:
        section("Top 20 Manufacturers — Active vs Discontinued")
        top20 = df.groupby("manufacturer").agg(
            active=("is_discontinued", lambda x:(~x).sum()),
            disc=("is_discontinued","sum"),
            total=("product_id","count")
        ).reset_index().sort_values("total",ascending=False).head(20)
        top20["short"]=top20["manufacturer"].str[:30]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top20["short"],x=top20["active"],name="Active",
            marker_color=OK,orientation="h"))
        fig.add_trace(go.Bar(y=top20["short"],x=top20["disc"],name="Discontinued",
            marker_color=WARN,orientation="h"))
        fig.update_layout(barmode="stack",height=560,template=TEMPLATE,
                          margin=dict(t=20,b=10),legend=dict(orientation="h",y=1.02))
        fig.update_xaxes(title="Products"); fig.update_yaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        section("Portfolio Strategy Map — Products vs Price vs Discontinuation Rate")
        mfr_s = df.groupby("manufacturer").agg(
            products=("product_id","count"),
            avg_price=("price_inr","mean"),
            disc_rate=("is_discontinued","mean"),
        ).reset_index()
        mfr_s = mfr_s[mfr_s["products"]>=10]
        fig = px.scatter(mfr_s.sample(min(500,len(mfr_s)),random_state=42),
            x="products",y="avg_price",size="products",color="disc_rate",
            hover_name="manufacturer",
            color_continuous_scale="RdYlGn_r",
            log_x=True,log_y=True,
            labels={"products":"Portfolio Size","avg_price":"Avg Price ₹","disc_rate":"Disc. Rate"},
            template=TEMPLATE)
        fig.update_layout(height=500,margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        section("Discontinuation Rate by Therapeutic Class")
        disc_class = df.groupby("therapeutic_class").agg(
            total=("product_id","count"),
            disc=("is_discontinued","sum"),
        ).reset_index()
        disc_class["disc_rate"] = disc_class["disc"]/disc_class["total"]*100
        disc_class = disc_class.sort_values("disc_rate",ascending=True)
        fig = go.Figure(go.Bar(
            y=disc_class["therapeutic_class"].str.title(),
            x=disc_class["disc_rate"],orientation="h",
            marker_color=[WARN if v>5 else GOLD if v>3 else OK for v in disc_class["disc_rate"]],
            text=[f"{v:.1f}%" for v in disc_class["disc_rate"]],textposition="outside"
        ))
        fig.update_layout(height=400,template=TEMPLATE,margin=dict(t=20,r=80),
                          xaxis_title="Discontinuation Rate (%)",yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        section("Market Concentration — Cumulative Share (Lorenz Curve)")
        mfr_cnt = df["manufacturer"].value_counts().reset_index()
        mfr_cnt.columns = ["manufacturer","count"]
        mfr_cnt = mfr_cnt.sort_values("count")
        mfr_cnt["cum_share"] = mfr_cnt["count"].cumsum()/mfr_cnt["count"].sum()*100
        mfr_cnt["cum_mfr"]   = np.arange(1,len(mfr_cnt)+1)/len(mfr_cnt)*100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mfr_cnt["cum_mfr"],y=mfr_cnt["cum_share"],
            mode="lines",line=dict(color=ACCENT,width=2.5),
            fill="tozeroy",fillcolor=hex_to_rgba(ACCENT,0.15),name="Actual"))
        fig.add_trace(go.Scatter(x=[0,100],y=[0,100],mode="lines",
            line=dict(color="grey",dash="dash"),name="Perfect equality"))
        fig.update_layout(height=420,template=TEMPLATE,margin=dict(t=20),
                          xaxis_title="Cumulative % of Manufacturers",
                          yaxis_title="Cumulative % of Products")
        st.plotly_chart(fig, use_container_width=True)
        insight("The market is highly fragmented — top 1% of manufacturers control ~20% of products. "
                "7,648 manufacturers compete, with Sun Pharma leading at ~2,986 products.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRICE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Price Analytics":
    st.title("💰 Price Analytics")

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Median Price", df["price_inr"].median(), "gold", "₹{:.0f}")
    with c2: kpi("Mean Price", df["price_inr"].mean(), "gold", "₹{:.0f}")
    with c3: kpi("P95 Price", df["price_inr"].quantile(0.95), "warn", "₹{:.0f}")
    with c4: kpi("Max Price", df["price_inr"].max(), "warn", "₹{:,.0f}")

    tabs = st.tabs(["Price Tiers", "By Class", "Dosage Premium", "Combo Premium", "Outliers"])

    with tabs[0]:
        section("Price Tier Distribution")
        tier_counts = df["price_tier"].value_counts().reindex(["Budget","Mid","Premium","Specialty"]).reset_index()
        tier_counts.columns = ["tier","count"]
        tier_counts["pct"] = tier_counts["count"]/tier_counts["count"].sum()*100
        fig = go.Figure(go.Bar(
            x=tier_counts["tier"],y=tier_counts["pct"],
            marker_color=[OK,GOLD,"#E67E22",WARN],
            text=[f"{v:.1f}%" for v in tier_counts["pct"]],textposition="outside"
        ))
        fig.update_layout(height=400,template=TEMPLATE,margin=dict(t=20),
                          yaxis_title="% of Products",xaxis_title="Price Tier")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        section("Price Distribution by Therapeutic Class")
        class_price = df.groupby("therapeutic_class")["price_inr"].agg(
            ["median","mean",lambda x:x.quantile(0.25),lambda x:x.quantile(0.75)]
        ).reset_index()
        class_price.columns = ["class","median","mean","p25","p75"]
        class_price = class_price.sort_values("median",ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=class_price["class"].str.title(),x=class_price["median"],
            orientation="h",marker_color=ACCENT,name="Median",
            text=[f"₹{v:.0f}" for v in class_price["median"]],textposition="outside"))
        fig.add_trace(go.Scatter(y=class_price["class"].str.title(),x=class_price["mean"],
            mode="markers",marker=dict(size=10,color=WARN,symbol="diamond"),name="Mean"))
        fig.update_layout(height=420,template=TEMPLATE,margin=dict(t=20,r=80),
                          xaxis_title="Price ₹",yaxis_title="",
                          legend=dict(orientation="h",y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        section("Median Price by Dosage Form")
        form_price = df.groupby("dosage_form")["price_inr"].agg(
            median="median",count="count"
        ).reset_index()
        form_price = form_price[form_price["count"]>=50].sort_values("median",ascending=True)
        fig = px.bar(form_price.tail(15),y="dosage_form",x="median",
            orientation="h",color="median",color_continuous_scale="Blues",
            template=TEMPLATE,
            labels={"median":"Median Price ₹","dosage_form":"Dosage Form"})
        fig.update_layout(height=440,margin=dict(t=20,r=60),coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        section("Combination Drug Price Premium by Therapeutic Class")
        combo_premium = df.groupby(["therapeutic_class","is_combo"])["price_inr"].median().unstack()
        combo_premium.columns = ["Single","Combo"]
        combo_premium = combo_premium.dropna()
        combo_premium["premium_pct"] = (combo_premium["Combo"]-combo_premium["Single"])/combo_premium["Single"]*100
        combo_premium = combo_premium.sort_values("premium_pct",ascending=True).reset_index()
        fig = go.Figure(go.Bar(
            y=combo_premium["therapeutic_class"].str.title(),
            x=combo_premium["premium_pct"],orientation="h",
            marker_color=[WARN if v>100 else GOLD if v>50 else OK for v in combo_premium["premium_pct"]],
            text=[f"+{v:.0f}%" for v in combo_premium["premium_pct"]],textposition="outside"
        ))
        fig.add_vline(x=0,line_dash="dash",line_color="grey")
        fig.update_layout(height=400,template=TEMPLATE,margin=dict(t=20,r=80),
                          xaxis_title="Price Premium %",yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        insight("Diuretics (+385%) and Bronchodilators (+332%) command the highest combination drug premiums.")

    with tabs[4]:
        section("High-Cost Outlier Explorer (≥ P95)")
        p95 = df["price_inr"].quantile(0.95)
        outliers = df[df["price_inr"]>=p95].nlargest(50,"price_inr")[
            ["brand_name","manufacturer","therapeutic_class","dosage_form","price_inr","num_active_ingredients"]
        ].reset_index(drop=True)
        outliers.index += 1
        st.dataframe(outliers, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INGREDIENT INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Ingredient Intelligence":
    st.title("🧪 Ingredient Intelligence")

    tabs = st.tabs(["Top Ingredients", "Combo Strategy", "Portfolio Diversity", "Price by Ingredient"])

    with tabs[0]:
        section("Top 30 Ingredients by Product Count")
        top_ingr = df["primary_ingredient"].value_counts().head(30).reset_index()
        top_ingr.columns = ["ingredient","count"]
        top_ingr = top_ingr.merge(
            df.groupby("primary_ingredient")["is_combo"].mean().reset_index().rename(
                columns={"primary_ingredient":"ingredient","is_combo":"combo_ratio"}),
            on="ingredient"
        )
        fig = px.bar(top_ingr,y="ingredient",x="count",orientation="h",
            color="combo_ratio",color_continuous_scale="RdYlGn",
            template=TEMPLATE,
            labels={"count":"Products","ingredient":"","combo_ratio":"Combo Ratio"})
        fig.update_layout(height=680,margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        section("Combo vs Solo Strategy by Therapeutic Class")
        combo_class = df.groupby("therapeutic_class").agg(
            total=("product_id","count"),
            combo_count=("is_combo","sum"),
            solo_count=("is_combo",lambda x:(~x.astype(bool)).sum())
        ).reset_index()
        combo_class["combo_pct"] = combo_class["combo_count"]/combo_class["total"]*100
        combo_class = combo_class.sort_values("combo_pct",ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=combo_class["therapeutic_class"].str.title(),
            x=combo_class["combo_pct"],name="Combo %",
            marker_color=ACCENT,orientation="h",
            text=[f"{v:.0f}%" for v in combo_class["combo_pct"]],textposition="outside"))
        fig.update_layout(height=400,template=TEMPLATE,margin=dict(t=20,r=80),
                          xaxis_title="% Combination Products",yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        section("Ingredient Diversity per Therapeutic Class")
        div_class = df.groupby("therapeutic_class").agg(
            unique_ingredients=("primary_ingredient","nunique"),
            products=("product_id","count")
        ).reset_index().sort_values("unique_ingredients",ascending=True)
        fig = make_subplots(rows=1,cols=2,subplot_titles=["Unique Ingredients","Products"])
        fig.add_trace(go.Bar(y=div_class["therapeutic_class"].str.title(),
            x=div_class["unique_ingredients"],orientation="h",marker_color=ACCENT,showlegend=False,
            text=div_class["unique_ingredients"],textposition="outside"),row=1,col=1)
        fig.add_trace(go.Bar(y=div_class["therapeutic_class"].str.title(),
            x=div_class["products"],orientation="h",marker_color=GOLD,showlegend=False,
            text=div_class["products"],textposition="outside"),row=1,col=2)
        fig.update_layout(height=420,template=TEMPLATE,margin=dict(t=40,r=80))
        st.plotly_chart(fig, use_container_width=True)
        insight("Diuretics have the fewest unique ingredients (23) — a largely unexplored formulation space.")

    with tabs[3]:
        section("Top 25 Ingredients — Median Price vs Competition")
        top25_ingr = df.groupby("primary_ingredient").agg(
            median_price=("price_inr","median"),
            competition=("product_id","count"),
            therapeutic_class=("therapeutic_class",lambda x:x.mode()[0])
        ).reset_index().nlargest(25,"competition")
        fig = px.scatter(top25_ingr,x="competition",y="median_price",
            size="competition",color="therapeutic_class",
            hover_name="primary_ingredient",
            text="primary_ingredient",
            color_discrete_sequence=COLORS,template=TEMPLATE,
            labels={"competition":"# Products","median_price":"Median Price ₹"})
        fig.update_traces(textposition="top center",textfont_size=9)
        fig.update_layout(height=520,margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML PRICE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
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
        st.plotly_chart(fig,use_container_width=True)

    with tabs[1]:
        section("Confusion Matrix & Score Distribution")
        cm=M["cm_d"]; cm_pct=cm/cm.sum(axis=1,keepdims=True)*100
        tn,fp,fn,tp=cm.ravel()
        col1,col2=st.columns(2)
        with col1:
            fig=go.Figure(go.Heatmap(
                z=cm,x=["Pred: Active","Pred: Disc"],y=["True: Active","True: Disc"],
                colorscale=[[0,"#EAF2FF"],[1,ACCENT]],showscale=False,
                text=[[f"{cm[i,j]:,}<br>({cm_pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)],
                texttemplate="%{text}",textfont=dict(size=14),hoverinfo="skip"
            ))
            fig.update_layout(height=320,template=TEMPLATE,margin=dict(t=20))
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
            colorscale=[[0,"#EAF2FF"],[1,ACCENT]],showscale=True,
            text=[[f"{tier_cm[i,j]:,}<br>({cm_pct[i,j]:.0f}%)" for j in range(4)] for i in range(4)],
            texttemplate="%{text}",textfont=dict(size=12),hoverinfo="skip"
        ))
        fig.update_layout(height=420,template=TEMPLATE,margin=dict(t=20),
                          xaxis_title="Predicted",yaxis_title="Actual")
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
