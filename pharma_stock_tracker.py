"""
╔══════════════════════════════════════════════════════════════════════╗
║   INDIAN PHARMA STOCK INTELLIGENCE PLATFORM                         ║
║   Live Tracker (NSE + BSE) + 25-Year AI Price Forecast              ║
║   Models: ARIMA · LSTM · XGBoost · Ensemble                         ║
╚══════════════════════════════════════════════════════════════════════╝
Run:  streamlit run pharma_stock_tracker.py
Deps: pip install streamlit yfinance pandas numpy plotly scikit-learn
      prophet tensorflow xgboost requests beautifulsoup4 ta
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta, date
import warnings, time, requests, re
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design constants ──────────────────────────────────────────────────────────
GREEN      = "#00C896"
RED        = "#FF4B6E"
GOLD       = "#F5A623"
BLUE       = "#1E90FF"
DARK_BG    = "#0A0E1A"
CARD_BG    = "#111827"
BORDER     = "#1F2937"
TEXT_PRI   = "#F9FAFB"
TEXT_SEC   = "#9CA3AF"
ACCENT     = "#00C896"
COLORS_10  = ["#00C896","#1E90FF","#F5A623","#FF4B6E","#A78BFA",
               "#34D399","#60A5FA","#FBBF24","#F87171","#C084FC"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Sora', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT_PRI};
}}
.main {{ background: {DARK_BG}; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}

section[data-testid="stSidebar"] {{
    background: #060912 !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_PRI} !important; }}
section[data-testid="stSidebar"] hr {{ border-color: {BORDER}; }}

/* Ticker card */
.ticker-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}}
.ticker-card:hover {{ border-color: {ACCENT}; }}
.ticker-name {{ font-size: 0.78rem; color: {TEXT_SEC}; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }}
.ticker-price {{ font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 700; color: {TEXT_PRI}; line-height: 1.1; }}
.ticker-change-up   {{ font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: {GREEN}; font-weight: 600; }}
.ticker-change-down {{ font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: {RED};   font-weight: 600; }}
.ticker-meta {{ font-size: 0.72rem; color: {TEXT_SEC}; margin-top: 4px; }}

/* KPI strip */
.kpi-strip {{
    background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 14px 20px;
    text-align: center; margin-bottom: 6px;
}}
.kpi-strip-val  {{ font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:{TEXT_PRI}; }}
.kpi-strip-lbl  {{ font-size:0.70rem; color:{TEXT_SEC}; text-transform:uppercase; letter-spacing:0.07em; margin-top:3px; }}
.kpi-strip-sub  {{ font-size:0.78rem; margin-top:2px; }}

/* Section header */
.section-hdr {{
    font-size: 0.72rem; font-weight: 700; color: {ACCENT};
    letter-spacing: 0.15em; text-transform: uppercase;
    border-bottom: 1px solid {BORDER}; padding-bottom: 8px;
    margin: 22px 0 14px 0;
}}

/* Sentiment pill */
.pill-pos  {{ display:inline-block; background:rgba(0,200,150,0.15); color:{GREEN};
              border:1px solid {GREEN}; border-radius:20px; padding:2px 12px; font-size:0.78rem; font-weight:600; }}
.pill-neg  {{ display:inline-block; background:rgba(255,75,110,0.15); color:{RED};
              border:1px solid {RED};   border-radius:20px; padding:2px 12px; font-size:0.78rem; font-weight:600; }}
.pill-neu  {{ display:inline-block; background:rgba(156,163,175,0.15); color:{TEXT_SEC};
              border:1px solid {BORDER}; border-radius:20px; padding:2px 12px; font-size:0.78rem; font-weight:600; }}

/* Alert box */
.alert-box {{
    background: rgba(245,166,35,0.10); border-left: 3px solid {GOLD};
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: {TEXT_PRI}; margin: 10px 0;
}}
.model-badge {{
    display:inline-block; background:rgba(30,144,255,0.15); color:{BLUE};
    border:1px solid {BLUE}; border-radius:6px; padding:2px 10px;
    font-size:0.72rem; font-weight:600; margin:2px;
}}

/* Dark plotly override */
[data-testid="stPlotlyChart"] {{ background: transparent !important; }}
div[data-testid="stTabs"] button {{ color: {TEXT_SEC} !important; font-size:0.82rem; }}
div[data-testid="stTabs"] button[aria-selected="true"] {{ color:{ACCENT} !important; border-bottom-color:{ACCENT} !important; }}

/* Scrolling marquee for live prices */
.marquee-wrap {{
    background: #060912; border-top: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};
    overflow: hidden; padding: 8px 0; margin-bottom: 18px;
}}
.marquee-inner {{
    display: flex; animation: marquee 60s linear infinite; white-space: nowrap;
}}
.marquee-item {{
    font-family:'JetBrains Mono',monospace; font-size:0.78rem;
    padding: 0 28px; border-right: 1px solid {BORDER};
}}
@keyframes marquee {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-50%); }} }}

input, select, textarea {{ background: {CARD_BG} !important; color: {TEXT_PRI} !important; }}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# COMPANY REGISTRY — All major listed Indian Pharma companies (NSE + BSE)
# ══════════════════════════════════════════════════════════════════════════════╝
PHARMA_COMPANIES = {
    # NSE Symbol         : (Display Name,                    BSE Code, Sector)
    "SUNPHARMA.NS"       : ("Sun Pharmaceutical",            "524715", "Large Cap"),
    "DRREDDY.NS"         : ("Dr. Reddy's Laboratories",      "500124", "Large Cap"),
    "CIPLA.NS"           : ("Cipla Ltd",                     "500087", "Large Cap"),
    "DIVISLAB.NS"        : ("Divi's Laboratories",           "532488", "Large Cap"),
    "MANKIND.NS"         : ("Mankind Pharma",                "543904", "Large Cap"),
    "TORNTPHARM.NS"      : ("Torrent Pharmaceuticals",       "500420", "Large Cap"),
    "LUPIN.NS"           : ("Lupin Ltd",                     "500257", "Large Cap"),
    "AUROPHARMA.NS"      : ("Aurobindo Pharma",              "524804", "Large Cap"),
    "ALKEM.NS"           : ("Alkem Laboratories",            "539523", "Large Cap"),
    "BIOCON.NS"          : ("Biocon Ltd",                    "532523", "Large Cap"),
    "ZYDUSLIFE.NS"       : ("Zydus Lifesciences",            "532321", "Large Cap"),
    "IPCALAB.NS"         : ("IPCA Laboratories",             "524494", "Mid Cap"),
    "GLENMARK.NS"        : ("Glenmark Pharmaceuticals",      "532296", "Mid Cap"),
    "AJANTPHARM.NS"      : ("Ajanta Pharma",                 "532331", "Mid Cap"),
    "GRANULES.NS"        : ("Granules India",                "532482", "Mid Cap"),
    "NATCOPHARM.NS"      : ("Natco Pharma",                  "524816", "Mid Cap"),
    "ABBOTINDIA.NS"      : ("Abbott India",                  "500488", "Mid Cap"),
    "PFIZER.NS"          : ("Pfizer India",                  "500680", "Mid Cap"),
    "SANOFI.NS"          : ("Sanofi India",                  "500674", "Mid Cap"),
    "GLAXO.NS"           : ("GSK Pharmaceuticals",           "500660", "Mid Cap"),
    "LAURUSLABS.NS"      : ("Laurus Labs",                   "540222", "Mid Cap"),
    "ERIS.NS"            : ("Eris Lifesciences",             "540596", "Mid Cap"),
    "JBCHEPHARM.NS"      : ("JB Chemicals & Pharma",         "506943", "Mid Cap"),
    "PIRAMALPHARM.NS"    : ("Piramal Pharma",                "543635", "Mid Cap"),
    "KSKPHARMA.NS"       : ("Kopran Ltd",                    "524280", "Small Cap"),
    "SOLARA.NS"          : ("Solara Active Pharma",          "541540", "Small Cap"),
    "STRIDES.NS"         : ("Strides Pharma",                "532531", "Small Cap"),
    "MARKSANS.NS"        : ("Marksans Pharma",               "524404", "Small Cap"),
    "CAPLIPOINT.NS"      : ("Caplin Point Lab",              "524742", "Small Cap"),
    "SEQUENT.NS"         : ("Sequent Scientific",            "512529", "Small Cap"),
}

# Nifty Pharma Index
NIFTY_PHARMA = "^CNXPHARMA"
SENSEX        = "^BSESN"
NIFTY50       = "^NSEI"

PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(family="Sora, sans-serif", color=TEXT_PRI, size=12),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, showgrid=True),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
        margin=dict(t=40, b=40, l=60, r=20),
    )
)

def apply_theme(fig, height=420):
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(family="Sora, sans-serif", color=TEXT_PRI, size=12),
        height=height, margin=dict(t=40,b=40,l=60,r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    )
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, showgrid=True, zeroline=False)
    return fig


# ╔══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════╝
@st.cache_data(ttl=300, show_spinner=False)   # 5-min cache for live prices
def fetch_live_quote(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="2d", interval="1m")
        prev_close = info.get("previousClose", info.get("regularMarketPreviousClose", None))
        price      = info.get("currentPrice", info.get("regularMarketPrice", None))
        if price is None and not hist.empty:
            price = float(hist["Close"].iloc[-1])
        change     = price - prev_close if (price and prev_close) else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        return {
            "price":          round(price, 2) if price else None,
            "prev_close":     round(prev_close, 2) if prev_close else None,
            "change":         round(change, 2),
            "change_pct":     round(change_pct, 2),
            "open":           info.get("open", info.get("regularMarketOpen")),
            "high":           info.get("dayHigh", info.get("regularMarketDayHigh")),
            "low":            info.get("dayLow",  info.get("regularMarketDayLow")),
            "volume":         info.get("volume",  info.get("regularMarketVolume")),
            "market_cap":     info.get("marketCap"),
            "pe_ratio":       info.get("trailingPE"),
            "pb_ratio":       info.get("priceToBook"),
            "eps":            info.get("trailingEps"),
            "div_yield":      info.get("dividendYield"),
            "week52_high":    info.get("fiftyTwoWeekHigh"),
            "week52_low":     info.get("fiftyTwoWeekLow"),
            "avg_volume":     info.get("averageVolume"),
            "beta":           info.get("beta"),
            "roe":            info.get("returnOnEquity"),
            "profit_margin":  info.get("profitMargins"),
            "revenue":        info.get("totalRevenue"),
            "debt_equity":    info.get("debtToEquity"),
            "name":           info.get("longName", ticker),
            "sector":         info.get("sector","Healthcare"),
            "exchange":       info.get("exchange","NSE"),
        }
    except Exception as e:
        return {"price": None, "change_pct": 0, "change": 0, "error": str(e)}


@st.cache_data(ttl=900, show_spinner=False)   # 15-min cache for history
def fetch_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        return df
    except:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        income    = t.quarterly_financials
        balance   = t.quarterly_balance_sheet
        cashflow  = t.quarterly_cashflow
        return {"income": income, "balance": balance, "cashflow": cashflow}
    except:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_index_data() -> dict:
    result = {}
    for sym, label in [(NIFTY_PHARMA,"Nifty Pharma"),(SENSEX,"BSE Sensex"),(NIFTY50,"Nifty 50")]:
        try:
            t = yf.Ticker(sym)
            info = t.info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev  = info.get("previousClose") or info.get("regularMarketPreviousClose")
            change_pct = ((price-prev)/prev*100) if (price and prev) else 0
            result[label] = {"price": price, "change_pct": round(change_pct,2)}
        except:
            result[label] = {"price": None, "change_pct": 0}
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_sentiment(company_name: str) -> list:
    """Fetch recent headlines and assign simple sentiment via keyword matching."""
    pos_words = {"gain","rise","surge","profit","growth","strong","beat","record",
                 "upgrade","outperform","buy","positive","robust","rally","high"}
    neg_words = {"fall","drop","loss","decline","miss","downgrade","sell","weak",
                 "concern","risk","cut","negative","pressure","low","lawsuit"}
    try:
        query = company_name.replace(" ","+") + "+stock+India"
        url   = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        resp  = requests.get(url, timeout=8,
                    headers={"User-Agent":"Mozilla/5.0"})
        items = re.findall(r"<title>(.*?)</title>", resp.text)[2:12]
        news  = []
        for title in items:
            title = re.sub(r"<[^>]+>","",title).strip()
            words = set(title.lower().split())
            pos   = len(words & pos_words)
            neg   = len(words & neg_words)
            sentiment = "positive" if pos>neg else ("negative" if neg>pos else "neutral")
            news.append({"title": title, "sentiment": sentiment})
        return news[:8]
    except:
        return []


# ╔══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════╝
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30:
        return df
    df = df.copy()
    # Moving averages
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_20"]  = df["Close"].ewm(span=20).mean()
    # Bollinger Bands
    bb_std        = df["Close"].rolling(20).std()
    df["BB_upper"] = df["SMA_20"] + 2*bb_std
    df["BB_lower"] = df["SMA_20"] - 2*bb_std
    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100/(1+rs))
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    # ATR
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl,hpc,lpc],axis=1).max(axis=1).rolling(14).mean()
    # OBV
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV"] = obv
    # Volume MA
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    return df


# ╔══════════════════════════════════════════════════════════════════════════════
# ML PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════╝
@st.cache_data(ttl=86400, show_spinner=False)   # cache predictions 24h
def run_all_predictions(ticker: str, years_ahead: int = 25) -> dict:
    """
    Run ARIMA, LSTM, XGBoost and Ensemble predictions.
    Returns dict with forecast DataFrames and model metadata.
    """
    df = fetch_history(ticker, period="max", interval="1d")
    if df.empty or len(df) < 252:
        return {"error": "Insufficient historical data (need ≥1 year)"}

    prices   = df["Close"].values.astype(float)
    log_ret  = np.diff(np.log(prices))
    last_date = df.index[-1]
    n_days   = years_ahead * 252   # trading days
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=n_days, freq="B")    # Business days only

    results = {}

    # ── 1. ARIMA ──────────────────────────────────────────────────────────────
    try:
        from statsmodels.tsa.arima.model import ARIMA
        # Fit on log returns (stationary)
        model_arima = ARIMA(log_ret[-min(1500,len(log_ret)):], order=(2,0,2))
        fit_arima   = model_arima.fit()
        fc_ret      = fit_arima.forecast(steps=n_days)
        # Monte Carlo to get confidence bands
        mc_paths = []
        np.random.seed(42)
        for _ in range(500):
            noise = np.random.normal(0, fit_arima.resid.std(), n_days)
            path  = fc_ret + noise
            mc_paths.append(np.exp(np.cumsum(path)) * prices[-1])
        mc_arr     = np.array(mc_paths)
        arima_mean = np.median(mc_arr, axis=0)
        arima_lo   = np.percentile(mc_arr, 10, axis=0)
        arima_hi   = np.percentile(mc_arr, 90, axis=0)
        results["ARIMA"] = {
            "mean": arima_mean, "lo": arima_lo, "hi": arima_hi,
            "aic": round(fit_arima.aic, 2),
            "color": GOLD,
        }
    except Exception as e:
        results["ARIMA"] = {"error": str(e)}

    # ── 2. XGBoost ────────────────────────────────────────────────────────────
    try:
        import xgboost as xgb
        from sklearn.preprocessing import MinMaxScaler

        # Feature engineering on full log-price series
        lp       = np.log(prices)
        window   = 30
        X_rows, y_rows = [], []
        for i in range(window, len(lp)):
            feat = lp[i-window:i]
            feat_eng = np.array([
                feat[-1], feat[-5], feat[-20],
                feat[-1] - feat[-5],
                feat[-1] - feat[-20],
                np.std(feat[-5:]),
                np.std(feat[-20:]),
                np.mean(feat[-5:]),
                np.mean(feat[-20:]),
                (feat[-1] - feat.min()) / (feat.max() - feat.min() + 1e-8),
            ])
            X_rows.append(feat_eng)
            y_rows.append(lp[i] - lp[i-1])   # predict log return

        X_arr = np.array(X_rows); y_arr = np.array(y_rows)
        split = int(len(X_arr) * 0.85)
        Xtr, Xte = X_arr[:split], X_arr[split:]
        ytr, yte = y_arr[:split], y_arr[split:]

        scaler_x = MinMaxScaler().fit(Xtr)
        Xtr_s = scaler_x.transform(Xtr)
        Xte_s = scaler_x.transform(Xte)

        model_xgb = xgb.XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0)
        model_xgb.fit(Xtr_s, ytr,
            eval_set=[(Xte_s, yte)], verbose=False)

        resid_std = np.std(yte - model_xgb.predict(Xte_s))

        # Rollout
        xgb_paths = []
        np.random.seed(42)
        for _ in range(500):
            current_lp = list(lp[-window:])
            path_prices = []
            for _ in range(n_days):
                feat = np.array(current_lp[-window:])
                feat_eng = np.array([[
                    feat[-1], feat[-5], feat[-20],
                    feat[-1]-feat[-5], feat[-1]-feat[-20],
                    np.std(feat[-5:]), np.std(feat[-20:]),
                    np.mean(feat[-5:]), np.mean(feat[-20:]),
                    (feat[-1]-feat.min())/(feat.max()-feat.min()+1e-8),
                ]])
                feat_s = scaler_x.transform(feat_eng)
                pred_ret = float(model_xgb.predict(feat_s)[0])
                noise    = np.random.normal(0, resid_std)
                new_lp   = current_lp[-1] + pred_ret + noise
                current_lp.append(new_lp)
                path_prices.append(np.exp(new_lp))
            xgb_paths.append(path_prices)

        xgb_arr  = np.array(xgb_paths)
        xgb_mean = np.median(xgb_arr, axis=0)
        xgb_lo   = np.percentile(xgb_arr, 10, axis=0)
        xgb_hi   = np.percentile(xgb_arr, 90, axis=0)
        results["XGBoost"] = {
            "mean": xgb_mean, "lo": xgb_lo, "hi": xgb_hi,
            "feature_imp": model_xgb.feature_importances_,
            "color": BLUE,
        }
    except Exception as e:
        results["XGBoost"] = {"error": str(e)}

    # ── 3. LSTM ───────────────────────────────────────────────────────────────
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler as MMS
        tf.get_logger().setLevel("ERROR")

        seq_len  = 60
        lp       = np.log(prices)
        scaler_l = MMS(feature_range=(0,1))
        lp_s     = scaler_l.fit_transform(lp.reshape(-1,1)).flatten()

        # Build sequences
        Xl, yl = [], []
        for i in range(seq_len, len(lp_s)):
            Xl.append(lp_s[i-seq_len:i])
            yl.append(lp_s[i])
        Xl = np.array(Xl)[..., np.newaxis]; yl = np.array(yl)

        split  = int(len(Xl)*0.85)
        Xl_tr, Xl_te = Xl[:split], Xl[split:]
        yl_tr, yl_te = yl[:split], yl[split:]

        tf.random.set_seed(42)
        model_lstm = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True,
                input_shape=(seq_len,1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        model_lstm.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                           loss="mse")
        model_lstm.fit(Xl_tr, yl_tr, epochs=20, batch_size=32,
                       validation_data=(Xl_te, yl_te), verbose=0)

        pred_te = model_lstm.predict(Xl_te, verbose=0).flatten()
        resid_s = np.std(yl_te - pred_te)

        # Rollout with dropout uncertainty (MC dropout)
        lstm_paths = []
        np.random.seed(42)
        for _ in range(300):
            seq = list(lp_s[-seq_len:])
            path_lp_s = []
            for _ in range(n_days):
                inp = np.array(seq[-seq_len:]).reshape(1,seq_len,1)
                pred_s = float(model_lstm(inp, training=True).numpy()[0,0])
                noise  = np.random.normal(0, resid_s)
                new_s  = pred_s + noise
                seq.append(new_s)
                path_lp_s.append(new_s)
            # Inverse transform
            path_lp = scaler_l.inverse_transform(
                np.array(path_lp_s).reshape(-1,1)).flatten()
            lstm_paths.append(np.exp(path_lp))

        lstm_arr  = np.array(lstm_paths)
        lstm_mean = np.median(lstm_arr, axis=0)
        lstm_lo   = np.percentile(lstm_arr, 10, axis=0)
        lstm_hi   = np.percentile(lstm_arr, 90, axis=0)
        results["LSTM"] = {
            "mean": lstm_mean, "lo": lstm_lo, "hi": lstm_hi,
            "color": ACCENT,
        }
    except Exception as e:
        results["LSTM"] = {"error": str(e)}

    # ── 4. Ensemble ───────────────────────────────────────────────────────────
    valid = {k:v for k,v in results.items() if "mean" in v}
    if len(valid) >= 2:
        weights = {"ARIMA":0.20, "XGBoost":0.35, "LSTM":0.45}
        total_w = sum(weights[k] for k in valid)
        ens_mean = sum(weights[k]/total_w * valid[k]["mean"] for k in valid)
        ens_lo   = sum(weights[k]/total_w * valid[k]["lo"]   for k in valid)
        ens_hi   = sum(weights[k]/total_w * valid[k]["hi"]   for k in valid)
        results["Ensemble"] = {
            "mean": ens_mean, "lo": ens_lo, "hi": ens_hi,
            "color": "#C084FC",
        }

    # Annual milestones (every 5 years)
    milestones = {}
    for model_name, data in results.items():
        if "mean" not in data: continue
        ms = {}
        for yr in [1,3,5,10,15,20,25]:
            idx = min(yr*252 - 1, n_days-1)
            ms[f"Y+{yr}"] = {
                "mean": round(data["mean"][idx],2),
                "lo":   round(data["lo"][idx],2),
                "hi":   round(data["hi"][idx],2),
            }
        milestones[model_name] = ms

    # Downsample for chart performance (1 point per week)
    step = 5
    chart_dates = future_dates[::step]
    for model_name, data in results.items():
        if "mean" in data:
            data["dates"]      = chart_dates
            data["mean_chart"] = data["mean"][::step]
            data["lo_chart"]   = data["lo"][::step]
            data["hi_chart"]   = data["hi"][::step]

    return {
        "models":     results,
        "milestones": milestones,
        "last_price": round(float(prices[-1]),2),
        "last_date":  str(last_date.date()),
        "history_yrs": round(len(df)/252, 1),
        "ticker":     ticker,
    }


# ╔══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════╝
def build_candlestick(df: pd.DataFrame, ticker: str, company: str) -> go.Figure:
    df_ind = add_indicators(df)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=["Price & Indicators","MACD","Volume"],
        vertical_spacing=0.04)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_ind.index, open=df_ind["Open"], high=df_ind["High"],
        low=df_ind["Low"], close=df_ind["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        name="Price"), row=1, col=1)

    # Bollinger Bands
    if "BB_upper" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_upper"],
            line=dict(color="rgba(245,166,35,0.3)",width=1), name="BB Upper",
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_lower"],
            line=dict(color="rgba(245,166,35,0.3)",width=1), name="BB Lower",
            fill="tonexty", fillcolor="rgba(245,166,35,0.04)",
            showlegend=False), row=1, col=1)

    for ma, col, lw in [("SMA_20",BLUE,1.2),("SMA_50",GOLD,1.2),("SMA_200","#C084FC",1.5)]:
        if ma in df_ind:
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind[ma],
                line=dict(color=col,width=lw), name=ma), row=1, col=1)

    # MACD
    if "MACD" in df_ind:
        colors_hist = [GREEN if v>=0 else RED for v in df_ind["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df_ind.index, y=df_ind["MACD_hist"],
            marker_color=colors_hist, name="MACD Hist", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD"],
            line=dict(color=BLUE,width=1.2), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD_signal"],
            line=dict(color=GOLD,width=1.2), name="Signal"), row=2, col=1)

    # Volume
    vol_colors = [GREEN if df_ind["Close"].iloc[i] >= df_ind["Open"].iloc[i] else RED
                  for i in range(len(df_ind))]
    fig.add_trace(go.Bar(x=df_ind.index, y=df_ind["Volume"],
        marker_color=vol_colors, name="Volume", showlegend=False), row=3, col=1)
    if "Vol_MA20" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Vol_MA20"],
            line=dict(color=GOLD,width=1.2), name="Vol MA20"), row=3, col=1)

    apply_theme(fig, height=680)
    fig.update_layout(xaxis_rangeslider_visible=False,
        title=dict(text=f"<b>{company}</b>  ·  {ticker}", font=dict(size=15,color=TEXT_SEC)))
    return fig


def build_prediction_chart(pred: dict, company: str, show_models: list) -> go.Figure:
    df_hist = fetch_history(pred["ticker"], period="5y")
    fig = go.Figure()

    # Historical
    if not df_hist.empty:
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist["Close"],
            line=dict(color=TEXT_SEC,width=1.5), name="Historical",
            hovertemplate="₹%{y:,.2f}<extra>Historical</extra>"))

    model_data = pred.get("models",{})
    for model_name in show_models:
        data = model_data.get(model_name,{})
        if "mean_chart" not in data: continue
        color = data["color"]
        # CI band
        fig.add_trace(go.Scatter(
            x=list(data["dates"]) + list(data["dates"])[::-1],
            y=list(data["hi_chart"]) + list(data["lo_chart"])[::-1],
            fill="toself", fillcolor=f"rgba({_hex_rgb(color)},0.10)",
            line=dict(width=0), name=f"{model_name} CI (P10–P90)",
            showlegend=True, hoverinfo="skip"))
        # Mean
        fig.add_trace(go.Scatter(x=data["dates"], y=data["mean_chart"],
            line=dict(color=color,width=2), name=model_name,
            hovertemplate=f"<b>{model_name}</b><br>₹%{{y:,.0f}}<br>%{{x}}<extra></extra>"))

    # Vertical "today" line
    fig.add_vline(x=pred["last_date"], line_dash="dash",
        line_color=TEXT_SEC, opacity=0.5,
        annotation_text="Today", annotation_font_size=10)

    apply_theme(fig, height=580)
    fig.update_layout(
        title=dict(text=f"<b>{company}</b> — 25-Year Price Forecast", font=dict(size=15)),
        xaxis_title="Year", yaxis_title="Price ₹",
        legend=dict(orientation="h", y=1.05, x=0, xanchor="left"),
        hovermode="x unified")
    return fig


def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    df_ind = add_indicators(df.tail(365))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI"],
        line=dict(color=ACCENT,width=2), name="RSI(14)",
        hovertemplate="RSI: %{y:.1f}<extra></extra>"))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,75,110,0.08)",
        line_width=0, annotation_text="Overbought",
        annotation_font_size=10, annotation_font_color=RED)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,200,150,0.08)",
        line_width=0, annotation_text="Oversold",
        annotation_font_size=10, annotation_font_color=GREEN)
    fig.add_hline(y=70, line_dash="dash", line_color=RED,   opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN, opacity=0.5)
    apply_theme(fig, height=260)
    fig.update_layout(title="RSI (14-day)", yaxis_range=[0,100],
        margin=dict(t=36,b=30,l=50,r=10))
    return fig


def build_returns_dist(df: pd.DataFrame) -> go.Figure:
    rets = df["Close"].pct_change().dropna() * 100
    fig  = go.Figure()
    fig.add_trace(go.Histogram(x=rets, nbinsx=80, name="Daily Returns",
        marker_color=ACCENT, opacity=0.75))
    mean_r = rets.mean(); std_r = rets.std()
    fig.add_vline(x=mean_r, line_dash="dash", line_color=GOLD,
        annotation_text=f"μ={mean_r:.2f}%")
    apply_theme(fig, height=280)
    fig.update_layout(title=f"Daily Return Distribution  |  σ={std_r:.2f}%",
        xaxis_title="Daily Return %", yaxis_title="Frequency",
        margin=dict(t=36,b=30))
    return fig


def build_correlation_heatmap(tickers: list, names: list) -> go.Figure:
    closes = {}
    for tk, nm in zip(tickers, names):
        h = fetch_history(tk, period="2y")
        if not h.empty:
            closes[nm[:15]] = h["Close"].resample("W").last()
    if len(closes) < 2:
        return go.Figure()
    df_c  = pd.DataFrame(closes).dropna()
    corr  = df_c.pct_change().dropna().corr().round(2)
    fig   = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,RED],[0.5,"#1F2937"],[1,GREEN]],
        zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}", textfont=dict(size=9),
        hovertemplate="<b>%{y} vs %{x}</b><br>r = %{z:.2f}<extra></extra>"))
    apply_theme(fig, height=500)
    fig.update_layout(title="Weekly Return Correlation Matrix",
        margin=dict(t=50,b=80,l=120,r=20),
        xaxis=dict(tickangle=-35))
    return fig


def build_sector_performance(quotes: dict) -> go.Figure:
    rows = []
    for ticker, data in quotes.items():
        if data.get("price"):
            name, _, cap_type = PHARMA_COMPANIES[ticker]
            rows.append({"name": name[:20], "change": data["change_pct"], "cap": cap_type,
                "price": data["price"], "mktcap": data.get("market_cap",0) or 0})
    if not rows: return go.Figure()
    df_sec = pd.DataFrame(rows).sort_values("change")
    colors = [GREEN if v>=0 else RED for v in df_sec["change"]]
    fig    = go.Figure(go.Bar(
        y=df_sec["name"], x=df_sec["change"], orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in df_sec["change"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:+.2f}%<extra></extra>"))
    apply_theme(fig, height=max(420, len(rows)*22))
    fig.update_layout(title="Today's Performance — All Pharma Stocks",
        xaxis_title="Change %", margin=dict(t=40,r=80))
    return fig


def build_milestone_table(milestones: dict, last_price: float) -> pd.DataFrame:
    rows = []
    for model_name, ms in milestones.items():
        for yr_label, vals in ms.items():
            mult = vals["mean"] / last_price
            rows.append({
                "Model":   model_name,
                "Horizon": yr_label,
                "Low ₹":   f"₹{vals['lo']:,.0f}",
                "Target ₹":f"₹{vals['mean']:,.0f}",
                "High ₹":  f"₹{vals['hi']:,.0f}",
                "CAGR":    f"{((vals['mean']/last_price)**(1/int(yr_label[2:]))-1)*100:.1f}%",
                "Upside":  f"{(mult-1)*100:+.0f}%",
            })
    return pd.DataFrame(rows)


def _hex_rgb(h: str) -> str:
    """Convert hex colour to 'R,G,B' string for rgba()."""
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"{r},{g},{b}"


def fmt_number(n, prefix="₹"):
    if n is None: return "N/A"
    if n >= 1e12: return f"{prefix}{n/1e12:.2f}T"
    if n >= 1e9:  return f"{prefix}{n/1e9:.2f}B"
    if n >= 1e7:  return f"{prefix}{n/1e7:.2f}Cr"
    if n >= 1e5:  return f"{prefix}{n/1e5:.2f}L"
    return f"{prefix}{n:,.2f}"


# ╔══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown(f"""
    <div style='padding:16px 0 8px 0;'>
        <div style='font-family:JetBrains Mono;font-size:1.1rem;font-weight:700;color:{ACCENT};'>
            📈 PHARMA INTEL
        </div>
        <div style='font-size:0.72rem;color:{TEXT_SEC};margin-top:2px;'>
            NSE · BSE · Live + AI Forecast
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("", [
        "🏠  Market Overview",
        "📊  Live Stock Tracker",
        "🕯️  Candlestick & Technicals",
        "🔮  25-Year AI Forecast",
        "🗞️  News & Sentiment",
        "📐  Correlation & Risk",
        "📋  Fundamentals Deep Dive",
    ], label_visibility="collapsed")

    st.divider()

    # Company selector (used across pages)
    company_labels = [f"{v[0]} ({k.replace('.NS','')})"
                      for k,v in PHARMA_COMPANIES.items()]
    selected_label = st.selectbox("Select Company", company_labels,
                                   label_visibility="visible")
    sel_idx     = company_labels.index(selected_label)
    sel_ticker  = list(PHARMA_COMPANIES.keys())[sel_idx]
    sel_name    = PHARMA_COMPANIES[sel_ticker][0]

    st.divider()
    st.markdown(f"""
    <div style='font-size:0.70rem;color:{TEXT_SEC};line-height:1.6;'>
    <b style='color:{ACCENT};'>Data Sources</b><br>
    • yfinance (NSE/BSE)<br>
    • Google Finance RSS<br>
    • Nifty Pharma Index<br><br>
    <b style='color:{ACCENT};'>Refresh Rates</b><br>
    • Prices: 5 min cache<br>
    • History: 15 min cache<br>
    • Predictions: 24h cache<br><br>
    <b style='color:{ACCENT};'>Disclaimer</b><br>
    For research only.<br>
    Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════╝
if "Overview" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>🏠 Indian Pharma Market Overview</h1>", unsafe_allow_html=True)
    now = datetime.now().strftime("%d %b %Y, %I:%M %p IST")
    st.markdown(f"<div style='font-size:0.78rem;color:{TEXT_SEC};margin-bottom:18px;'>Last updated: {now} &nbsp;·&nbsp; Prices ~5 min delayed</div>", unsafe_allow_html=True)

    # ── Index strip ───────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>MARKET INDICES</div>", unsafe_allow_html=True)
    with st.spinner("Fetching index data..."):
        idx_data = fetch_index_data()

    ci1, ci2, ci3, ci4 = st.columns(4)
    for col_obj, (label, data) in zip([ci1,ci2,ci3], idx_data.items()):
        price = data["price"]; pct = data["change_pct"]
        color = GREEN if pct>=0 else RED
        arrow = "▲" if pct>=0 else "▼"
        with col_obj:
            st.markdown(f"""
            <div class='kpi-strip'>
                <div class='kpi-strip-val'>{f'{price:,.2f}' if price else 'N/A'}</div>
                <div class='kpi-strip-sub' style='color:{color};'>{arrow} {abs(pct):.2f}%</div>
                <div class='kpi-strip-lbl'>{label}</div>
            </div>""", unsafe_allow_html=True)

    # 4th KPI: total pharma market cap
    with ci4:
        st.markdown(f"""
        <div class='kpi-strip'>
            <div class='kpi-strip-val'>~₹12.4T</div>
            <div class='kpi-strip-sub' style='color:{TEXT_SEC};'>Listed Pharma Sector</div>
            <div class='kpi-strip-lbl'>Market Cap (est.)</div>
        </div>""", unsafe_allow_html=True)

    # ── Fetch all quotes ──────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>ALL PHARMA STOCKS</div>", unsafe_allow_html=True)

    all_tickers = list(PHARMA_COMPANIES.keys())
    quotes = {}
    prog   = st.progress(0, text="Fetching live prices...")
    for i, tk in enumerate(all_tickers):
        quotes[tk] = fetch_live_quote(tk)
        prog.progress((i+1)/len(all_tickers), text=f"Loading {tk}...")
    prog.empty()

    # ── Performance bar ───────────────────────────────────────────────────────
    perf_fig = build_sector_performance(quotes)
    st.plotly_chart(perf_fig, use_container_width=True)

    # ── Scrolling ticker cards ────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>LIVE PRICE GRID</div>", unsafe_allow_html=True)

    cols_per_row = 5
    tickers_list = list(PHARMA_COMPANIES.keys())
    for row_start in range(0, len(tickers_list), cols_per_row):
        row_tickers = tickers_list[row_start:row_start+cols_per_row]
        cols = st.columns(cols_per_row)
        for col_obj, tk in zip(cols, row_tickers):
            name, bse, cap = PHARMA_COMPANIES[tk]
            q     = quotes.get(tk,{})
            price = q.get("price")
            chg   = q.get("change_pct",0)
            vol   = q.get("volume")
            color = GREEN if chg>=0 else RED
            arrow = "▲" if chg>=0 else "▼"
            chg_class = "ticker-change-up" if chg>=0 else "ticker-change-down"
            with col_obj:
                st.markdown(f"""
                <div class='ticker-card'>
                    <div class='ticker-name'>{tk.replace('.NS','')}</div>
                    <div class='ticker-price'>₹{f'{price:,.2f}' if price else '--'}</div>
                    <div class='{chg_class}'>{arrow} {abs(chg):.2f}%</div>
                    <div class='ticker-meta'>{name[:22]}</div>
                    <div class='ticker-meta'>Vol: {fmt_number(vol,'')}</div>
                </div>""", unsafe_allow_html=True)

    # ── Gainers & Losers ──────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>TOP MOVERS</div>", unsafe_allow_html=True)
    valid_quotes = {k:v for k,v in quotes.items() if v.get("price") and v.get("change_pct") is not None}
    sorted_q     = sorted(valid_quotes.items(), key=lambda x: x[1]["change_pct"])
    losers3      = sorted_q[:3]
    gainers3     = sorted_q[-3:][::-1]

    cg, cl = st.columns(2)
    with cg:
        st.markdown(f"<div style='color:{GREEN};font-weight:700;font-size:0.85rem;'>🚀 TOP GAINERS</div>", unsafe_allow_html=True)
        for tk, q in gainers3:
            name = PHARMA_COMPANIES[tk][0]
            st.markdown(f"""
            <div class='ticker-card' style='border-color:{GREEN}30;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div><div class='ticker-name'>{tk.replace('.NS','')}</div>
                    <div style='font-size:0.82rem;color:{TEXT_SEC};'>{name[:25]}</div></div>
                    <div><div class='ticker-price' style='font-size:1.2rem;'>₹{q['price']:,.2f}</div>
                    <div class='ticker-change-up'>▲ {q['change_pct']:.2f}%</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    with cl:
        st.markdown(f"<div style='color:{RED};font-weight:700;font-size:0.85rem;'>📉 TOP LOSERS</div>", unsafe_allow_html=True)
        for tk, q in losers3:
            name = PHARMA_COMPANIES[tk][0]
            st.markdown(f"""
            <div class='ticker-card' style='border-color:{RED}30;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div><div class='ticker-name'>{tk.replace('.NS','')}</div>
                    <div style='font-size:0.82rem;color:{TEXT_SEC};'>{name[:25]}</div></div>
                    <div><div class='ticker-price' style='font-size:1.2rem;'>₹{q['price']:,.2f}</div>
                    <div class='ticker-change-down'>▼ {abs(q['change_pct']):.2f}%</div></div>
                </div>
            </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE STOCK TRACKER
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Live Stock" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>📊 {sel_name}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.78rem;color:{TEXT_SEC};margin-bottom:12px;'>{sel_ticker} &nbsp;·&nbsp; NSE &nbsp;·&nbsp; BSE {PHARMA_COMPANIES[sel_ticker][1]}</div>", unsafe_allow_html=True)

    with st.spinner("Fetching live data..."):
        q = fetch_live_quote(sel_ticker)

    if q.get("price") is None:
        st.error("Could not fetch live data. Market may be closed or ticker unavailable.")
        st.stop()

    price = q["price"]; chg = q["change_pct"]; chg_abs = q["change"]
    color = GREEN if chg >= 0 else RED; arrow = "▲" if chg>=0 else "▼"

    # ── Hero price ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;
                padding:28px 36px;margin-bottom:20px;'>
        <div style='font-family:JetBrains Mono;font-size:3rem;font-weight:700;color:{TEXT_PRI};line-height:1;'>
            ₹{price:,.2f}
        </div>
        <div style='font-family:JetBrains Mono;font-size:1.3rem;color:{color};margin-top:6px;'>
            {arrow} ₹{abs(chg_abs):.2f} ({abs(chg):.2f}%) today
        </div>
        <div style='font-size:0.78rem;color:{TEXT_SEC};margin-top:8px;'>
            Prev Close: ₹{q.get('prev_close','N/A')} &nbsp;·&nbsp;
            Open: ₹{q.get('open') or 'N/A'} &nbsp;·&nbsp;
            {PHARMA_COMPANIES[sel_ticker][2]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row 1 ─────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis1 = [
        ("Day High", f"₹{q.get('high') or 'N/A'}"),
        ("Day Low",  f"₹{q.get('low') or 'N/A'}"),
        ("Volume",   fmt_number(q.get("volume"),"") or "N/A"),
        ("Market Cap", fmt_number(q.get("market_cap"))),
        ("Avg Volume", fmt_number(q.get("avg_volume"),"") or "N/A"),
    ]
    for col_obj, (label, val) in zip([c1,c2,c3,c4,c5], kpis1):
        with col_obj:
            st.markdown(f"""<div class='kpi-strip'>
                <div class='kpi-strip-val' style='font-size:1.1rem;'>{val}</div>
                <div class='kpi-strip-lbl'>{label}</div></div>""", unsafe_allow_html=True)

    # ── KPI row 2: fundamentals ───────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    pe  = q.get("pe_ratio");  pb   = q.get("pb_ratio")
    eps = q.get("eps");       beta = q.get("beta")
    roe = q.get("roe");       pm   = q.get("profit_margin")
    kpis2 = [
        ("P/E Ratio",      f"{pe:.2f}" if pe else "N/A"),
        ("P/B Ratio",      f"{pb:.2f}" if pb else "N/A"),
        ("EPS (TTM)",      f"₹{eps:.2f}" if eps else "N/A"),
        ("Beta",           f"{beta:.2f}" if beta else "N/A"),
        ("ROE",            f"{roe*100:.1f}%" if roe else "N/A"),
        ("Profit Margin",  f"{pm*100:.1f}%" if pm else "N/A"),
    ]
    for col_obj, (label, val) in zip([c1,c2,c3,c4,c5,c6], kpis2):
        with col_obj:
            st.markdown(f"""<div class='kpi-strip'>
                <div class='kpi-strip-val' style='font-size:1.0rem;'>{val}</div>
                <div class='kpi-strip-lbl'>{label}</div></div>""", unsafe_allow_html=True)

    # ── 52-week range ─────────────────────────────────────────────────────────
    wh = q.get("week52_high"); wl = q.get("week52_low")
    if wh and wl and price:
        pct_pos = (price - wl) / (wh - wl) * 100
        st.markdown(f"<div class='section-hdr'>52-WEEK RANGE</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;padding:16px 24px;'>
            <div style='display:flex;justify-content:space-between;font-size:0.8rem;color:{TEXT_SEC};margin-bottom:8px;'>
                <span>52W Low: <b style='color:{GREEN};'>₹{wl:,.2f}</b></span>
                <span>Current: <b style='color:{TEXT_PRI};'>₹{price:,.2f}</b> ({pct_pos:.0f}% of range)</span>
                <span>52W High: <b style='color:{RED};'>₹{wh:,.2f}</b></span>
            </div>
            <div style='background:{BORDER};border-radius:4px;height:10px;position:relative;'>
                <div style='background:linear-gradient(90deg,{GREEN},{GOLD},{RED});
                            border-radius:4px;height:100%;width:{pct_pos:.0f}%;'></div>
                <div style='position:absolute;top:-4px;left:{pct_pos:.0f}%;
                            width:18px;height:18px;background:{TEXT_PRI};border-radius:50%;
                            transform:translateX(-50%);border:2px solid {DARK_BG};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Price history chart ───────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>PRICE HISTORY</div>", unsafe_allow_html=True)
    period_sel = st.radio("Period", ["1mo","3mo","6mo","1y","2y","5y"],
        horizontal=True, index=3, key="hist_period")

    with st.spinner("Loading history..."):
        df_hist = fetch_history(sel_ticker, period=period_sel)

    if not df_hist.empty:
        fig_area = go.Figure()
        close_arr = df_hist["Close"].values
        color_line = GREEN if close_arr[-1] >= close_arr[0] else RED
        fig_area.add_trace(go.Scatter(x=df_hist.index, y=df_hist["Close"],
            mode="lines", line=dict(color=color_line, width=2),
            fill="tozeroy", fillcolor=f"rgba({_hex_rgb(color_line)},0.08)",
            name="Close"))
        apply_theme(fig_area, height=360)
        fig_area.update_layout(hovermode="x unified",
            xaxis_title="", yaxis_title="Price ₹",
            xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_area, use_container_width=True)

    # ── Volume bar ────────────────────────────────────────────────────────────
    if not df_hist.empty:
        fig_vol = go.Figure()
        vol_c = [GREEN if df_hist["Close"].iloc[i]>=df_hist["Close"].iloc[max(0,i-1)] else RED
                 for i in range(len(df_hist))]
        fig_vol.add_trace(go.Bar(x=df_hist.index, y=df_hist["Volume"],
            marker_color=vol_c, name="Volume"))
        apply_theme(fig_vol, height=200)
        fig_vol.update_layout(margin=dict(t=20,b=30), title="Volume",
            xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_vol, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: CANDLESTICK & TECHNICALS
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Candlestick" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>🕯️ Technical Analysis — {sel_name}</h1>", unsafe_allow_html=True)

    period_sel = st.radio("Period", ["3mo","6mo","1y","2y"], horizontal=True, key="candle_period", index=1)
    with st.spinner("Building chart..."):
        df_c = fetch_history(sel_ticker, period=period_sel)

    if df_c.empty:
        st.error("No data available.")
        st.stop()

    # Main candlestick
    fig_candle = build_candlestick(df_c, sel_ticker, sel_name)
    st.plotly_chart(fig_candle, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        rsi_fig = build_rsi_chart(df_c)
        st.plotly_chart(rsi_fig, use_container_width=True)
    with col2:
        ret_fig = build_returns_dist(df_c)
        st.plotly_chart(ret_fig, use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>TECHNICAL SUMMARY</div>", unsafe_allow_html=True)
    df_ind = add_indicators(df_c)
    last   = df_ind.iloc[-1]

    rsi_val  = last.get("RSI", None)
    sma20    = last.get("SMA_20", None)
    sma50    = last.get("SMA_50", None)
    sma200   = last.get("SMA_200", None)
    macd_val = last.get("MACD", None)
    sig_val  = last.get("MACD_signal", None)
    close    = last["Close"]

    signals = []
    if rsi_val:
        if rsi_val > 70:   signals.append(("RSI", f"{rsi_val:.1f}", "Overbought", RED))
        elif rsi_val < 30: signals.append(("RSI", f"{rsi_val:.1f}", "Oversold",   GREEN))
        else:              signals.append(("RSI", f"{rsi_val:.1f}", "Neutral",    GOLD))
    if sma20 and sma50:
        if sma20 > sma50:  signals.append(("MA Cross", "20 > 50", "Bullish", GREEN))
        else:              signals.append(("MA Cross", "20 < 50", "Bearish", RED))
    if close and sma200:
        if close > sma200: signals.append(("vs SMA200", f"₹{close:.0f} > ₹{sma200:.0f}", "Bullish", GREEN))
        else:              signals.append(("vs SMA200", f"₹{close:.0f} < ₹{sma200:.0f}", "Bearish", RED))
    if macd_val and sig_val:
        if macd_val > sig_val: signals.append(("MACD", "Above signal", "Bullish", GREEN))
        else:                  signals.append(("MACD", "Below signal", "Bearish", RED))

    cols = st.columns(len(signals))
    for col_obj, (indicator, val, label, color) in zip(cols, signals):
        with col_obj:
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {color};'>
                <div class='kpi-strip-val' style='color:{color};font-size:0.9rem;'>{val}</div>
                <div class='kpi-strip-sub' style='color:{color};'>{label}</div>
                <div class='kpi-strip-lbl'>{indicator}</div>
            </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: 25-YEAR AI FORECAST
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Forecast" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>🔮 25-Year AI Forecast — {sel_name}</h1>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='alert-box'>
    🧠 Running <b>ARIMA</b>, <b>XGBoost</b>, <b>LSTM</b> and <b>Ensemble</b> models on full historical data.
    Predictions use Monte Carlo simulation (300–500 paths) to produce confidence intervals.
    First run may take 3–5 minutes. Results cached for 24 hours.
    </div>
    """, unsafe_allow_html=True)

    show_models = st.multiselect("Show models",
        ["ARIMA","XGBoost","LSTM","Ensemble"],
        default=["Ensemble","XGBoost"], key="model_select")

    with st.spinner(f"Running ML models for {sel_name}... (first run ~3 min)"):
        pred = run_all_predictions(sel_ticker, years_ahead=25)

    if "error" in pred:
        st.error(f"Prediction failed: {pred['error']}")
        st.stop()

    # ── Model availability badges ─────────────────────────────────────────────
    badge_html = ""
    for m_name, m_data in pred["models"].items():
        status = "✅" if "mean" in m_data else "❌"
        err    = m_data.get("error","")
        badge_html += f'<span class="model-badge">{status} {m_name}</span>'
        if err: badge_html += f'<span style="font-size:0.7rem;color:{RED};"> ({err[:40]})</span>'
    st.markdown(badge_html, unsafe_allow_html=True)

    # ── Hero KPIs: ensemble year targets ─────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>PRICE TARGETS (ENSEMBLE)</div>", unsafe_allow_html=True)
    ens_ms = pred.get("milestones",{}).get("Ensemble",{})
    last_p = pred["last_price"]
    if ens_ms:
        cols_ms = st.columns(5)
        for col_obj, yr in zip(cols_ms, ["Y+1","Y+3","Y+5","Y+10","Y+25"]):
            if yr in ens_ms:
                v = ens_ms[yr]
                upside = (v["mean"]/last_p-1)*100
                color  = GREEN if upside>0 else RED
                cagr   = ((v["mean"]/last_p)**(1/int(yr[2:]))-1)*100
                with col_obj:
                    st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {color};'>
                        <div class='kpi-strip-val'>₹{v['mean']:,.0f}</div>
                        <div class='kpi-strip-sub' style='color:{color};'>{upside:+.0f}%  CAGR {cagr:.1f}%</div>
                        <div class='kpi-strip-lbl'>{yr.replace('Y+','')} Year Target</div>
                    </div>""", unsafe_allow_html=True)

    # ── Forecast chart ────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>PRICE FORECAST CHART</div>", unsafe_allow_html=True)
    if show_models:
        fig_pred = build_prediction_chart(pred, sel_name, show_models)
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Select at least one model above.")

    # ── Milestone table ───────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>FULL MILESTONE TABLE</div>", unsafe_allow_html=True)
    ms_df = build_milestone_table(pred.get("milestones",{}), last_p)
    if not ms_df.empty:
        model_filter = st.multiselect("Filter by model", ms_df["Model"].unique().tolist(),
            default=ms_df["Model"].unique().tolist(), key="ms_filter")
        st.dataframe(
            ms_df[ms_df["Model"].isin(model_filter)].reset_index(drop=True),
            use_container_width=True, hide_index=True, height=400)

    # ── Model metadata ────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>MODEL DETAILS</div>", unsafe_allow_html=True)
    tab_a, tab_x, tab_l, tab_e = st.tabs(["ARIMA","XGBoost","LSTM","Ensemble"])
    with tab_a:
        st.markdown(f"""
        **ARIMA(2,0,2)** fitted on log-returns of last 1,500 trading days.
        500 Monte Carlo paths with residual noise. Confidence: P10–P90.
        AIC: `{pred['models'].get('ARIMA',{}).get('aic','N/A')}`
        - Best for: Short-term mean-reversion, stationary patterns
        - Weakness: Assumes linear dynamics; variance grows rapidly over 25 years
        """)
    with tab_x:
        st.markdown("""
        **XGBoost Gradient Boosting Regressor** on 10 engineered lag features
        (1-day, 5-day, 20-day log returns, volatility, momentum).
        400 trees, depth 5. 85/15 train/test split. 500 MC rollout paths.
        - Best for: Non-linear regime detection, trend continuation
        - Weakness: Can over-extrapolate in very long horizons
        """)
    with tab_l:
        st.markdown("""
        **LSTM (64→32 units)** with 60-day lookback window on scaled log-prices.
        MC Dropout (training=True at inference) for uncertainty estimation.
        20 epochs, batch 32. 300 dropout paths.
        - Best for: Capturing sequential patterns and long-term dependencies
        - Weakness: Computationally heavy; uncertainty grows non-linearly
        """)
    with tab_e:
        st.markdown("""
        **Weighted Ensemble** — LSTM (45%), XGBoost (35%), ARIMA (20%).
        Weights assigned by typical out-of-sample accuracy on equity time series.
        Confidence bands are weighted averages of individual model P10/P90.
        - Recommended: Reduces single-model bias; most robust long-horizon estimate
        """)
    st.markdown(f"""
    <div class='alert-box' style='border-color:{RED};background:rgba(255,75,110,0.08);'>
    ⚠️ <b>Important:</b> 25-year forecasts have very wide uncertainty bounds by year 10+.
    No model can reliably predict markets 25 years ahead. These are <b>probabilistic scenarios</b>,
    not guarantees. Use for research and scenario planning only.
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════╝
elif "News" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>🗞️ News & Sentiment — {sel_name}</h1>", unsafe_allow_html=True)

    with st.spinner("Fetching news..."):
        news = fetch_news_sentiment(sel_name)

    if not news:
        st.warning("No news found. Google News RSS may be temporarily unavailable.")
    else:
        pos_count = sum(1 for n in news if n["sentiment"]=="positive")
        neg_count = sum(1 for n in news if n["sentiment"]=="negative")
        neu_count = sum(1 for n in news if n["sentiment"]=="neutral")
        total     = len(news)

        # Sentiment summary
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            overall = "Bullish" if pos_count>neg_count else ("Bearish" if neg_count>pos_count else "Neutral")
            col_ov  = GREEN if overall=="Bullish" else (RED if overall=="Bearish" else GOLD)
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {col_ov};'>
                <div class='kpi-strip-val' style='color:{col_ov};font-size:1.0rem;'>{overall}</div>
                <div class='kpi-strip-lbl'>Overall Sentiment</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {GREEN};'>
                <div class='kpi-strip-val' style='color:{GREEN};'>{pos_count}</div>
                <div class='kpi-strip-lbl'>Positive Headlines</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {RED};'>
                <div class='kpi-strip-val' style='color:{RED};'>{neg_count}</div>
                <div class='kpi-strip-lbl'>Negative Headlines</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {GOLD};'>
                <div class='kpi-strip-val' style='color:{GOLD};'>{neu_count}</div>
                <div class='kpi-strip-lbl'>Neutral Headlines</div></div>""", unsafe_allow_html=True)

        # Sentiment donut
        fig_sent = go.Figure(go.Pie(
            labels=["Positive","Neutral","Negative"],
            values=[pos_count, neu_count, neg_count],
            marker_colors=[GREEN, GOLD, RED],
            hole=0.65,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} headlines<extra></extra>"))
        fig_sent.add_annotation(text=f"<b>{total}</b><br>Headlines",
            x=0.5, y=0.5, font_size=14, showarrow=False,
            font_color=TEXT_PRI)
        apply_theme(fig_sent, height=300)
        fig_sent.update_layout(showlegend=False, margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_sent, use_container_width=True)

        # Headlines
        st.markdown(f"<div class='section-hdr'>RECENT HEADLINES</div>", unsafe_allow_html=True)
        for item in news:
            s = item["sentiment"]
            pill_cls  = "pill-pos" if s=="positive" else ("pill-neg" if s=="negative" else "pill-neu")
            pill_text = "🟢 Positive" if s=="positive" else ("🔴 Negative" if s=="negative" else "⚪ Neutral")
            st.markdown(f"""
            <div class='ticker-card' style='margin-bottom:6px;'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:12px;'>
                    <div style='font-size:0.87rem;color:{TEXT_PRI};line-height:1.4;'>{item['title']}</div>
                    <div class='{pill_cls}' style='white-space:nowrap;'>{pill_text}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Sector sentiment comparison ────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>COMPARE SENTIMENT — TOP 6 PHARMA STOCKS</div>", unsafe_allow_html=True)
    top6 = list(PHARMA_COMPANIES.keys())[:6]
    sent_rows = []
    with st.spinner("Fetching sector sentiment..."):
        for tk in top6:
            nm  = PHARMA_COMPANIES[tk][0]
            nws = fetch_news_sentiment(nm)
            if nws:
                pos = sum(1 for n in nws if n["sentiment"]=="positive")
                neg = sum(1 for n in nws if n["sentiment"]=="negative")
                sent_rows.append({"Company":nm[:20],"Positive":pos,"Negative":neg,
                    "Score": pos-neg, "Total":len(nws)})
    if sent_rows:
        sent_df = pd.DataFrame(sent_rows).sort_values("Score")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(y=sent_df["Company"], x=sent_df["Positive"],
            name="Positive", marker_color=GREEN, orientation="h"))
        fig_cmp.add_trace(go.Bar(y=sent_df["Company"], x=-sent_df["Negative"],
            name="Negative", marker_color=RED, orientation="h"))
        apply_theme(fig_cmp, height=320)
        fig_cmp.update_layout(barmode="overlay", xaxis_title="Headline Count",
            title="Sentiment Balance (→ positive, ← negative)")
        st.plotly_chart(fig_cmp, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATION & RISK
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Correlation" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>📐 Correlation & Risk Analysis</h1>", unsafe_allow_html=True)

    cap_filter = st.multiselect("Filter by Market Cap",
        ["Large Cap","Mid Cap","Small Cap"],
        default=["Large Cap","Mid Cap"])
    filtered_tickers = [k for k,v in PHARMA_COMPANIES.items() if v[2] in cap_filter]
    filtered_names   = [PHARMA_COMPANIES[k][0] for k in filtered_tickers]

    with st.spinner("Computing correlation matrix (fetching 2Y weekly data)..."):
        corr_fig = build_correlation_heatmap(filtered_tickers, filtered_names)
    st.plotly_chart(corr_fig, use_container_width=True)

    # ── Risk metrics ──────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>RISK METRICS — {sel_name}</div>", unsafe_allow_html=True)
    with st.spinner("Computing risk metrics..."):
        df_risk = fetch_history(sel_ticker, period="3y")

    if not df_risk.empty:
        rets    = df_risk["Close"].pct_change().dropna()
        ann_ret = (1 + rets.mean())**252 - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd  = ((df_risk["Close"] / df_risk["Close"].cummax()) - 1).min()
        var_95  = np.percentile(rets, 5)
        pos_days= (rets > 0).mean() * 100

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        risk_kpis = [
            ("Ann. Return",   f"{ann_ret*100:.1f}%",  GREEN if ann_ret>0 else RED),
            ("Ann. Volatility",f"{ann_vol*100:.1f}%", GOLD),
            ("Sharpe Ratio",  f"{sharpe:.2f}",         GREEN if sharpe>1 else (GOLD if sharpe>0 else RED)),
            ("Max Drawdown",  f"{max_dd*100:.1f}%",    RED),
            ("VaR (95%, 1D)", f"{var_95*100:.2f}%",    RED),
            ("% Positive Days",f"{pos_days:.1f}%",     GREEN if pos_days>50 else RED),
        ]
        for col_obj, (label, val, color) in zip([c1,c2,c3,c4,c5,c6], risk_kpis):
            with col_obj:
                st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {color};'>
                    <div class='kpi-strip-val' style='color:{color};font-size:1.0rem;'>{val}</div>
                    <div class='kpi-strip-lbl'>{label}</div></div>""", unsafe_allow_html=True)

        # Rolling volatility
        roll_vol = rets.rolling(30).std() * np.sqrt(252) * 100
        fig_vol  = go.Figure()
        fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol,
            fill="tozeroy", fillcolor=f"rgba({_hex_rgb(GOLD)},0.12)",
            line=dict(color=GOLD, width=2), name="30D Rolling Volatility"))
        apply_theme(fig_vol, height=280)
        fig_vol.update_layout(title="30-Day Rolling Annualised Volatility (%)",
            yaxis_title="Volatility %", margin=dict(t=36,b=30))
        st.plotly_chart(fig_vol, use_container_width=True)

        # Drawdown chart
        dd = (df_risk["Close"] / df_risk["Close"].cummax()) - 1
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd*100,
            fill="tozeroy", fillcolor=f"rgba({_hex_rgb(RED)},0.15)",
            line=dict(color=RED, width=1.5), name="Drawdown"))
        apply_theme(fig_dd, height=260)
        fig_dd.update_layout(title="Underwater / Drawdown Chart",
            yaxis_title="Drawdown %", margin=dict(t=36,b=30))
        st.plotly_chart(fig_dd, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE: FUNDAMENTALS DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════╝
elif "Fundamentals" in page:
    st.markdown(f"<h1 style='font-size:1.8rem;font-weight:700;'>📋 Fundamentals — {sel_name}</h1>", unsafe_allow_html=True)

    with st.spinner("Loading financials..."):
        q    = fetch_live_quote(sel_ticker)
        fins = fetch_financials(sel_ticker)

    # ── Valuation scorecard ───────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>VALUATION & PROFITABILITY</div>", unsafe_allow_html=True)

    metrics = {
        "P/E Ratio":      (q.get("pe_ratio"),  10, 30,  "lower"),
        "P/B Ratio":      (q.get("pb_ratio"),   1,  5,  "lower"),
        "EPS (TTM ₹)":   (q.get("eps"),         None, None, "higher"),
        "Dividend Yield": (q.get("div_yield",0) or 0, 0, 0.05, "higher"),
        "Beta":           (q.get("beta"),        0.8, 1.2, "neutral"),
        "Debt/Equity":    (q.get("debt_equity"), 0, 100, "lower"),
        "ROE %":          ((q.get("roe") or 0)*100, 10, 25, "higher"),
        "Profit Margin %":((q.get("profit_margin") or 0)*100, 5, 20, "higher"),
    }

    cols = st.columns(4)
    for idx, (label, (val, lo, hi, direction)) in enumerate(metrics.items()):
        with cols[idx % 4]:
            if val is None:
                display = "N/A"; color = TEXT_SEC
            else:
                display = f"{val:.2f}"
                if lo is not None and hi is not None:
                    if direction == "lower":
                        color = GREEN if val<lo else (GOLD if val<hi else RED)
                    elif direction == "higher":
                        color = GREEN if val>hi else (GOLD if val>lo else RED)
                    else:
                        color = GREEN if lo<=val<=hi else RED
                else:
                    color = GREEN if val > 0 else RED
            st.markdown(f"""<div class='kpi-strip' style='border-top:3px solid {color};'>
                <div class='kpi-strip-val' style='color:{color};font-size:1.1rem;'>{display}</div>
                <div class='kpi-strip-lbl'>{label}</div></div>""", unsafe_allow_html=True)

    # ── Peer comparison table ─────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>PEER COMPARISON</div>", unsafe_allow_html=True)
    peer_tickers = list(PHARMA_COMPANIES.keys())[:10]
    peer_rows = []
    with st.spinner("Fetching peer data..."):
        for tk in peer_tickers:
            pq = fetch_live_quote(tk)
            if pq.get("price"):
                peer_rows.append({
                    "Company":     PHARMA_COMPANIES[tk][0][:22],
                    "Price ₹":     f"₹{pq['price']:,.2f}",
                    "Change %":    f"{pq['change_pct']:+.2f}%",
                    "Mkt Cap":     fmt_number(pq.get("market_cap")),
                    "P/E":         f"{pq['pe_ratio']:.1f}" if pq.get("pe_ratio") else "N/A",
                    "P/B":         f"{pq['pb_ratio']:.1f}" if pq.get("pb_ratio") else "N/A",
                    "ROE %":       f"{pq['roe']*100:.1f}%" if pq.get("roe") else "N/A",
                    "52W High ₹":  f"₹{pq['week52_high']:,.0f}" if pq.get("week52_high") else "N/A",
                    "52W Low ₹":   f"₹{pq['week52_low']:,.0f}" if pq.get("week52_low") else "N/A",
                    "Segment":     PHARMA_COMPANIES[tk][2],
                })
    if peer_rows:
        peer_df = pd.DataFrame(peer_rows)
        st.dataframe(peer_df, use_container_width=True, hide_index=True, height=380)

    # ── P/E bubble ────────────────────────────────────────────────────────────
    if peer_rows:
        st.markdown(f"<div class='section-hdr'>VALUATION BUBBLE MAP</div>", unsafe_allow_html=True)
        raw_rows = []
        for tk in peer_tickers:
            pq = fetch_live_quote(tk)
            mc = pq.get("market_cap") or 0
            pe = pq.get("pe_ratio")
            pb = pq.get("pb_ratio")
            roe= pq.get("roe")
            if pq.get("price") and pe and pb:
                raw_rows.append({
                    "name": PHARMA_COMPANIES[tk][0][:18],
                    "pe": pe, "pb": pb,
                    "mktcap": mc/1e9,
                    "roe": (roe or 0)*100,
                    "change": pq["change_pct"],
                })
        if raw_rows:
            bub_df = pd.DataFrame(raw_rows)
            fig_bub = px.scatter(bub_df, x="pe", y="pb",
                size="mktcap", color="roe",
                text="name", size_max=50,
                color_continuous_scale=[[0,RED],[0.5,GOLD],[1,GREEN]],
                labels={"pe":"P/E Ratio","pb":"P/B Ratio","mktcap":"Mkt Cap (B)","roe":"ROE %"},
                hover_data={"pe":":.1f","pb":":.1f","mktcap":":.1f","roe":":.1f"})
            fig_bub.update_traces(textposition="top center", textfont_size=9)
            apply_theme(fig_bub, height=520)
            fig_bub.update_layout(title="P/E vs P/B — Size = Market Cap | Color = ROE%")
            st.plotly_chart(fig_bub, use_container_width=True)
