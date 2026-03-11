# 📈 Indian Pharma Stock Intelligence Platform

Live NSE/BSE tracker + 25-year AI price forecast for all major Indian pharmaceutical companies.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_stock.txt

# 2. Run
streamlit run pharma_stock_tracker.py
```

Opens at `http://localhost:8501`

---

## 📦 What's Inside

### 7 Dashboard Pages

| Page | What it shows |
|---|---|
| 🏠 Market Overview | All 30 pharma stocks live, index strip (Nifty Pharma, Sensex, Nifty 50), top gainers/losers |
| 📊 Live Stock Tracker | Hero price, P/E, P/B, EPS, Beta, ROE, 52-week range slider, price history, volume |
| 🕯️ Candlestick & Technicals | Full OHLCV candlestick, Bollinger Bands, SMA 20/50/200, MACD, RSI, returns distribution |
| 🔮 25-Year AI Forecast | ARIMA + XGBoost + LSTM + Ensemble with Monte Carlo confidence intervals, milestone table |
| 🗞️ News & Sentiment | Google News RSS headlines, keyword sentiment scoring, sector sentiment comparison |
| 📐 Correlation & Risk | Weekly return correlation heatmap, Sharpe, VaR, Max Drawdown, rolling volatility |
| 📋 Fundamentals Deep Dive | Valuation scorecard, peer comparison table, P/E vs P/B bubble map |

### 30 Listed Companies Covered

**Large Cap:** Sun Pharma, Dr. Reddy's, Cipla, Divi's Labs, Mankind, Torrent, Lupin, Aurobindo, Alkem, Biocon, Zydus

**Mid Cap:** IPCA Labs, Glenmark, Ajanta, Granules, Natco, Abbott India, Pfizer India, Sanofi, GSK, Laurus Labs, Eris, JB Chemicals, Piramal

**Small Cap:** Kopran, Solara, Strides, Marksans, Caplin Point, Sequent Scientific

### AI Prediction Models

| Model | Architecture | Paths | Horizon |
|---|---|---|---|
| ARIMA(2,0,2) | Statistical on log-returns | 500 MC | Best: 1–3Y |
| XGBoost | 10 lag features, 400 trees | 500 MC | Best: 3–10Y |
| LSTM | 64→32 units, 60-day lookback | 300 MC Dropout | Best: 5–15Y |
| Ensemble | LSTM 45% + XGBoost 35% + ARIMA 20% | Weighted | Full 25Y |

---

## ⚠️ Important Notes

- **Data delay**: yfinance provides ~15-minute delayed data during market hours. Not suitable for HFT.
- **Market hours**: NSE/BSE open 9:15 AM – 3:30 PM IST, Mon–Fri. Prices frozen after close.
- **Predictions**: 25-year forecasts are probabilistic scenarios with very wide uncertainty bands beyond year 5. For research use only. **Not financial advice.**
- **First forecast run**: Takes 3–5 minutes per company (LSTM training). Subsequent runs use 24-hour cache.

---

## 🔄 Cache TTLs

| Data | Cache | Reason |
|---|---|---|
| Live prices | 5 minutes | Balance freshness vs API limits |
| Price history | 15 minutes | Intraday updates rare |
| Financials | 30 minutes | Quarterly data |
| Index data | 10 minutes | Slightly faster refresh |
| News | 1 hour | Google RSS rate limits |
| ML Predictions | 24 hours | Expensive compute |

---

## 🛠️ Extending the App

**Add more companies**: Edit `PHARMA_COMPANIES` dict at top of file. Use `TICKER.NS` for NSE, `TICKER.BO` for BSE.

**Real-time websocket**: Replace `yfinance` with Zerodha Kite API (`pip install kiteconnect`) for true real-time.

**Better sentiment**: Replace Google RSS with a proper news API (NewsAPI.org, Refinitiv) for richer NLP.

**SARIMA**: Swap the ARIMA block for `SARIMAX` from statsmodels to add seasonal components.

**Deploy to Streamlit Cloud**:
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repo, set main file to `pharma_stock_tracker.py`
4. Add `requirements_stock.txt`

---

## 📊 Data Sources

- **Price data**: Yahoo Finance via `yfinance` (NSE suffix `.NS`)
- **News**: Google News RSS (free, no API key needed)
- **Indices**: Nifty Pharma (`^CNXPHARMA`), BSE Sensex (`^BSESN`), Nifty 50 (`^NSEI`)
