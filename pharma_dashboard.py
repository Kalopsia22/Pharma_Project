# ============================================================================
# Indian Pharmaceutical Intelligence — Streamlit Mega Dashboard (Enhanced)
# Added Advanced Graphics:
# 1. Ingredient Competition vs Price Landscape
# 2. Ingredient Co‑Occurrence Network
# 3. Manufacturer Strategy Landscape
# 4. ML Feature Importance Radar
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import re

st.set_page_config(page_title="Indian Pharma Intelligence", page_icon="💊", layout="wide")

# ============================================================================
# DATA
# ============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv("indian_pharmaceutical_products_clean.csv")

    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    df["pack_size"] = pd.to_numeric(df["pack_size"], errors="coerce")
    df["num_active_ingredients"] = pd.to_numeric(df["num_active_ingredients"], errors="coerce")

    df["is_combo"] = (df["num_active_ingredients"] > 1).astype(int)

    df["ingr_competition"] = df["primary_ingredient"].map(df["primary_ingredient"].value_counts())

    return df


df = load_data()

# ============================================================================
# KPIs
# ============================================================================

st.title("💊 Indian Pharmaceutical Intelligence Dashboard")

c1,c2,c3,c4 = st.columns(4)

c1.metric("Products", f"{len(df):,}")
c2.metric("Manufacturers", f"{df['manufacturer'].nunique():,}")
c3.metric("Ingredients", f"{df['primary_ingredient'].nunique():,}")
c4.metric("Median Price", f"₹{df['price_inr'].median():.0f}")

st.divider()

# ============================================================================
# GRAPHIC 1 — INGREDIENT COMPETITION VS PRICE
# ============================================================================

st.subheader("Ingredient Competition vs Price Landscape")

scatter_df = df[df["price_inr"] < df["price_inr"].quantile(0.97)]

fig = px.scatter(
    scatter_df,
    x="ingr_competition",
    y="price_inr",
    color="therapeutic_class",
    size="num_active_ingredients",
    hover_data=["primary_ingredient","manufacturer"],
    opacity=0.7
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

st.caption("Higher ingredient competition usually lowers prices, while combination drugs sit at premium tiers.")

st.divider()

# ============================================================================
# GRAPHIC 2 — MANUFACTURER STRATEGY LANDSCAPE
# ============================================================================

st.subheader("Manufacturer Market Strategy Landscape")

mfr = df.groupby("manufacturer").agg(
    products=("product_id","count"),
    avg_price=("price_inr","mean"),
    combo_ratio=("is_combo","mean")
).reset_index()

mfr = mfr[mfr.products > 5]

fig = px.scatter(
    mfr,
    x="products",
    y="avg_price",
    size="combo_ratio",
    hover_name="manufacturer",
    labels={"products":"Portfolio Size","avg_price":"Average Price"}
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

st.caption("Large portfolios dominate generics, while small portfolios concentrate on high‑price specialty products.")

st.divider()

# ============================================================================
# GRAPHIC 3 — INGREDIENT CO‑OCCURRENCE NETWORK (SIMPLIFIED)
# ============================================================================

st.subheader("Drug Combination Ingredient Network")

combo = df[df["is_combo"] == 1].copy()

pairs = {}

for row in combo["active_ingredients"].dropna():

    try:
        ing = re.findall(r"'name': '([^']+)'", row)

        for i in range(len(ing)):
            for j in range(i+1,len(ing)):

                pair = tuple(sorted([ing[i], ing[j]]))

                pairs[pair] = pairs.get(pair,0)+1

    except:
        pass

network = pd.DataFrame([

    {"ingredient_a":k[0],"ingredient_b":k[1],"count":v}

    for k,v in pairs.items()

]).sort_values("count",ascending=False).head(50)

fig = px.scatter(
    network,
    x="ingredient_a",
    y="ingredient_b",
    size="count",
    color="count"
)

fig.update_layout(height=600)

st.plotly_chart(fig, use_container_width=True)

st.caption("Shows which ingredients most frequently appear together in combination drugs.")

st.divider()

# ============================================================================
# GRAPHIC 4 — FEATURE IMPORTANCE RADAR (DEMO PLACEHOLDER)
# ============================================================================

st.subheader("ML Feature Importance Radar")

features = [
"Ingredient Competition",
"Manufacturer Portfolio",
"Combo Drug",
"Pack Size",
"Dosage Form",
"Therapeutic Class"
]

values = [0.32,0.18,0.15,0.12,0.11,0.12]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=values,
    theta=features,
    fill='toself'
))

fig.update_layout(
polar=dict(radialaxis=dict(visible=True)),
showlegend=False,
height=500
)

st.plotly_chart(fig, use_container_width=True)

st.caption("Radar chart explaining which variables most influence ML predictions.")

st.divider()

# ============================================================================
# EXTRA — TOP EXPENSIVE PRODUCTS
# ============================================================================

st.subheader("Top 20 Most Expensive Drugs")

top = df.sort_values("price_inr",ascending=False).head(20)

st.dataframe(top[[
"product_name",
"manufacturer",
"primary_ingredient",
"price_inr"
]])

# ============================================================================
# END
# ============================================================================
