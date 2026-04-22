import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
import numpy as np
import time

# ================= CONFIG =================
st.set_page_config(page_title="AgroData CM", layout="wide")

# ================= ANIMATION ACCUEIL =================
with st.spinner("Chargement de AgroData CM..."):
    time.sleep(2)

st.markdown("""
<h1 style='text-align:center;'>🌱 AgroData CM</h1>
<p style='text-align:center;'>Analyse intelligente des données agricoles 🇨🇲</p>
""", unsafe_allow_html=True)

# ================= NAV =================
menu = st.sidebar.radio("📌 Navigation", ["🏠 Accueil", "📊 Dashboard"])

# ================= ACCUEIL =================
if menu == "🏠 Accueil":
    st.success("Bienvenue sur une plateforme intelligente d'analyse agricole 🇨🇲")

    st.markdown("""
    ### 🎯 Objectif
    Transformer les données agricoles en décisions utiles.

    ### ⚙️ Fonctionnalités
    - 📊 Graphiques simples
    - 🗺️ Carte interactive
    - 🧠 Analyse automatique
    - 📥 Rapport PDF
    """)

    st.info("👉 Accédez au Dashboard pour explorer les données")

# ================= DASHBOARD =================
else:
    conn = sqlite3.connect('database.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS data
                 (region TEXT, prix REAL, production REAL, superficie REAL)''')

    regions = ["Centre","Littoral","Ouest","Nord","Sud"]

    # DATA AUTO
    df_check = pd.read_sql_query("SELECT * FROM data", conn)
    if df_check.empty:
        np.random.seed(42)
        for _ in range(100):
            region = np.random.choice(regions)
            prix = np.random.randint(500, 5000)
            superficie = np.random.randint(1, 50)
            production = superficie * np.random.uniform(1.5, 3.5)
            conn.execute("INSERT INTO data VALUES (?,?,?,?)",
                         (region, prix, production, superficie))
        conn.commit()

    df = pd.read_sql_query("SELECT * FROM data", conn)

    # ================= ANALYSE REGION =================
    region_stats = df.groupby("region")["production"].mean().sort_values(ascending=False)

    best_region = region_stats.index[0]
    top3 = region_stats.head(3)

    st.success(f"🧠 Analyse automatique : La région {best_region} est la plus productive.")

    st.subheader("🏆 Top 3 des régions")
    for i, (region, value) in enumerate(top3.items(), start=1):
        st.write(f"{i}. {region} → {round(value,2)}")
        st.success("✔ Analyse terminée")
