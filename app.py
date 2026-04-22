import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt

# ================= CONFIG =================
st.set_page_config(page_title="AgroData CM", layout="wide")

# ================= ANIMATION ACCUEIL =================
with st.spinner("Chargement de AgroData CM..."):
    time.sleep(2)

# ================= STYLE =================
st.markdown("""
<style>
body {background-color: #0e1117;}
h1 {color: #00c853; text-align:center;}
.stMetric {background-color: #1c1f26; padding:15px; border-radius:10px;}
button[kind="primary"] {background-color:#00c853; color:white;}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<h1>🌱 AgroData CM</h1>
<p style='text-align:center;'>Analyse intelligente des données agricoles 🇨🇲</p>
""", unsafe_allow_html=True)

# ================= NAVIGATION =================
menu = st.sidebar.radio("📌 Navigation", ["🏠 Accueil", "📊 Dashboard"])

# ================= PAGE ACCUEIL =================
if menu == "🏠 Accueil":

    st.success("Bienvenue sur une plateforme intelligente d'analyse agricole 🇨🇲")

    st.markdown("""
    ## 🎯 Objectif
    Transformer les données agricoles en décisions utiles.

    ## ⚙️ Fonctionnalités
    - 📥 Collecte des données
    - 📊 Visualisation simple
    - 🧠 Analyse automatique
    - 🏆 Comparaison des régions
    - 📥 Export PDF
    """)

    st.info("👉 Utilisez le menu à gauche pour accéder au Dashboard")

# ================= DASHBOARD =================
else:

    st.title("📊 Dashboard intelligent")

    # ================= DATABASE =================
    conn = sqlite3.connect('database.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS data
                 (region TEXT, prix REAL, production REAL, superficie REAL)''')

    regions = ["Centre","Littoral","Ouest","Nord","Sud"]

    # ================= DONNÉES AUTO =================
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

    # ================= ANALYSE AUTOMATIQUE =================
    region_stats = df.groupby("region")["production"].mean().sort_values(ascending=False)

    best_region = region_stats.index[0]
    top3 = region_stats.head(3)

    st.success(f"🧠 Analyse : La région {best_region} est la plus productive.")

    st.subheader("🏆 Top 3 des régions")
    for i, (region, value) in enumerate(top3.items(), start=1):
        st.write(f"{i}. {region} → {round(value,2)}")

    # ================= FILTRE =================
    st.markdown("### 🔎 Filtrer les données")
    colf1, colf2 = st.columns(2)

    region_filter = colf1.selectbox("🌍 Région", ["Toutes"] + regions)
    prix_max = colf2.slider("💰 Prix max", int(df['prix'].min()), int(df['prix'].max()), int(df['prix'].max()))

    if region_filter != "Toutes":
        df = df[df['region'] == region_filter]

    df = df[df['prix'] <= prix_max]

    # ================= KPI =================
    col1, col2, col3 = st.columns(3)

    col1.metric("📈 Production", round(df['production'].mean(),2))
    col2.metric("💰 Prix", round(df['prix'].mean(),2))
    col3.metric("🌱 Superficie", round(df['superficie'].mean(),2))

    # ================= GRAPHIQUES =================
    st.subheader("📊 Visualisation")

    fig1 = px.bar(df, x="region", y="production", color="region",
                  title="Production par région")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="prix", title="Distribution des prix")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(df, x="superficie", y="production", color="region",
                      title="Superficie vs Production")
    st.plotly_chart(fig3, use_container_width=True)

    # ================= CARTE =================
    st.subheader("🗺️ Carte du Cameroun")

    coords = {
        "Centre": [3.848, 11.502],
        "Littoral": [4.051, 9.767],
        "Ouest": [5.473, 10.417],
        "Nord": [9.301, 13.397],
        "Sud": [2.889, 11.157]
    }

    m = folium.Map(location=[5, 12], zoom_start=6)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=coords[row['region']],
            radius=5,
            color="green",
            fill=True
        ).add_to(m)

    st_folium(m, width=800)

    # ================= RÉGRESSION =================
    st.subheader("📉 Régression linéaire")

    model = LinearRegression()
    model.fit(df[['superficie']], df['production'])

    coef = model.coef_[0]
    st.info(f"Relation superficie-production : {round(coef,2)}")

    # ================= PDF =================
    def generate_pdf():
        plt.figure()
        df.groupby("region")["production"].mean().plot(kind='bar')
        plt.title("Production par région")
        plt.savefig("graph.png")
        plt.close()

        doc = SimpleDocTemplate("rapport.pdf")
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("Rapport AgroData CM", styles['Title']))
        content.append(Paragraph(f"Production moyenne: {round(df['production'].mean(),2)}", styles['Normal']))
        content.append(Image("graph.png"))

        doc.build(content)

    if st.button("📥 Générer rapport PDF"):
        generate_pdf()
        with open("rapport.pdf", "rb") as f:
            st.download_button("⬇️ Télécharger PDF", f, file_name="rapport.pdf")

    # ================= ANIMATION =================
    if st.button("⚡ Lancer analyse avancée"):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)
        st.success("✔ Analyse terminée")
