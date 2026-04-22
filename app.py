import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
from datetime import datetime

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score

# ================= CONFIG =================
st.set_page_config(page_title="INF232 EC2 - Smart Health Analyzer", layout="wide")

# ================= CSS =================
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# ================= DATABASE =================
class DB:
    def __init__(self):
        self.conn = sqlite3.connect("data.db", check_same_thread=False)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE,
            nom TEXT,
            age INTEGER,
            genre TEXT,
            poids REAL,
            taille REAL,
            tension_sys INTEGER,
            tension_dia INTEGER,
            glycemie REAL,
            date_obs TIMESTAMP
        )
        """)

    def add(self, data):
        try:
            self.conn.execute("""
            INSERT INTO patients (
                patient_id, nom, age, genre, poids, taille,
                tension_sys, tension_dia, glycemie, date_obs
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """, tuple(data.values()))
            self.conn.commit()
            return True
        except Exception as e:
            st.error(e)
            return False

    def all(self):
        return pd.read_sql("SELECT * FROM patients", self.conn)

db = DB()

# ================= UTILS =================
def prep(df):
    df = df.copy()
    df['date_obs'] = pd.to_datetime(df['date_obs'])
    df['imc'] = df['poids'] / (df['taille']**2)
    df['cat'] = pd.cut(df['imc'], [0,18.5,25,30,40],
                       labels=["Insuffisant","Normal","Surpoids","Obésité"])
    return df

def forecast_time(df):
    df = df.copy()
    df_time = df.groupby(df['date_obs'].dt.date)['tension_sys'].mean().reset_index()
    df_time.columns = ['date', 'tension']
    df_time['t'] = np.arange(len(df_time))

    model = LinearRegression().fit(df_time[['t']], df_time['tension'])

    future_t = np.arange(len(df_time), len(df_time)+7).reshape(-1,1)
    preds = model.predict(future_t)

    future_dates = pd.date_range(start=df_time['date'].iloc[-1], periods=8)[1:]

    df_future = pd.DataFrame({'date': future_dates, 'tension': preds})

    return df_time, df_future

# ================= HEADER =================
st.title("🚀 INF 232 EC2 - Smart Health Analyzer")
st.write("Collecte, analyse descriptive et machine learning")

# ================= TABS =================
tab0, tab1, tab2, tab3 = st.tabs([
    "🚀 Dashboard Pro",
    "📥 Collecte",
    "📊 Analyse",
    "🔮 Machine Learning"
])

df = db.all()

# ================= DASHBOARD =================
with tab0:
    if df.empty:
        st.warning("Aucune donnée")
    else:
        df = prep(df)

        # KPI
        col1, col2, col3, col4 = st.columns(4)

        risk = ((df['tension_sys']/140) + (df['glycemie']/1.2) + (df['imc']/25))/3

        col1.metric("Patients", len(df))
        col2.metric("IMC moyen", round(df['imc'].mean(),1))
        col3.metric("Tension moyenne", round(df['tension_sys'].mean(),0))
        col4.metric("Risque moyen", round(risk.mean(),2))

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(px.histogram(df, x="imc"), use_container_width=True)

        with col2:
            st.plotly_chart(px.pie(df, names="genre"), use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(px.scatter(df, x="imc", y="tension_sys", color="cat", trendline="ols"), use_container_width=True)

        with col4:
            st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=True), use_container_width=True)

        st.markdown("---")

        # Forecast
        st.subheader("📈 Prévisions temporelles")

        df_time, df_future = forecast_time(df)

        fig = px.line(df_time, x='date', y='tension')

        fig.add_scatter(
            x=df_future['date'],
            y=df_future['tension'],
            mode='lines+markers',
            name="Prévision",
            line=dict(dash='dash')
        )

        st.plotly_chart(fig, use_container_width=True)

# ================= COLLECTE =================
with tab1:
    with st.form("form"):
        col1, col2 = st.columns(2)

        with col1:
            pid = st.text_input("ID")
            nom = st.text_input("Nom")
            age = st.number_input("Age", 1, 120)
            genre = st.selectbox("Genre", ["M","F"])

        with col2:
            poids = st.number_input("Poids", 30.0, 200.0)
            taille = st.number_input("Taille", 1.0, 2.5)
            ts = st.number_input("Tension Sys", 80, 220)
            td = st.number_input("Tension Dia", 40, 130)
            glyc = st.number_input("Glycémie", 0.5, 5.0)

        if st.form_submit_button("Ajouter"):
            if taille <= 0:
                st.error("Taille invalide")
            else:
                db.add({
                    "patient_id": pid,
                    "nom": nom,
                    "age": age,
                    "genre": genre,
                    "poids": poids,
                    "taille": taille,
                    "tension_sys": ts,
                    "tension_dia": td,
                    "glycemie": glyc,
                    "date_obs": datetime.now()
                })
                st.success("Ajouté")

# ================= ANALYSE =================
with tab2:
    if df.empty:
        st.warning("Pas de données")
    else:
        df = prep(df)

        st.write(df.describe())

        st.plotly_chart(px.box(df, y="imc"), use_container_width=True)

        st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=True), use_container_width=True)

# ================= ML =================
with tab3:
    if len(df) < 10:
        st.warning("Ajouter plus de données")
    else:
        df = prep(df)

        X = df[['age','poids','taille','glycemie']]
        y = df['tension_sys']

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

        # Regression simple
        st.subheader("Régression simple")
        model = LinearRegression().fit(df[['imc']], y)
        st.write("R²:", r2_score(y, model.predict(df[['imc']])))

        # Regression multiple
        st.subheader("Régression multiple")
        model2 = LinearRegression().fit(X_tr, y_tr)
        st.write("R²:", r2_score(y_te, model2.predict(X_te)))

        # PCA
        st.subheader("PCA")
        scaler = StandardScaler()
        comp = PCA(n_components=2).fit_transform(scaler.fit_transform(X))
        st.plotly_chart(px.scatter(x=comp[:,0], y=comp[:,1]), use_container_width=True)

        # Classification
        st.subheader("Classification")
        y_class = LabelEncoder().fit_transform(df['cat'].astype(str))
        clf = LogisticRegression(max_iter=200).fit(X_tr, y_class[:len(X_tr)])
        st.write("Accuracy:", clf.score(X_te, y_class[len(X_tr):]))

        # KMeans
        st.subheader("KMeans")
        X_scaled = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=3).fit(X_scaled)
        st.write("Silhouette:", silhouette_score(X_scaled, km.labels_))
