import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
import tempfile
from datetime import datetime
from fpdf import FPDF
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MediData AI Expert | Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CSS (3X-UI / GLASSMORPHISM) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #f0f2f5 0%, #e0e5ec 100%);
    }

    /* Glassmorphism Card */
    .st-emotion-cache-12w0qpk, .glass-card {
        background: rgba(255, 255, 255, 0.7) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(5px) !important;
        -webkit-backdrop-filter: blur(5px) !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
    }

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Custom Metric Style */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #2563eb !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px !important;
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATABASE ENGINE ---
class DataManager:
    def __init__(self, db_name="medical_expert.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS patients 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      nom TEXT, age INTEGER, poids REAL, 
                      taille REAL, tension REAL, glycemie REAL, cholest REAL,
                      date_enc TEXT)''')
        # Add some initial dummy data if empty for demonstration
        c.execute("SELECT count(*) FROM patients")
        if c.fetchone()[0] == 0:
            initial_data = [
                ('Jean Dupont', 45, 80.5, 1.75, 130, 1.05, 2.1, '2023-10-01'),
                ('Marie Curie', 34, 62.0, 1.65, 115, 0.95, 1.8, '2023-10-02'),
                ('Luc Petit', 67, 88.0, 1.70, 155, 1.40, 2.5, '2023-10-03'),
                ('Sarah Connor', 29, 58.0, 1.68, 110, 0.88, 1.9, '2023-10-04'),
                ('Paul Atreides', 52, 75.0, 1.80, 142, 1.25, 2.3, '2023-10-05'),
                ('Ellen Ripley', 41, 68.0, 1.72, 125, 1.10, 2.0, '2023-10-06')
            ]
            c.executemany("INSERT INTO patients (nom, age, poids, taille, tension, glycemie, cholest, date_enc) VALUES (?,?,?,?,?,?,?,?)", initial_data)
        self.conn.commit()

    def get_data(self):
        df = pd.read_sql("SELECT * FROM patients", self.conn)
        if not df.empty:
            df['IMC'] = df['poids'] / (df['taille']**2)
            df['Risque'] = (df['tension'] > 140).astype(int)
        return df

    def add_patient(self, data):
        c = self.conn.cursor()
        query = "INSERT INTO patients (nom, age, poids, taille, tension, glycemie, cholest, date_enc) VALUES (?,?,?,?,?,?,?,?)"
        c.execute(query, (*data, datetime.now().strftime("%Y-%m-%d")))
        self.conn.commit()

db = DataManager()

# --- APP LOGIC ---
def main():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2864/2864230.png", width=80)
        st.title("MediData AI")
        st.markdown("---")
        menu = st.radio("NAVIGATION", 
            ["📊 Dashboard & Collecte", "📈 Regressions Cliniques", "📉 Réduction Dim (PCA)", "🧠 IA & Classification", "📄 Rapport Expert Export"],
            index=0)
        st.markdown("---")
        st.caption("INF 232 EC2 - Science des Données")

    df = db.get_data()

    if menu == "📊 Dashboard & Collecte":
        render_dashboard(df)
    elif menu == "📈 Regressions Cliniques":
        render_regressions(df)
    elif menu == "📉 Réduction Dim (PCA)":
        render_pca(df)
    elif menu == "🧠 IA & Classification":
        render_ai(df)
    elif menu == "📄 Rapport Expert Export":
        render_report(df)

def render_dashboard(df):
    st.title("📊 Dashboard & Collecte de Données")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📝 Nouveau Patient")
        with st.form("patient_form", clear_on_submit=True):
            nom = st.text_input("Nom Complet")
            age = st.slider("Âge", 1, 100, 40)
            col_a, col_b = st.columns(2)
            poids = col_a.number_input("Poids (kg)", 30.0, 200.0, 70.0)
            taille = col_b.number_input("Taille (m)", 1.0, 2.5, 1.75)
            tension = st.number_input("Tension Systolique", 80, 220, 120)
            glycemie = st.number_input("Glycémie (g/L)", 0.5, 3.0, 1.0)
            cholest = st.number_input("Cholestérol (g/L)", 1.0, 5.0, 2.0)
            
            if st.form_submit_button("Enregistrer le Patient"):
                if nom:
                    db.add_patient((nom, age, poids, taille, tension, glycemie, cholest))
                    st.success(f"Patient {nom} ajouté avec succès !")
                    st.rerun()
                else:
                    st.error("Veuillez saisir un nom.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("📈 Aperçu Global de la Population")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Patients", len(df))
        m2.metric("Moyenne Tension", f"{df['tension'].mean():.1f}")
        m3.metric("Âge Moyen", f"{df['age'].mean():.1f} ans")
        
        fig = px.scatter(df, x="age", y="tension", size="IMC", color="cholest",
                         hover_name="nom", title="Distribution Tension vs Âge (Taille = IMC)",
                         color_continuous_scale="Viridis", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Voir la base de données brute"):
            st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

def render_regressions(df):
    st.title("📈 Modélisation par Régression")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("1. Régression Linéaire Simple")
        st.info("Prédire la Tension en fonction de l'Âge")
        
        X = df[['age']]
        y = df['tension']
        model = LinearRegression().fit(X, y)
        df['pred_tension'] = model.predict(X)
        
        fig = px.scatter(df, x='age', y='tension', trendline="ols", 
                         title="Relation Âge / Tension")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Équation :** Tension = {model.coef_[0]:.2f} × Age + {model.intercept_:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("2. Régression Multiple")
        st.info("Facteurs d'influence sur le Risque Global (Score Combiné)")
        
        # Création d'un score de santé synthétique
        df['Score_Sante'] = (df['tension'] * 0.4) + (df['glycemie'] * 20) + (df['IMC'] * 0.5)
        X_mult = df[['age', 'poids', 'glycemie', 'cholest']]
        y_mult = df['Score_Sante']
        
        model_m = LinearRegression().fit(X_mult, y_mult)
        importance = pd.DataFrame({'Variable': X_mult.columns, 'Coeff': model_m.coef_})
        
        fig_imp = px.bar(importance, x='Variable', y='Coeff', color='Coeff',
                         title="Impact des variables sur le Score de Risque")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.write(f"**Précision du modèle (R²) :** {model_m.score(X_mult, y_mult):.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

def render_pca(df):
    st.title("📉 Réduction de Dimensions (PCA)")
    st.markdown("""
    Cette technique permet de visualiser des données complexes (5+ variables) dans un espace 3D 
    tout en conservant le maximum d'information (variance).
    """)
    
    features = ['age', 'poids', 'tension', 'glycemie', 'cholest', 'IMC']
    x = df[features]
    x_scaled = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=3)
    components = pca.fit_transform(x_scaled)
    
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
    pca_df['nom'] = df['nom']
    pca_df['age'] = df['age']
    
    fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='age',
                        hover_name='nom', title="Projection PCA 3D des profils patients",
                        color_continuous_scale="Plasma")
    st.plotly_chart(fig, use_container_width=True)
    
    var_exp = pca.explained_variance_ratio_.sum() * 100
    st.success(f"L'analyse PCA conserve **{var_exp:.2f}%** de l'information originale.")

def render_ai(df):
    st.title("🧠 Intelligence Artificielle")
    
    tab1, tab2 = st.tabs(["Classification (Supervisée)", "Clustering (Non-Supervisée)"])
    
    with tab1:
        st.subheader("Prédiction du Risque d'Hypertension")
        X = df[['age', 'IMC', 'glycemie', 'cholest']]
        y = df['Risque']
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        
        st.write("### Testeur de Diagnostic IA")
        c1, c2, c3, c4 = st.columns(4)
        t_age = c1.number_input("Âge Test", 18, 90, 50)
        t_imc = c2.number_input("IMC Test", 15.0, 45.0, 25.0)
        t_gly = c3.number_input("Glycémie Test", 0.5, 3.0, 1.0)
        t_cho = c4.number_input("Cholestérol Test", 1.0, 5.0, 2.0)
        
        if st.button("Lancer la Prédiction IA"):
            res = clf.predict([[t_age, t_imc, t_gly, t_cho]])
            prob = clf.predict_proba([[t_age, t_imc, t_gly, t_cho]])[0][1]
            
            if res[0] == 1:
                st.error(f"Résultat : RISQUE ÉLEVÉ ({prob*100:.1f}%)")
            else:
                st.success(f"Résultat : RISQUE FAIBLE ({(1-prob)*100:.1f}%)")

    with tab2:
        st.subheader("Segmentation Automatique des Patients (K-Means)")
        k_val = st.slider("Nombre de segments (Clusters)", 2, 5, 3)
        
        X_clust = df[['age', 'tension', 'IMC']]
        kmeans = KMeans(n_clusters=k_val, n_init=10, random_state=42).fit(X_clust)
        df['Segment'] = kmeans.labels_.astype(str)
        
        fig = px.scatter(df, x='age', y='tension', color='Segment', size='IMC',
                         symbol='Segment', title="Groupes de patients identifiés par l'IA")
        st.plotly_chart(fig, use_container_width=True)
        st.info("L'IA regroupe les patients ayant des caractéristiques biométriques similaires.")

def render_report(df):
    st.title("📄 Génération de Rapport Expert")
    st.write("Générez un bilan clinique complet incluant tous les graphiques d'analyse.")

    if st.button("🚀 Générer le Rapport PDF Complet"):
        with st.spinner("Analyse des données et rendu des graphiques..."):
            try:
                # Setup PDF
                pdf = FPDF()
                pdf.add_page()
                
                # Title
                pdf.set_font("Arial", 'B', 24)
                pdf.set_text_color(37, 99, 235)
                pdf.cell(0, 20, "RAPPORT MEDICAL EXPERT AI", ln=True, align='C')
                
                pdf.set_font("Arial", '', 12)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 10, f"Date du rapport : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='R')
                pdf.ln(10)

                # Stats section
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "1. Synthese de la Cohorte", ln=True)
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 10, f"Nombre total de patients analyses : {len(df)}\n"
                                     f"Moyenne IMC : {df['IMC'].mean():.2f}\n"
                                     f"Tension Arterielle Moyenne : {df['tension'].mean():.1f} mmHg\n"
                                     f"Taux de risque hypertension detecte : {(df['Risque'].sum()/len(df))*100:.1f}%")
                
                # Temporary directory for images
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Chart 1: Distribution
                    fig1 = px.histogram(df, x="tension", title="Distribution de la Tension")
                    img1_path = os.path.join(tmpdir, "chart1.png")
                    fig1.write_image(img1_path, engine="kaleido")
                    pdf.image(img1_path, x=10, y=None, w=180)
                    
                    pdf.add_page()
                    # Chart 2: Regression
                    fig2 = px.scatter(df, x='age', y='tension', trendline="ols")
                    img2_path = os.path.join(tmpdir, "chart2.png")
                    fig2.write_image(img2_path, engine="kaleido")
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, "2. Analyse de Regression (Age vs Tension)", ln=True)
                    pdf.image(img2_path, x=10, y=None, w=180)

                    # Table
                    pdf.ln(10)
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, "3. Liste Detaillee des Patients", ln=True)
                    pdf.set_font("Arial", '', 10)
                    
                    # Table Header
                    pdf.set_fill_color(200, 220, 255)
                    pdf.cell(40, 10, "Nom", 1, 0, 'C', True)
                    pdf.cell(20, 10, "Age", 1, 0, 'C', True)
                    pdf.cell(30, 10, "Tension", 1, 0, 'C', True)
                    pdf.cell(30, 10, "IMC", 1, 0, 'C', True)
                    pdf.cell(30, 10, "Risque", 1, 1, 'C', True)
                    
                    for _, row in df.iterrows():
                        pdf.cell(40, 10, str(row['nom']), 1)
                        pdf.cell(20, 10, str(int(row['age'])), 1)
                        pdf.cell(30, 10, f"{row['tension']:.1f}", 1)
                        pdf.cell(30, 10, f"{row['IMC']:.2f}", 1)
                        pdf.cell(30, 10, "OUI" if row['Risque']==1 else "NON", 1, 1)

                # Output
                pdf_bytes = pdf.output(dest='S')
                st.download_button(
                    label="💾 Télécharger le Bilan PDF",
                    data=bytes(pdf_bytes),
                    file_name=f"Rapport_MediData_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
                st.success("Rapport généré avec succès !")
            except Exception as e:
                st.error(f"Erreur lors de la génération : {str(e)}")
                st.info("Note : Assurez-vous que le package 'kaleido' est bien installé pour l'export des images.")

if __name__ == "__main__":
    main()
