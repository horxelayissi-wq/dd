"""
╔════════════════════════════════════════════════════════════════════╗
║          PROGRAMME INF 232 EC2 - ANALYSE DE DONNÉES              ║
║    Application de Collecte et Analyse Descriptive de Données     ║
╚════════════════════════════════════════════════════════════════════╝

Fonctionnalités:
✓ Régression Linéaire (Simple & Multiple)
✓ Techniques de Réduction de Dimensionnalité (PCA)
✓ Classification Supervisée (Logistic Regression, Decision Tree)
✓ Classification Non-Supervisée (K-Means)
✓ Génération PDF Complète avec Graphes
✓ Interface Moderne et Intuitive
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
from datetime import datetime
import os
import io

# Machine Learning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score, 
                             confusion_matrix, classification_report, silhouette_score)

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION STREAMLIT
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="INF 232 EC2 - Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ Fichier style.css non trouvé")


# ═══════════════════════════════════════════════════════════════════
# DATABASE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

class DatabaseManager:
    """Gestion robuste de la base de données SQLite"""
    
    def __init__(self, db_name='analytics_data.db'):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        """Initialiser la base de données"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id TEXT NOT NULL UNIQUE,
                        nom TEXT NOT NULL,
                        age INTEGER NOT NULL,
                        genre TEXT NOT NULL,
                        poids REAL NOT NULL,
                        taille REAL NOT NULL,
                        tension_sys INTEGER NOT NULL,
                        tension_dia INTEGER NOT NULL,
                        glycemie REAL NOT NULL,
                        date_obs TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        except Exception as e:
            st.error(f"❌ Erreur initialisation BD: {str(e)}")

    def add_record(self, data):
        """Ajouter un enregistrement"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO records 
                    (patient_id, nom, age, genre, poids, taille, 
                     tension_sys, tension_dia, glycemie, date_obs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', tuple(data.values()))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            return False

    def get_all_data(self):
        """Récupérer tous les données"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                return pd.read_sql('SELECT * FROM records ORDER BY date_obs DESC', conn)
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            return pd.DataFrame()

    def get_patient_data(self, patient_name):
        """Récupérer les données d'un patient"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                query = 'SELECT * FROM records WHERE nom = ? ORDER BY date_obs DESC'
                return pd.read_sql(query, conn, params=(patient_name,))
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            return pd.DataFrame()

    def delete_patient(self, patient_id):
        """Supprimer un patient"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM records WHERE patient_id = ?', (patient_id,))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            return False

@st.cache_resource
def init_db():
    return DatabaseManager()

db = init_db()


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def prepare_data(df):
    """Préparer les données"""
    if df.empty:
        return None
    df_prep = df.copy()
    df_prep['imc'] = df_prep['poids'] / (df_prep['taille'] ** 2)
    df_prep['imc_category'] = pd.cut(df_prep['imc'], 
                                      bins=[0, 18.5, 25, 30, 40],
                                      labels=['Insuffisant', 'Normal', 'Surpoids', 'Obésité'])
    return df_prep

def linear_regression_simple(df):
    """Régression linéaire simple: IMC vs Tension"""
    try:
        X = df[['imc']].values
        y = df['tension_sys'].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        return {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'coef': model.coef_[0],
            'intercept': model.intercept_
        }
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None

def linear_regression_multiple(df):
    """Régression linéaire multiple"""
    try:
        X = df[['age', 'poids', 'taille', 'glycemie']].values
        y = df['tension_sys'].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        return {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'coef': model.coef_
        }
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None

def pca_analysis(df):
    """PCA: Réduction de dimensionnalité"""
    try:
        X = df[['age', 'poids', 'taille', 'tension_sys', 'tension_dia', 'glycemie']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        return {
            'variance_ratio': pca.explained_variance_ratio_,
            'X_pca': X_pca,
            'total_variance': sum(pca.explained_variance_ratio_)
        }
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None

def classification_supervised(df):
    """Classification supervisée"""
    try:
        y = LabelEncoder().fit_transform(df['imc_category'].astype(str))
        X = df[['age', 'poids', 'taille', 'tension_sys', 'glycemie']].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        lr = LogisticRegression(max_iter=200, random_state=42)
        lr.fit(X_train, y_train)
        acc_lr = lr.score(X_test, y_test)
        
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        acc_dt = dt.score(X_test, y_test)
        
        return {'lr_accuracy': acc_lr, 'dt_accuracy': acc_dt}
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None

def classification_unsupervised(df):
    """K-Means clustering"""
    try:
        X = df[['age', 'poids', 'taille', 'tension_sys', 'glycemie']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, clusters)
        return {'silhouette': score, 'clusters': clusters, 'X_scaled': X_scaled}
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None

def predict_with_confidence_interval(df, age, poids, taille, glycemie, confidence=0.95):
    """
    Prédire la tension systolique avec intervalle de confiance
    """
    try:
        from scipy import stats
        
        # Préparer les données
        X = df[['age', 'poids', 'taille', 'glycemie']].values
        y = df['tension_sys'].values
        
        # Entraîner le modèle
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédiction
        X_new = np.array([[age, poids, taille, glycemie]])
        y_pred = model.predict(X_new)[0]
        
        # Calcul des résidus et de l'erreur standard
        y_pred_all = model.predict(X)
        residuals = y - y_pred_all
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        
        # Erreur standard de la prédiction
        n = len(df)
        X_mean = np.mean(X, axis=0)
        X_diff = X_new[0] - X_mean
        
        # Matrix X avec intercept
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        
        X_new_with_intercept = np.array([1] + list(X_new[0]))
        se_prediction = rmse * np.sqrt(1 + X_new_with_intercept @ XtX_inv @ X_new_with_intercept.T)
        
        # Intervalle de confiance
        alpha = 1 - confidence
        t_value = stats.t.ppf(1 - alpha/2, n - 5)
        margin_error = t_value * se_prediction
        
        ci_lower = y_pred - margin_error
        ci_upper = y_pred + margin_error
        
        return {
            'prediction': y_pred,
            'lower_bound': ci_lower,
            'upper_bound': ci_upper,
            'margin_error': margin_error,
            'rmse': rmse,
            'r2': r2_score(y, y_pred_all)
        }
    except Exception as e:
        st.error(f"❌ Erreur prédiction: {str(e)}")
        return None

# ═══════════════════════════════════════════════════════════════════
# PDF GENERATION
# ═══════════════════════════════════════════════════════════════════

class ReportPDF(FPDF):
    """Génération de rapports PDF"""
    def header(self):
        self.set_fill_color(14, 165, 233)
        self.rect(0, 0, 210, 20, 'F')
        self.set_font('Arial', 'B', 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 15, 'INF 232 EC2 - Rapport d\'Analyse', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 0, 'C')

    def add_section(self, title):
        self.ln(5)
        self.set_font('Arial', 'B', 11)
        self.set_text_color(14, 165, 233)
        self.cell(0, 8, title, 0, 1)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

def generate_pdf(df, df_prep, analysis_results):
    """Générer un PDF complet"""
    try:
        pdf = ReportPDF()
        pdf.add_page()
        
        # Page 1: Résumé
        pdf.add_section("📊 RÉSUMÉ EXÉCUTIF")
        pdf.set_font('Arial', '', 10)
        summary = f"""
Nombre de patients: {len(df['nom'].unique())}
Total enregistrements: {len(df)}
Période: {str(df['date_obs'].min())[:10]} à {str(df['date_obs'].max())[:10]}

STATISTIQUES DESCRIPTIVES:
- Âge moyen: {df_prep['age'].mean():.1f} ± {df_prep['age'].std():.1f} ans
- Poids moyen: {df_prep['poids'].mean():.1f} ± {df_prep['poids'].std():.1f} kg
- Taille moyenne: {df_prep['taille'].mean():.2f} ± {df_prep['taille'].std():.2f} m
- IMC moyen: {df_prep['imc'].mean():.1f} ± {df_prep['imc'].std():.1f}
- Tension moyenne: {df_prep['tension_sys'].mean():.1f} ± {df_prep['tension_sys'].std():.1f} mmHg
- Glycémie moyenne: {df_prep['glycemie'].mean():.2f} ± {df_prep['glycemie'].std():.2f} g/L
"""
        pdf.multi_cell(0, 4, summary)
        
        # Analyses
        pdf.add_page()
        pdf.add_section("📈 ANALYSES")
        pdf.set_font('Arial', '', 9)
        
        if analysis_results:
            analyses_text = ""
            
            if 'regression' in analysis_results and analysis_results['regression']:
                reg = analysis_results['regression']
                analyses_text += f"RÉGRESSION LINÉAIRE:\n"
                analyses_text += f"- Simple (IMC→Tension): R²={reg['simple']['r2']:.4f}, RMSE={reg['simple']['rmse']:.2f}\n"
                analyses_text += f"- Multiple: R²={reg['multiple']['r2']:.4f}, RMSE={reg['multiple']['rmse']:.2f}\n\n"
            
            if 'pca' in analysis_results and analysis_results['pca']:
                pca = analysis_results['pca']
                analyses_text += f"PCA: Variance PC1={pca['variance_ratio'][0]:.2%}, PC2={pca['variance_ratio'][1]:.2%}\n\n"
            
            if 'classification_sup' in analysis_results and analysis_results['classification_sup']:
                clf = analysis_results['classification_sup']
                analyses_text += f"CLASSIFICATION SUPERVISÉE:\n"
                analyses_text += f"- Logistic Regression: {clf['lr_accuracy']:.2%}\n"
                analyses_text += f"- Decision Tree: {clf['dt_accuracy']:.2%}\n\n"
            
            if 'classification_unsup' in analysis_results and analysis_results['classification_unsup']:
                clf = analysis_results['classification_unsup']
                analyses_text += f"K-MEANS: Silhouette Score = {clf['silhouette']:.4f}\n"
            
            pdf.multi_cell(0, 4, analyses_text)
        
        # Données
        pdf.add_page()
        pdf.add_section("📋 DONNÉES BRUTES")
        pdf.set_font('Arial', 'B', 8)
        pdf.set_fill_color(240, 240, 240)
        
        headers = ['Nom', 'Âge', 'Poids', 'IMC', 'Tension', 'Glyc.']
        w = [32, 18, 20, 18, 25, 20]
        
        for h, width in zip(headers, w):
            pdf.cell(width, 6, h, 1, 0, 'C', True)
        pdf.ln()
        
        pdf.set_font('Arial', '', 7)
        for _, row in df.head(20).iterrows():
            imc = row['poids'] / (row['taille']**2)
            tension = f"{row['tension_sys']}/{row['tension_dia']}"
            pdf.cell(w[0], 6, str(row['nom'])[:15], 1)
            pdf.cell(w[1], 6, str(row['age']), 1, 0, 'C')
            pdf.cell(w[2], 6, f"{row['poids']:.0f}", 1, 0, 'C')
            pdf.cell(w[3], 6, f"{imc:.1f}", 1, 0, 'C')
            pdf.cell(w[4], 6, tension, 1, 0, 'C')
            pdf.cell(w[5], 6, f"{row['glycemie']:.1f}", 1, 1, 'C')
        
        return pdf.output()
    except Exception as e:
        st.error(f"❌ Erreur PDF: {str(e)}")
        return None


# ═══════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════

def main():
    # Sidebar Menu
    st.sidebar.markdown("---")
    st.sidebar.title("🎯 NAVIGATION")
    
    menu = st.sidebar.radio("", [
        "🏠 Dashboard",
        "📝 Nouvelle Entrée",
        "� Prédictions",
        "�📊 Analyses",
        "👤 Profil Patient",
        "⚙️ Outils"
    ])
    
    df = db.get_all_data()
    
    # ═══════════════════════════════════════════════════════════════════
    # DASHBOARD
    # ═══════════════════════════════════════════════════════════════════
    
    if menu == "🏠 Dashboard":
        st.title("📊 Dashboard INF 232 EC2")
        st.markdown("*Application de collecte et analyse descriptive de données*")
        
        if df.empty:
            st.info("📌 Commencez par ajouter des données via 'Nouvelle Entrée'")
        else:
            df_prep = prepare_data(df)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("👥 Patients", len(df['nom'].unique()))
            col2.metric("📊 Enregistrements", len(df))
            col3.metric("📏 Âge Moy.", f"{df_prep['age'].mean():.1f} ans")
            col4.metric("⚖️ IMC Moy.", f"{df_prep['imc'].mean():.1f}")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(df, names='genre', title="Distribution Genre")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(df, x='age', nbins=15, title="Distribution Âge")
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df.head(10), use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # NOUVELLE ENTRÉE
    # ═══════════════════════════════════════════════════════════════════
    
    elif menu == "📝 Nouvelle Entrée":
        st.title("📝 Enregistrement de Données")
        
        with st.form("data_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                patient_id = st.text_input("ID Patient", placeholder="P-001")
                nom = st.text_input("Nom Complet")
                age = st.number_input("Âge", 1, 120, 30)
                genre = st.selectbox("Genre", ["Masculin", "Féminin", "Autre"])
            
            with col2:
                poids = st.number_input("Poids (kg)", 30.0, 250.0, 70.0)
                taille = st.number_input("Taille (m)", 1.0, 2.5, 1.75, step=0.01)
                tension_sys = st.slider("Tension Sys (mmHg)", 80, 220, 120)
                tension_dia = st.slider("Tension Dia (mmHg)", 40, 130, 80)
                glycemie = st.number_input("Glycémie (g/L)", 0.2, 6.0, 1.0, step=0.1)
            
            if st.form_submit_button("✅ Enregistrer", use_container_width=True):
                if not patient_id or not nom:
                    st.error("❌ ID et Nom obligatoires")
                elif taille <= 0 or poids <= 0:
                    st.error("❌ Poids/Taille doivent être positifs")
                else:
                    data = {
                        "patient_id": patient_id,
                        "nom": nom,
                        "age": age,
                        "genre": genre,
                        "poids": poids,
                        "taille": taille,
                        "tension_sys": tension_sys,
                        "tension_dia": tension_dia,
                        "glycemie": glycemie,
                        "date_obs": datetime.now()
                    }
                    
                    if db.add_record(data):
                        st.success("✅ Données enregistrées!")
                        st.rerun()
                    else:
                        st.error("❌ Erreur: ID déjà existant")
    
    # ═══════════════════════════════════════════════════════════════════
    # PRÉDICTIONS
    # ═══════════════════════════════════════════════════════════════════
    
    elif menu == "🔮 Prédictions":
        st.title("🔮 Estimation & Prédictions Médicales")
        st.markdown("*Estimez la tension systolique avec intervalle de confiance*")
        
        if df.empty:
            st.warning("⚠️ Au moins 5 enregistrements sont nécessaires")
            return
        
        if len(df) < 5:
            st.warning(f"⚠️ {len(df)} enregistrements. Besoin de 5+ pour une bonne prédiction")
            return
        
        df_prep = prepare_data(df)
        
        st.markdown("---")
        st.subheader("📋 Paramètres du Patient")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Données Biométriques:**")
                age_pred = st.slider("Âge (ans)", 1, 120, 35)
                poids_pred = st.slider("Poids (kg)", 30, 250, 75)
                taille_pred = st.slider("Taille (m)", 1.0, 2.5, 1.75, step=0.01)
            
            with col2:
                st.write("**Marqueurs Biologiques:**")
                glycemie_pred = st.slider("Glycémie (g/L)", 0.2, 5.0, 1.0, step=0.1)
                confidence_level = st.selectbox(
                    "Niveau de Confiance",
                    [0.90, 0.95, 0.99],
                    format_func=lambda x: f"{int(x*100)}%"
                )
            
            if st.form_submit_button("🔮 Générer Prédiction", use_container_width=True):
                with st.spinner("Calcul en cours..."):
                    result = predict_with_confidence_interval(
                        df_prep, age_pred, poids_pred, taille_pred, 
                        glycemie_pred, confidence_level
                    )
                
                if result:
                    st.markdown("---")
                    st.subheader("📊 Résultats de l'Estimation")
                    
                    # Metrics principales
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Tension Estimée",
                            f"{result['prediction']:.1f} mmHg",
                            delta=f"R² = {result['r2']:.3f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Marge d'Erreur",
                            f"± {result['margin_error']:.1f} mmHg",
                            delta=f"{confidence_level*100:.0f}% confiance"
                        )
                    
                    with col3:
                        st.metric(
                            "RMSE du Modèle",
                            f"{result['rmse']:.2f} mmHg",
                            delta="Erreur moyenne"
                        )
                    
                    st.markdown("---")
                    
                    # Intervalle de confiance
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"""
                        **Intervalle de Confiance ({confidence_level*100:.0f}%):**
                        
                        Estimation: **{result['prediction']:.1f} mmHg**
                        
                        - Limite inférieure: {result['lower_bound']:.1f} mmHg
                        - Limite supérieure: {result['upper_bound']:.1f} mmHg
                        
                        La vraie tension systolique a {confidence_level*100:.0f}% de chance de se situer dans cet intervalle.
                        """)
                    
                    with col2:
                        # Visualisation de l'intervalle
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=[result['lower_bound'], result['upper_bound']],
                            y=[1, 1],
                            mode='lines+markers',
                            line=dict(color='rgba(14, 165, 233, 0.5)', width=15),
                            marker=dict(size=12, color='#0ea5e9'),
                            name='IC',
                            hovertemplate='<b>%{x:.1f} mmHg</b><extra></extra>'
                        ))
                        
                        fig.add_vline(
                            x=result['prediction'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Prédiction: {result['prediction']:.1f}",
                            annotation_position="top"
                        )
                        
                        fig.update_layout(
                            title=f"Intervalle de Confiance ({confidence_level*100:.0f}%)",
                            xaxis_title="Tension Systolique (mmHg)",
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📈 Analyse Comparative")
                    
                    # Calculer IMC et comparer
                    imc_pred = poids_pred / (taille_pred ** 2)
                    imc_mean = df_prep['imc'].mean()
                    tension_mean = df_prep['tension_sys'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "IMC Estimé",
                            f"{imc_pred:.1f}",
                            f"vs Moy: {imc_mean:.1f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Tension Estimée",
                            f"{result['prediction']:.1f}",
                            f"vs Moy: {tension_mean:.1f}"
                        )
                    
                    with col3:
                        # Classification de la tension
                        if result['prediction'] < 120:
                            status = "✅ Normale"
                            color = "green"
                        elif result['prediction'] < 140:
                            status = "⚠️ Élevée"
                            color = "orange"
                        else:
                            status = "🔴 Hypertension"
                            color = "red"
                        st.metric("Statut Tension", status)
                    
                    st.markdown("---")
                    st.subheader("🎯 Interprétation Clinique")
                    
                    interpretation = f"""
**Profil du Patient:**
- Âge: {age_pred} ans
- IMC: {imc_pred:.1f} ({'Insuffisant' if imc_pred < 18.5 else 'Normal' if imc_pred <= 25 else 'Surpoids' if imc_pred <= 30 else 'Obésité'})
- Glycémie: {glycemie_pred:.1f} g/L

**Prédiction de la Tension Systolique:**
- Valeur estimée: {result['prediction']:.1f} mmHg
- Intervalle de confiance: [{result['lower_bound']:.1f}, {result['upper_bound']:.1f}] mmHg
- Marge d'erreur: ±{result['margin_error']:.1f} mmHg

**Fiabilité du Modèle:**
- Coefficient R²: {result['r2']:.4f} (explique {result['r2']*100:.1f}% de la variance)
- RMSE: {result['rmse']:.2f} mmHg (erreur moyenne du modèle)
- Nombre d'observations: {len(df)}

**Recommandation:**
Cette estimation est basée sur les données historiques. Pour un diagnostic précis, 
consultez un professionnel de santé.
                    """
                    st.markdown(interpretation)
        
        st.markdown("---")
        st.subheader("📚 Comprendre les Intervalles de Confiance")
        
        with st.expander("ℹ️ Comment interpréter les résultats?"):
            st.markdown("""
            **Intervalle de Confiance 95%:**
            Si on répète l'expérience 100 fois, la vraie valeur se situera dans cet intervalle 95 fois.
            
            **Marge d'Erreur:**
            La différence maximale attendue entre la prédiction et la vraie valeur.
            
            **Exemple:**
            - Prédiction: 130 mmHg
            - IC 95%: [120, 140] mmHg
            - Marge: ±10 mmHg
            
            → Il y a 95% de chance que la vraie tension soit entre 120 et 140 mmHg.
            
            **Amélioration de la Précision:**
            - Plus de données → meilleure estimation
            - Données plus homogènes → intervalle plus serré
            - Modèle plus simple → résultats plus robustes
            """)
    
    
    elif menu == "📊 Analyses":
        st.title("📊 Analyses Avancées (INF 232 EC2)")
        
        if df.empty:
            st.warning("⚠️ Données insuffisantes")
            return
        
        df_prep = prepare_data(df)
        
        tabs = st.tabs([
            "1️⃣ Régression",
            "2️⃣ PCA",
            "3️⃣ Classification Sup.",
            "4️⃣ K-Means",
            "5️⃣ Corrélations"
        ])
        
        # Tab 1: Régression
        with tabs[0]:
            st.subheader("Régression Linéaire")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Simple (IMC → Tension)")
                reg_simple = linear_regression_simple(df_prep)
                if reg_simple:
                    st.metric("R²", f"{reg_simple['r2']:.4f}")
                    st.metric("RMSE", f"{reg_simple['rmse']:.2f}")
                    st.info(f"Coefficient: {reg_simple['coef']:.4f}")
            
            with col2:
                st.subheader("Multiple")
                reg_multi = linear_regression_multiple(df_prep)
                if reg_multi:
                    st.metric("R²", f"{reg_multi['r2']:.4f}")
                    st.metric("RMSE", f"{reg_multi['rmse']:.2f}")
            
            # Scatter
            fig = px.scatter(df_prep, x='imc', y='tension_sys', 
                           title="IMC vs Tension Sys", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: PCA
        with tabs[1]:
            st.subheader("Réduction de Dimensionnalité (PCA)")
            pca_result = pca_analysis(df_prep)
            
            if pca_result:
                col1, col2 = st.columns(2)
                col1.metric("PC1 Variance", f"{pca_result['variance_ratio'][0]:.2%}")
                col2.metric("PC2 Variance", f"{pca_result['variance_ratio'][1]:.2%}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pca_result['X_pca'][:, 0], 
                                        y=pca_result['X_pca'][:, 1],
                                        mode='markers',
                                        marker=dict(size=5, color=df_prep['age']),
                                        text=df['nom']))
                fig.update_layout(title="PCA 2D", xaxis_title="PC1", yaxis_title="PC2")
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Classification Supervisée
        with tabs[2]:
            st.subheader("Classification Supervisée")
            clf_sup = classification_supervised(df_prep)
            
            if clf_sup:
                col1, col2 = st.columns(2)
                col1.metric("Logistic Regression", f"{clf_sup['lr_accuracy']:.2%}")
                col2.metric("Decision Tree", f"{clf_sup['dt_accuracy']:.2%}")
        
        # Tab 4: K-Means
        with tabs[3]:
            st.subheader("K-Means Clustering")
            clf_unsup = classification_unsupervised(df_prep)
            
            if clf_unsup:
                st.metric("Silhouette Score", f"{clf_unsup['silhouette']:.4f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(x=df_prep['age'], 
                                          y=df_prep['poids'],
                                          z=df_prep['taille'],
                                          mode='markers',
                                          marker=dict(size=5, color=clf_unsup['clusters'])))
                fig.update_layout(title="K-Means 3D")
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 5: Corrélations
        with tabs[4]:
            st.subheader("Matrice de Corrélation")
            corr = df_prep[['age', 'poids', 'taille', 'tension_sys', 'glycemie', 'imc']].corr()
            
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # PROFIL PATIENT
    # ═══════════════════════════════════════════════════════════════════
    
    elif menu == "👤 Profil Patient":
        st.title("👤 Profil Patient")
        
        if df.empty:
            st.warning("⚠️ Aucune donnée")
            return
        
        patient = st.selectbox("Sélectionner", sorted(df['nom'].unique()))
        patient_df = db.get_patient_data(patient)
        patient_df = prepare_data(patient_df)
        
        if patient_df.empty:
            return
        
        latest = patient_df.iloc[0]
        
        # Info
        col1, col2, col3 = st.columns(3)
        col1.write(f"**Nom:** {patient}\n**ID:** {latest['patient_id']}\n**Âge:** {latest['age']}")
        col2.write(f"**Poids:** {latest['poids']:.1f} kg\n**Taille:** {latest['taille']:.2f} m\n**IMC:** {latest['imc']:.1f}")
        col3.write(f"**Tension:** {latest['tension_sys']}/{latest['tension_dia']}\n**Glycémie:** {latest['glycemie']:.2f}")
        
        st.markdown("---")
        
        # Download PDF
        df_prep = prepare_data(df)
        analysis_results = {
            'regression': {'simple': linear_regression_simple(df_prep), 'multiple': linear_regression_multiple(df_prep)},
            'pca': pca_analysis(df_prep),
            'classification_sup': classification_supervised(df_prep),
            'classification_unsup': classification_unsupervised(df_prep)
        }
        
        pdf_bytes = generate_pdf(df, df_prep, analysis_results)
        if pdf_bytes:
            st.download_button(
                "📥 Télécharger Rapport PDF",
                pdf_bytes,
                f"Rapport_{patient.replace(' ', '_')}.pdf",
                "application/pdf"
            )
        
        st.markdown("---")
        
        # Charts
        tab1, tab2 = st.tabs(["Évolution", "Distribution"])
        
        with tab1:
            fig = make_subplots(rows=2, cols=2, subplot_titles=("IMC", "Tension", "Glycémie", "Poids"))
            fig.add_trace(go.Scatter(x=patient_df['date_obs'], y=patient_df['imc'], name='IMC', mode='lines+markers'), row=1, col=1)
            fig.add_trace(go.Scatter(x=patient_df['date_obs'], y=patient_df['tension_sys'], name='Tension', mode='lines+markers'), row=1, col=2)
            fig.add_trace(go.Scatter(x=patient_df['date_obs'], y=patient_df['glycemie'], name='Glycémie', mode='lines+markers'), row=2, col=1)
            fig.add_trace(go.Scatter(x=patient_df['date_obs'], y=patient_df['poids'], name='Poids', mode='lines+markers'), row=2, col=2)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(y=patient_df['imc'], title="Distribution IMC")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(y=patient_df['tension_sys'], title="Distribution Tension")
                st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # OUTILS
    # ═══════════════════════════════════════════════════════════════════
    
    elif menu == "⚙️ Outils":
        st.title("⚙️ Outils")
        
        tab1, tab2, tab3 = st.tabs(["Export", "Gestion", "Rapport Complet"])
        
        with tab1:
            st.subheader("Exporter les Données")
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 CSV",
                    csv,
                    f"donnees_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
        
        with tab2:
            st.subheader("Gestion Données")
            if not df.empty:
                action = st.radio("Action", ["Afficher", "Supprimer"])
                
                if action == "Afficher":
                    st.dataframe(df, use_container_width=True)
                else:
                    patient_del = st.selectbox("Patient", df['nom'].unique())
                    if st.button("🗑️ Supprimer"):
                        patient_id = df[df['nom'] == patient_del]['patient_id'].iloc[0]
                        if db.delete_patient(patient_id):
                            st.success("✅ Supprimé")
                            st.rerun()
        
        with tab3:
            st.subheader("Rapport Complet")
            if not df.empty and st.button("📄 Générer"):
                with st.spinner("Génération..."):
                    df_prep = prepare_data(df)
                    analysis_results = {
                        'regression': {'simple': linear_regression_simple(df_prep), 'multiple': linear_regression_multiple(df_prep)},
                        'pca': pca_analysis(df_prep),
                        'classification_sup': classification_supervised(df_prep),
                        'classification_unsup': classification_unsupervised(df_prep)
                    }
                    pdf_bytes = generate_pdf(df, df_prep, analysis_results)
                    
                    if pdf_bytes:
                        st.download_button(
                            "📥 Télécharger",
                            pdf_bytes,
                            f"Rapport_Complet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            "application/pdf"
                        )
                        st.success("✅ Généré!")

if __name__ == "__main__":
    main()
