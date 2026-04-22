import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MediData Analytics Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- DATABASE ENGINE ---
class DBManager:
    def __init__(self, db_name='medidata_pro.db'):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    nom TEXT NOT NULL,
                    age INTEGER,
                    genre TEXT,
                    poids REAL,
                    taille REAL,
                    tension_sys INTEGER,
                    tension_dia INTEGER,
                    glycemie REAL,
                    date_obs TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def add_record(self, data):
        with sqlite3.connect(self.db_name) as conn:
            df = pd.DataFrame([data])
            df.to_sql('health_records', conn, if_exists='append', index=False)

    def get_all_data(self):
        with sqlite3.connect(self.db_name) as conn:
            return pd.read_sql('SELECT * FROM health_records ORDER BY date_obs DESC', conn)

db = DBManager()

# --- PDF GENERATION ENGINE ---
class MedicalReport(FPDF):
    def header(self):
        self.set_fill_color(30, 58, 138)  # Dark Blue
        self.rect(0, 0, 210, 40, 'F')
        self.set_font('Arial', 'B', 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 20, 'MEDIDATA PRO - BILAN CLINIQUE', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Document généré le {datetime.now().strftime("%d/%m/%Y %H:%M")} | Page {self.page_no()}', 0, 0, 'C')

def generate_patient_pdf(patient_name, df_patient):
    pdf = MedicalReport()
    pdf.add_page()
    
    # Patient Info Header
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Rapport de Santé : {patient_name.upper()}", 0, 1)
    pdf.line(10, 55, 200, 55)
    pdf.ln(10)

    # Table Header
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 10)
    headers = ['Date', 'IMC', 'Tension', 'Glycémie (g/L)']
    col_widths = [50, 40, 50, 50]
    
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 10, h, 1, 0, 'C', True)
    pdf.ln()

    # Data Rows
    pdf.set_font('Arial', '', 10)
    for _, row in df_patient.iterrows():
        imc = round(row['poids'] / (row['taille']**2), 2)
        tension = f"{row['tension_sys']}/{row['tension_dia']}"
        
        pdf.cell(col_widths[0], 10, str(row['date_obs'])[:10], 1, 0, 'C')
        pdf.cell(col_widths[1], 10, str(imc), 1, 0, 'C')
        pdf.cell(col_widths[2], 10, tension, 1, 0, 'C')
        pdf.cell(col_widths[3], 10, str(row['glycemie']), 1, 1, 'C')

    # Clinical Analysis Section
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Conclusion Médicale Automatisée :", 0, 1)
    pdf.set_font('Arial', '', 10)
    
    last_rec = df_patient.iloc[0]
    last_imc = last_rec['poids'] / (last_rec['taille']**2)
    
    analysis = "Le patient présente "
    if 18.5 <= last_imc <= 25: analysis += "un IMC normal. "
    else: analysis += "un IMC hors des plages recommandées. "
    
    if last_rec['tension_sys'] > 140: analysis += "Attention : Risque d'hypertension détecté. "
    
    pdf.multi_cell(0, 10, analysis)
    
    return pdf.output()

# --- APP UI ---
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Aller vers", ["🏠 Dashboard", "➕ Nouvelle Entrée", "👤 Analyse Patient", "🌍 Analytics Global"])

    # Load Data
    df = db.get_all_data()

    if menu == "🏠 Dashboard":
        st.title("🏥 MediData Analytics Dashboard")
        
        if not df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Patients Total", len(df['nom'].unique()))
            c2.metric("Enregistrements", len(df))
            c3.metric("Âge Moyen", f"{int(df['age'].mean())} ans")
            c4.metric("Glycémie Moyenne", f"{df['glycemie'].mean():.2f} g/L")

            st.subheader("Dernières Activités")
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.info("Bienvenue ! Commencez par ajouter des données via le menu 'Nouvelle Entrée'.")

    elif menu == "➕ Nouvelle Entrée":
        st.title("📝 Enregistrement Médical")
        
        with st.form("patient_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                id_p = st.text_input("ID Patient (Unique)", placeholder="EX: P-100")
                nom = st.text_input("Nom Complet")
                age = st.number_input("Âge", 1, 120, 25)
                genre = st.selectbox("Genre", ["Masculin", "Féminin", "Autre"])
            with c2:
                poids = st.number_input("Poids (kg)", 1.0, 250.0, 70.0)
                taille = st.number_input("Taille (m)", 0.5, 2.5, 1.75)
                t_sys = st.slider("Tension Systolique (mmHg)", 80, 220, 120)
                t_dia = st.slider("Tension Diastolique (mmHg)", 40, 130, 80)
                glycemie = st.number_input("Glycémie (g/L)", 0.2, 6.0, 1.0)
            
            submitted = st.form_submit_button("Enregistrer les données")
            if submitted:
                if not id_p or not nom:
                    st.error("L'ID et le Nom sont obligatoires.")
                else:
                    new_data = {
                        "patient_id": id_p, "nom": nom, "age": age, "genre": genre,
                        "poids": poids, "taille": taille, "tension_sys": t_sys,
                        "tension_dia": t_dia, "glycemie": glycemie,
                        "date_obs": datetime.now()
                    }
                    db.add_record(new_data)
                    st.success("Données enregistrées avec succès !")

    elif menu == "👤 Analyse Patient":
        st.title("🔍 Profil Médical Individuel")
        
        if df.empty:
            st.warning("Aucune donnée disponible.")
            return

        patient_list = sorted(df['nom'].unique())
        selected_patient = st.selectbox("Sélectionner un patient", patient_list)
        
        patient_df = df[df['nom'] == selected_patient].copy()
        patient_df['imc'] = patient_df['poids'] / (patient_df['taille']**2)
        
        # Current Status
        latest = patient_df.iloc[0]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = latest['imc'],
                title = {'text': "IMC Actuel"},
                gauge = {
                    'axis': {'range': [10, 40]},
                    'bar': {'color': "#1e3a8a"},
                    'steps': [
                        {'range': [0, 18.5], 'color': "#A3E4D7"},
                        {'range': [18.5, 25], 'color': "#52BE80"},
                        {'range': [25, 30], 'color': "#F4D03F"},
                        {'range': [30, 40], 'color': "#E74C3C"}]
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with c2:
            st.markdown(f"### Détails Cliniques")
            st.write(f"**Patient:** {selected_patient}")
            st.write(f"**Âge:** {latest['age']} ans")
            st.write(f"**Genre:** {latest['genre']}")
            st.write(f"**Dernière Visite:** {str(latest['date_obs'])[:16]}")
            
        with c3:
            st.markdown("### Actions")
            pdf_bytes = generate_patient_pdf(selected_patient, patient_df)
            st.download_button(
                label="📥 Télécharger Rapport Complet (PDF)",
                data=bytes(pdf_bytes),
                file_name=f"Rapport_{selected_patient.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )

        # Evolution Charts
        st.subheader("Historique des Signes Vitaux")
        tab1, tab2 = st.tabs(["Tension & Glycémie", "Évolution IMC"])
        
        with tab1:
            fig_evol = go.Figure()
            fig_evol.add_trace(go.Scatter(x=patient_df['date_obs'], y=patient_df['tension_sys'], name="Sys (mmHg)", line=dict(color='red')))
            fig_evol.add_trace(go.Scatter(x=patient_df['date_obs'], y=patient_df['glycemie']*100, name="Glycémie (cg/L)", line=dict(color='blue')))
            st.plotly_chart(fig_evol, use_container_width=True)
        
        with tab2:
            fig_imc = px.area(patient_df, x='date_obs', y='imc', title="Courbe d'IMC")
            st.plotly_chart(fig_imc, use_container_width=True)

    elif menu == "🌍 Analytics Global":
        st.title("📊 Analyse de la Population")
        
        if df.empty:
            st.warning("Données insuffisantes pour l'analyse.")
            return

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution par Genre & Âge")
            fig1 = px.histogram(df, x="age", color="genre", marginal="rug", nbins=30, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Relation Tension vs IMC")
            df['imc'] = df['poids'] / (df['taille']**2)
            fig2 = px.scatter(df, x="imc", y="tension_sys", size="age", color="genre", hover_name="nom", trendline="ols")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Matrice de Corrélation des Facteurs de Risque")
        numeric_df = df[['age', 'poids', 'taille', 'tension_sys', 'tension_dia', 'glycemie']].copy()
        numeric_df['imc'] = numeric_df['poids'] / (numeric_df['taille']**2)
        corr = numeric_df.corr()
        fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
