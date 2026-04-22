from flask import Flask, render_template, request, redirect, url_for, send_file
import sqlite3
import pandas as pd
import random
import os
from fpdf import FPDF

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

app = Flask(__name__)
DB = "population.db"


# ---------- DB ----------
def get_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS population (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        sexe TEXT,
        region TEXT,
        niveau_etude TEXT,
        statut_matrimonial TEXT,
        taille_menage INTEGER,
        revenu REAL,
        poids REAL,
        taille REAL,
        imc REAL,
        activite_physique INTEGER,
        tabac TEXT,
        alcool TEXT,
        sommeil INTEGER
    )
    """)
    conn.commit()
    conn.close()

def seed_100():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM population")
    if cur.fetchone()["c"] >= 100:
        conn.close()
        return

    regions = ["Centre","Littoral","Ouest","Nord","Extrême-Nord",
               "Sud","Est","Adamaoua","Nord-Ouest","Sud-Ouest"]
    niveaux = ["Primaire","Secondaire","Université","Supérieur"]
    statuts = ["Célibataire","Marié(e)","Divorcé(e)","Veuf(ve)"]
    sexes = ["Homme","Femme"]

    for _ in range(100):
        age = random.randint(18, 75)
        sexe = random.choice(sexes)
        region = random.choice(regions)
        niveau = random.choice(niveaux)
        statut = random.choice(statuts)
        taille_menage = random.randint(1, 10)
        revenu = random.randint(30000, 500000)
        poids = random.uniform(45, 110)
        taille = random.uniform(1.5, 1.9)
        imc = poids / (taille ** 2)
        activite = random.randint(0, 300)
        tabac = random.choice(["Oui","Non"])
        alcool = random.choice(["Oui","Non"])
        sommeil = random.randint(1, 5)

        conn.execute("""
        INSERT INTO population
        (age,sexe,region,niveau_etude,statut_matrimonial,taille_menage,
         revenu,poids,taille,imc,activite_physique,tabac,alcool,sommeil)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (age,sexe,region,niveau,statut,taille_menage,revenu,
              poids,taille,imc,activite,tabac,alcool,sommeil))
    conn.commit()
    conn.close()


# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/add", methods=["POST"])
def add():
    data = request.form
    age = int(data["age"])
    sexe = data["sexe"]
    region = data["region"]
    niveau = data["niveau_etude"]
    statut = data["statut_matrimonial"]
    taille_menage = int(data["taille_menage"])
    revenu = float(data["revenu"])
    poids = float(data["poids"])
    taille = float(data["taille"])
    imc = poids / (taille ** 2)
    activite = int(data["activite_physique"])
    tabac = data["tabac"]
    alcool = data["alcool"]
    sommeil = int(data["sommeil"])

    conn = get_conn()
    conn.execute("""
    INSERT INTO population
    (age,sexe,region,niveau_etude,statut_matrimonial,taille_menage,
     revenu,poids,taille,imc,activite_physique,tabac,alcool,sommeil)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (age,sexe,region,niveau,statut,taille_menage,revenu,
          poids,taille,imc,activite,tabac,alcool,sommeil))
    conn.commit()
    conn.close()
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM population", conn)
    conn.close()

    if df.empty:
        return render_template("dashboard.html", empty=True)

    stats = {
        "n": len(df),
        "age_mean": round(df["age"].mean(), 1),
        "imc_mean": round(df["imc"].mean(), 1),
        "rev_mean": round(df["revenu"].mean(), 0)
    }
    sexe_counts = df["sexe"].value_counts().to_dict()
    region_counts = df["region"].value_counts().to_dict()
    niveau_counts = df["niveau_etude"].value_counts().to_dict()

    return render_template(
        "dashboard.html",
        empty=False,
        stats=stats,
        sexe_counts=sexe_counts,
        region_counts=region_counts,
        niveau_counts=niveau_counts
    )


@app.route("/analytics")
def analytics():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM population", conn)
    conn.close()

    if df.empty:
        return render_template("analytics.html", empty=True)

    # Régression simple : revenu ~ âge
    Xs = df[["age"]].values
    ys = df["revenu"].values
    reg_s = LinearRegression().fit(Xs, ys)
    ys_pred = reg_s.predict(Xs)
    reg_simple = {
        "coef": float(reg_s.coef_[0]),
        "intercept": float(reg_s.intercept_),
        "r2": float(r2_score(ys, ys_pred)),
        "rmse": float(mean_squared_error(ys, ys_pred, squared=False))
    }

    # Régression multiple : revenu ~ âge + taille_menage + sommeil
    Xm = df[["age","taille_menage","sommeil"]].values
    ym = df["revenu"].values
    reg_m = LinearRegression().fit(Xm, ym)
    ym_pred = reg_m.predict(Xm)
    reg_multi = {
        "coef": reg_m.coef_.tolist(),
        "intercept": float(reg_m.intercept_),
        "r2": float(r2_score(ym, ym_pred)),
        "rmse": float(mean_squared_error(ym, ym_pred, squared=False))
    }

    # PCA
    Xp = df[["age","revenu","taille_menage","activite_physique","imc","sommeil"]].values
    Xp_s = StandardScaler().fit_transform(Xp)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(Xp_s)
    pca_res = {
        "explained": pca.explained_variance_ratio_.tolist(),
        "comp1": comps[:,0].tolist(),
        "comp2": comps[:,1].tolist()
    }

    # Classification supervisée : classe de revenu
    bins = [0, 100000, 300000, 1e9]
    labels = ["faible","moyen","élevé"]
    df["classe_revenu"] = pd.cut(df["revenu"], bins=bins, labels=labels)
    le = LabelEncoder()
    y_cls = le.fit_transform(df["classe_revenu"].astype(str))
    X_cls = df[["age","taille_menage","activite_physique","imc","sommeil"]].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    cls_sup = {
        "classes": le.classes_.tolist(),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist()
    }

    # Clustering (K-Means)
    Xc = df[["age","revenu","activite_physique","imc","sommeil"]].values
    Xc_s = StandardScaler().fit_transform(Xc)
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels_km = km.fit_predict(Xc_s)
    df["cluster"] = labels_km
    cluster_counts = df["cluster"].value_counts().to_dict()

    return render_template(
        "analytics.html",
        empty=False,
        reg_simple=reg_simple,
        reg_multi=reg_multi,
        pca_res=pca_res,
        cls_sup=cls_sup,
        cluster_counts=cluster_counts
    )


@app.route("/upload", methods=["GET","POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return redirect(url_for("upload"))
        df = pd.read_csv(file)
        required = {"age","sexe","region","niveau_etude","statut_matrimonial",
                    "taille_menage","revenu","poids","taille",
                    "activite_physique","tabac","alcool","sommeil"}
        if not required.issubset(df.columns):
            return "Colonnes manquantes", 400

        conn = get_conn()
        for _, row in df.iterrows():
            imc = row["poids"] / (row["taille"] ** 2)
            conn.execute("""
            INSERT INTO population
            (age,sexe,region,niveau_etude,statut_matrimonial,taille_menage,
             revenu,poids,taille,imc,activite_physique,tabac,alcool,sommeil)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (int(row["age"]), row["sexe"], row["region"], row["niveau_etude"],
                  row["statut_matrimonial"], int(row["taille_menage"]),
                  float(row["revenu"]), float(row["poids"]), float(row["taille"]),
                  imc, int(row["activite_physique"]), row["tabac"],
                  row["alcool"], int(row["sommeil"])))
        conn.commit()
        conn.close()
        return redirect(url_for("dashboard"))
    return render_template("upload.html")


@app.route("/export_pdf")
def export_pdf():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM population", conn)
    conn.close()
    if df.empty:
        return "Pas de données", 400

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(15, 23, 42)
            self.rect(0, 0, 210, 15, "F")
            self.set_text_color(250, 204, 21)
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "Étude socio-sanitaire - Rapport descriptif", 0, 1, "C")
            self.ln(5)
        def footer(self):
            self.set_y(-10)
            self.set_font("Arial", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 5, f"Page {self.page_no()}", 0, 0, "C")

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 10)

    txt = f"Nombre d'individus : {len(df)}\n"
    txt += f"Âge moyen : {df['age'].mean():.1f}\n"
    txt += f"IMC moyen : {df['imc'].mean():.1f}\n"
    txt += f"Revenu moyen : {df['revenu'].mean():.0f} FCFA\n"
    pdf.multi_cell(0, 5, txt)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Répartition par sexe :", 0, 1)
    pdf.set_font("Arial", "", 10)
    for s, c in df["sexe"].value_counts().items():
        pdf.cell(0, 5, f"- {s} : {c}", 0, 1)

    pdf_output = "rapport_population.pdf"
    pdf.output(pdf_output)
    return send_file(pdf_output, as_attachment=True)


if __name__ == "__main__":
    init_db()
    seed_100()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

