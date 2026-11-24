import os
import html
import numpy as np
import pandas as pd
import joblib as jb
import altair as alt
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Clustering", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Les jeux de données : un dossier models/<folder> contenant artefacts.joblib
# (fichier créé à la fin de chaque notebook)
# ---------------------------------------------------------------------------
JEUX = {
    "🌾 Grains de blé (K-means)": {
        "folder": "seeds", "methode": "K-means",
        "source": "UCI : Seeds", "lien": "https://archive.ics.uci.edu/dataset/236/seeds",
        "texte": "Segmentation de grains de blé à partir de 7 mesures morphologiques (surface, périmètre, longueur, largeur...).",
        "couleurs": ("#43A047", "#1B5E20"),
    },
    "📱 Clients télécom (CAH)": {
        "folder": "customer", "methode": "CAH",
        "source": "UCI : Iranian Churn Dataset", "lien": "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
        "texte": "Segmentation de clients d'un opérateur télécom à partir de leur usage (appels, SMS, valeur du client).",
        "couleurs": ("#1E88E5", "#0D47A1"),
    },
    "🛒 Wholesale customers (K-means)": {
        "folder": "wholesale", "methode": "K-means",
        "source": "UCI : Wholesale customers", "lien": "https://archive.ics.uci.edu/dataset/292/wholesale+customers",
        "texte": "Segmentation de clients d'un grossiste à partir de leurs dépenses annuelles par catégorie de produits.",
        "couleurs": ("#FB8C00", "#E65100"),
    },
    "🍷 Wine Quality, vin rouge (CAH)": {
        "folder": "wine", "methode": "CAH",
        "source": "UCI : Wine Quality", "lien": "https://archive.ics.uci.edu/dataset/186/wine+quality",
        "texte": "Segmentation de vins rouges à partir de leurs mesures physico-chimiques.",
        "couleurs": ("#C2185B", "#6A1B4D"),
    },
}
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


@st.cache_resource
def charger(folder):
    return jb.load(os.path.join("models", folder, "artefacts.joblib"))


def predire_classe(nouveau_client, art):
    """Même fonction que dans les notebooks : classe du centroïde le plus proche."""
    client = np.array(nouveau_client, dtype=float).reshape(1, -1)
    if art.get("normaliser", True):  # les grains ne sont pas normalisés
        client = normalize(client)
    distances = cdist(client, art["centroides"])
    return art["classes"][distances.argmin()]


def affecter_client(art, *valeurs):
    classe = predire_classe(valeurs, art)
    nom = art["noms_classes"][classe]
    return art["phrase"] + nom.strip()


def nettoyer(noms):
    """Enlève les espaces en trop dans les noms de colonnes."""
    return [" ".join(str(c).split()) for c in noms]


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Clustering")
st.sidebar.caption("Choisis un jeu de données")
choix = st.sidebar.selectbox("Jeu de données", list(JEUX), label_visibility="collapsed")
jeu = JEUX[choix]
c1, c2 = jeu["couleurs"]

st.sidebar.markdown("---")
st.sidebar.caption("Projet Machine Learning · K-means et CAH · Sources : UCI Machine Learning Repository")

# ----- style
st.markdown(f"""
<style>
.block-container {{ max-width: 1100px; padding-top: 2rem; }}
.hero {{
    background: linear-gradient(135deg, {c1}, {c2});
    border-radius: 18px; padding: 28px 32px; color: white; margin-bottom: 1.2rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.18);
}}
.hero h1 {{ color: white; margin: 0 0 6px 0; font-size: 2rem; padding: 0; }}
.hero p {{ margin: 0 0 12px 0; font-size: 1.05rem; opacity: 0.95; }}
.badge {{
    display: inline-block; background: rgba(255,255,255,0.22); color: white !important;
    padding: 4px 12px; border-radius: 999px; font-size: 0.85rem; margin-right: 8px;
    text-decoration: none !important;
}}
.chip {{
    display: inline-block; padding: 5px 14px; border-radius: 999px; margin: 3px 6px 3px 0;
    color: white; font-size: 0.9rem; font-weight: 600;
}}
.result-card {{
    background: linear-gradient(135deg, {c1}, {c2}); color: white; border-radius: 16px;
    padding: 22px 26px; margin-top: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}}
.result-card .label {{ font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.85; }}
.result-card .text {{ font-size: 1.7rem; font-weight: 700; margin-top: 4px; }}
div[data-testid="stMetric"] {{
    background: rgba(128,128,128,0.10); border-radius: 14px; padding: 12px 16px;
}}
button[data-baseweb="tab"] {{ font-size: 1.05rem; }}
div.stButton > button, div.stDownloadButton > button {{
    background: linear-gradient(135deg, {c1}, {c2}); color: white; border: none;
    border-radius: 10px; padding: 0.5rem 1.4rem; font-weight: 600;
}}
div.stButton > button:hover, div.stDownloadButton > button:hover {{ color: white; opacity: 0.9; }}
</style>
""", unsafe_allow_html=True)

# ----- bandeau
st.markdown(f"""
<div class="hero">
  <h1>{html.escape(choix)}</h1>
  <p>{html.escape(jeu["texte"])}</p>
  <span class="badge">Méthode déployée : {jeu["methode"]}</span>
  <a class="badge" href="{jeu["lien"]}" target="_blank">Source : {html.escape(jeu["source"])}</a>
</div>
""", unsafe_allow_html=True)

try:
    art = charger(jeu["folder"])
except FileNotFoundError:
    st.error(f"Fichier introuvable : models/{jeu['folder']}/artefacts.joblib")
    st.stop()

colonnes = art["colonnes"]
noms_groupes = [art["noms_classes"][c].strip() for c in art["classes"]]

# ----- chiffres clés
m1, m2, m3 = st.columns(3)
m1.metric("Variables", len(colonnes))
m2.metric("Groupes", len(art["classes"]))
m3.metric("Méthode", jeu["methode"])

st.markdown("**Les groupes possibles :**")
st.markdown("".join(
    f'<span class="chip" style="background:{PALETTE[i % len(PALETTE)]}">{html.escape(n)}</span>'
    for i, n in enumerate(noms_groupes)), unsafe_allow_html=True)
st.write("")

tab1, tab2 = st.tabs(["🎯 Simple Prédiction", "📂 Prédiction multiple"])

# ----- Interface 1 : un seul individu
with tab1:
    st.write("Renseigne les valeurs, puis clique sur **Prédire**.")
    cols = st.columns(3)
    valeurs = []
    for i, col in enumerate(colonnes):
        with cols[i % 3]:
            valeurs.append(st.number_input(col, value=float(art["mediane"][col]), key=f"{choix}_{col}"))
    if st.button("Prédire", key=f"{choix}_btn"):
        resultat = affecter_client(art, *valeurs)
        st.success(resultat)
        st.markdown(f"""
        <div class="result-card">
          <div class="label">Résultat</div>
          <div class="text">{html.escape(resultat)}</div>
        </div>""", unsafe_allow_html=True)

# ----- Interface 2 : plusieurs individus avec un fichier csv
with tab2:
    attendues = nettoyer(colonnes)
    st.info("Le fichier csv doit contenir les colonnes : " + ", ".join(attendues))

    # fichier d'exemple téléchargeable
    exemple = pd.DataFrame([{c: round(art["mediane"][o] * f, 3) for c, o in zip(attendues, colonnes)}
                            for f in (0.7, 1.0, 1.3)])
    st.download_button("⬇️ Télécharger un fichier d'exemple", exemple.to_csv(index=False),
                       "exemple.csv", "text/csv", key=f"{choix}_exemple")

    fichier = st.file_uploader("Upload a csv file", type="csv", key=f"{choix}_file")
    if fichier is not None:
        df = pd.read_csv(fichier, sep=None, engine="python")  # détecte , ou ;
        df.columns = nettoyer(df.columns)
        manquantes = [c for c in attendues if c not in df.columns]
        if manquantes:
            st.error("Colonnes manquantes dans le fichier : " + ", ".join(manquantes))
        else:
            X = df[attendues].astype(float).values
            if art.get("normaliser", True):
                X = normalize(X)
            idx = cdist(X, art["centroides"]).argmin(axis=1)
            df["Classe"] = art["classes"][idx]
            df["Groupe"] = [art["noms_classes"][c].strip() for c in df["Classe"]]

            st.success(f"{len(df)} individus classés.")
            st.dataframe(df)

            comptes = df["Groupe"].value_counts().reset_index()
            comptes.columns = ["Groupe", "Nombre"]
            graphique = alt.Chart(comptes).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                x=alt.X("Groupe:N", sort="-y", axis=alt.Axis(labelAngle=0, title=None)),
                y=alt.Y("Nombre:Q", title="Nombre d'individus"),
                color=alt.Color("Groupe:N", legend=None, scale=alt.Scale(range=PALETTE)),
                tooltip=["Groupe", "Nombre"],
            ).properties(width="container", height=300)
            st.altair_chart(graphique)

            st.download_button("⬇️ Download a csv file", df.to_csv(index=False),
                               "predictions.csv", "text/csv", key=f"{choix}_dl")
