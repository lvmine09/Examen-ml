import os
import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Clustering", page_icon="📊")

# ---------------------------------------------------------------------------
# Les 3 jeux de données : un dossier models/<folder> contenant artefacts.joblib
# (fichier créé à la fin de chaque notebook)
# ---------------------------------------------------------------------------
JEUX = {
    "Grains de blé (K-means)": {
        "folder": "seeds",
        "source": "UCI : Seeds",
        "lien": "https://archive.ics.uci.edu/dataset/236/seeds",
        "texte": "Segmentation de grains de blé à partir de 7 mesures morphologiques (surface, périmètre, longueur, largeur...).",
    },
    "Clients télécom (CAH)": {
        "folder": "customer",
        "source": "UCI : Iranian Churn Dataset",
        "lien": "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
        "texte": "Segmentation de clients d'un opérateur télécom à partir de leur usage (appels, SMS, valeur du client).",
    },
    "Wholesale customers (K-means)": {
        "folder": "wholesale",
        "source": "UCI : Wholesale customers",
        "lien": "https://archive.ics.uci.edu/dataset/292/wholesale+customers",
        "texte": "Segmentation de clients d'un grossiste à partir de leurs dépenses annuelles par catégorie de produits.",
    },
    "Wine Quality, vin rouge (CAH)": {
        "folder": "wine",
        "source": "UCI : Wine Quality",
        "lien": "https://archive.ics.uci.edu/dataset/186/wine+quality",
        "texte": "Segmentation de vins rouges à partir de leurs mesures physico-chimiques.",
    },
}


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
st.sidebar.title("Clustering")
choix = st.sidebar.selectbox("Choisir un jeu de données", list(JEUX))
jeu = JEUX[choix]

st.title(choix)
st.write(jeu["texte"])
st.caption(f"Source : [{jeu['source']}]({jeu['lien']})")

try:
    art = charger(jeu["folder"])
except FileNotFoundError:
    st.error(f"Fichier introuvable : models/{jeu['folder']}/artefacts.joblib")
    st.stop()

colonnes = art["colonnes"]
tab1, tab2 = st.tabs(["Simple Prédiction", "Prédiction multiple"])

# ----- Interface 1 : un seul individu
with tab1:
    valeurs = [
        st.number_input(col, value=float(art["mediane"][col]), key=f"{choix}_{col}")
        for col in colonnes
    ]
    if st.button("Prédire", key=f"{choix}_btn"):
        st.success(affecter_client(art, *valeurs))

# ----- Interface 2 : plusieurs individus avec un fichier csv
with tab2:
    st.write("Le fichier csv doit contenir les colonnes : " + ", ".join(nettoyer(colonnes)))
    fichier = st.file_uploader("Upload a csv file", type="csv", key=f"{choix}_file")
    if fichier is not None:
        df = pd.read_csv(fichier, sep=None, engine="python")  # détecte , ou ;
        df.columns = nettoyer(df.columns)
        attendues = nettoyer(colonnes)
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
            st.dataframe(df)
            st.bar_chart(df["Groupe"].value_counts())
            st.download_button("Download a csv file", df.to_csv(index=False),
                               "predictions.csv", "text/csv")
