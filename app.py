import streamlit as st
import pandas as pd
from serpapi import GoogleSearch
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# ────────────────────────────────────────────────────
# 1. Configuration
# ────────────────────────────────────────────────────
st.set_page_config(page_title="Visibilité Voyage Privé – SERP", layout="wide")

try:
    SERPAPI_KEY = st.secrets["serpapi_key"]
except Exception:
    st.error("❌ Clé SerpApi manquante dans `.streamlit/secrets.toml`.")
    st.stop()

VP_DOMAIN = "voyageprive.com"

# ────────────────────────────────────────────────────
# 2. Requêtes prédéfinies (modifiables via l'UI)
# ────────────────────────────────────────────────────
DEFAULT_QUERIES = """voyage tout compris pas cher
séjour dernière minute
hôtel tout inclus bord de mer"""

# ────────────────────────────────────────────────────
# 3. Sidebar
# ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")
    hl = st.selectbox("Langue (hl)", ["fr", "en", "es", "de", "it"], index=0)
    gl = st.selectbox("Pays (gl)", ["fr", "us", "es", "de", "it"], index=0)
    max_workers = st.slider("Threads simultanés", 1, 8, 4)
    num_results = st.slider("Résultats organiques analysés", 10, 30, 10, step=10)

    st.markdown("---")
    st.markdown(
        "**Domaine cible**  \n`voyageprive.com`  \n\nL'app extrait uniquement les URLs VP dans :"
        "\n- 🔵 Résultats organiques\n- 🛍️ Carrousel offres"
    )

# ────────────────────────────────────────────────────
# 4. Zone de saisie des requêtes
# ────────────────────────────────────────────────────
st.title("🔍 Visibilité Voyage Privé sur Google")
st.markdown(
    "Entrez vos requêtes (une par ligne). L'app récupère **toutes les URLs voyageprive.com** "
    "présentes dans les résultats organiques et les carrousels d'offres."
)

queries_raw = st.text_area(
    "📋 Liste de requêtes",
    value=DEFAULT_QUERIES,
    height=180,
    help="Une requête par ligne.",
)
queries = [q.strip() for q in queries_raw.splitlines() if q.strip()]
st.caption(f"**{len(queries)} requête(s)** chargée(s)")

# ────────────────────────────────────────────────────
# 5. Extraction SerpApi
# ────────────────────────────────────────────────────

def extract_vp_results(query: str, hl: str, gl: str, num: int) -> list[dict]:
    """Retourne toutes les URLs VP trouvées pour une requête donnée."""
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": hl,
        "gl": gl,
        "num": num,
        "engine": "google",
    }
    try:
        search = GoogleSearch(params)
        data = search.get_dict()
    except Exception as exc:
        return [_error_row(query, str(exc))]

    rows = []

    # ── Résultats organiques ──────────────────────────
    for pos, r in enumerate(data.get("organic_results", []), start=1):
        link = r.get("link", "")
        if VP_DOMAIN in link:
            rows.append({
                "Requête": query,
                "Type": "🔵 Organique",
                "Position": pos,
                "Titre": r.get("title", "—"),
                "URL": link,
                "Snippet": r.get("snippet", "—"),
                "Date snippet": r.get("date", ""),
            })

    # ── Carrousels d'offres / shopping ───────────────
    carousel_sources = [
        ("shopping_results",    "🛍️ Shopping"),
        ("inline_shopping_results", "🛍️ Inline Shopping"),
        ("ads",                 "📢 Annonce"),
        ("top_stories",         "📰 Top Stories"),
        ("knowledge_graph",     "📖 Knowledge Graph"),
    ]

    for key, label in carousel_sources:
        items = data.get(key, [])
        if isinstance(items, dict):
            items = [items]
        for item in items:
            link = item.get("link", "") or item.get("url", "")
            if VP_DOMAIN in link:
                rows.append({
                    "Requête": query,
                    "Type": label,
                    "Position": item.get("position", "—"),
                    "Titre": item.get("title", "—"),
                    "URL": link,
                    "Snippet": item.get("snippet", item.get("price", "—")),
                    "Date snippet": "",
                })

    # ── Deals / offres carrousel (structure spécifique) ──
    immersive = data.get("immersive_products", []) or data.get("inline_products", [])
    for item in immersive:
        link = item.get("link", "")
        if VP_DOMAIN in link:
            rows.append({
                "Requête": query,
                "Type": "🎠 Carrousel produits",
                "Position": item.get("position", "—"),
                "Titre": item.get("title", "—"),
                "URL": link,
                "Snippet": item.get("price", "—"),
                "Date snippet": "",
            })

    if not rows:
        rows.append({
            "Requête": query,
            "Type": "❌ Absent",
            "Position": "—",
            "Titre": "Voyage Privé non trouvé dans les résultats",
            "URL": "",
            "Snippet": "",
            "Date snippet": "",
        })

    return rows


def _error_row(query: str, msg: str) -> dict:
    return {
        "Requête": query,
        "Type": "⚠️ Erreur",
        "Position": "—",
        "Titre": msg,
        "URL": "",
        "Snippet": "",
        "Date snippet": "",
    }


@st.cache_data(ttl=3_600, show_spinner=False)
def run_all(queries_tuple: tuple, hl: str, gl: str, num: int, workers: int) -> pd.DataFrame:
    rows = []
    progress = st.progress(0.0, text="🔄 Analyse des SERP en cours…")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_vp_results, q, hl, gl, num): q
            for q in queries_tuple
        }
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            progress.progress(i / total, text=f"🔄 {i}/{total} requêtes analysées…")
    progress.empty()
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────
# 6. Lancement + Affichage
# ────────────────────────────────────────────────────

if st.button("🚀 Lancer l'extraction", type="primary", disabled=len(queries) == 0):
    if not queries:
        st.warning("Ajoutez au moins une requête.")
        st.stop()

    df = run_all(tuple(queries), hl, gl, num_results, max_workers)

    # ── KPIs ──────────────────────────────────────────
    total = len(df)
    present = df[df["Type"] != "❌ Absent"]
    absent = df[df["Type"] == "❌ Absent"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Requêtes analysées", len(queries))
    col2.metric("Requêtes avec VP présent", len(present["Requête"].unique()) if not present.empty else 0)
    col3.metric("Résultats VP trouvés", len(present))
    col4.metric("Requêtes sans VP", len(absent))

    st.markdown("---")

    # ── Tableau principal ─────────────────────────────
    st.subheader("📊 Résultats détaillés")

    # Filtres
    col_f1, col_f2 = st.columns(2)
    type_filter = col_f1.multiselect(
        "Filtrer par type",
        options=df["Type"].unique().tolist(),
        default=df["Type"].unique().tolist(),
    )
    query_filter = col_f2.multiselect(
        "Filtrer par requête",
        options=df["Requête"].unique().tolist(),
        default=df["Requête"].unique().tolist(),
    )

    df_filtered = df[df["Type"].isin(type_filter) & df["Requête"].isin(query_filter)]

    st.dataframe(
        df_filtered,
        use_container_width=True,
        height=500,
        column_config={
            "URL": st.column_config.LinkColumn("URL", display_text="🔗 Voir"),
        },
        column_order=["Requête", "Type", "Position", "Titre", "URL", "Snippet"],
    )

    # ── Vue groupée par requête ───────────────────────
    st.markdown("---")
    st.subheader("🔍 Détail par requête")
    for query in df["Requête"].unique():
        subset = df[df["Requête"] == query]
        has_vp = subset[subset["Type"] != "❌ Absent"]
        label = f"{'✅' if not has_vp.empty else '❌'} {query} — {len(has_vp)} résultat(s) VP"
        with st.expander(label):
            for _, row in subset.iterrows():
                if row["Type"] == "❌ Absent":
                    st.info("Voyage Privé n'apparaît pas dans les résultats analysés.")
                else:
                    st.markdown(f"**{row['Type']}** — Position `{row['Position']}`")
                    st.markdown(f"**{row['Titre']}**")
                    if row["URL"]:
                        st.markdown(f"[{row['URL']}]({row['URL']})")
                    if row["Snippet"] and row["Snippet"] not in ("—", ""):
                        st.caption(row["Snippet"])
                    st.markdown("---")

    # ── Exports ───────────────────────────────────────
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)

    csv = df.to_csv(index=False).encode("utf-8")
    col_dl1.download_button(
        "💾 Télécharger CSV",
        data=csv,
        file_name="vp_serp_visibility.csv",
        mime="text/csv",
    )

    xlsx_buffer = BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="VP_SERP")
        # Onglet résumé
        summary = (
            df[df["Type"] != "❌ Absent"]
            .groupby(["Requête", "Type"])
            .size()
            .reset_index(name="Nombre d'URLs VP")
        )
        summary.to_excel(writer, index=False, sheet_name="Résumé")
    xlsx_buffer.seek(0)
    col_dl2.download_button(
        "📊 Télécharger XLSX",
        data=xlsx_buffer,
        file_name="vp_serp_visibility.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
