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

VP_DOMAIN = "voyage-prive.com"

# ────────────────────────────────────────────────────
# 2. Requêtes de test prédéfinies
# ────────────────────────────────────────────────────
DEFAULT_QUERIES = """voyage en Thaïlande
séjour tout compris pas cher
hôtel bord de mer"""

# ────────────────────────────────────────────────────
# 3. Sidebar
# ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")
    hl = st.selectbox("Langue (hl)", ["fr", "en", "es", "de", "it"], index=0)
    gl = st.selectbox("Pays (gl)", ["fr", "us", "es", "de", "it"], index=0)
    num_results = st.slider("Résultats organiques analysés", 10, 30, 10, step=10)
    max_workers = st.slider("Threads simultanés", 1, 8, 3)

    st.markdown("---")
    st.markdown(
        "**Sources extraites**\n\n"
        "- 🔵 Lien bleu principal (résultat organique VP)\n"
        "- 🎠 Sitelinks carrousel (offres VP sous le résultat)"
    )

# ────────────────────────────────────────────────────
# 4. Zone de saisie des requêtes
# ────────────────────────────────────────────────────
st.title("🔍 Visibilité Voyage Privé – Liens organiques & Sitelinks")
st.markdown(
    "Extrait les **URLs `voyage-prive.com`** depuis :\n"
    "- 🔵 Le **lien bleu principal** dans les résultats organiques\n"
    "- 🎠 Les **sitelinks en carrousel** (offres affichées sous le résultat principal)"
)

queries_raw = st.text_area(
    "📋 Liste de requêtes (une par ligne)",
    value=DEFAULT_QUERIES,
    height=160,
)
queries = [q.strip() for q in queries_raw.splitlines() if q.strip()]
st.caption(f"**{len(queries)} requête(s)** chargée(s)")

# ────────────────────────────────────────────────────
# 5. Extraction SerpApi
# ────────────────────────────────────────────────────

def extract_vp_results(query: str, hl: str, gl: str, num: int) -> list[dict]:
    """
    Extrait depuis SerpApi :
    - Le résultat organique principal VP (lien bleu)
    - Les sitelinks inline VP (carrousel d'offres sous le résultat)
    """
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
        return [_row(query, "⚠️ Erreur API", "—", str(exc), "", "—")]

    rows = []
    organic = data.get("organic_results", [])

    for pos, result in enumerate(organic, start=1):
        main_link = result.get("link", "")

        # ── 🔵 Lien bleu principal ────────────────────
        if VP_DOMAIN in main_link:
            rows.append(_row(
                query   = query,
                type_   = "🔵 Lien principal",
                position= pos,
                titre   = result.get("title", "—"),
                url     = main_link,
                snippet = result.get("snippet", "—"),
            ))

            # ── 🎠 Sitelinks inline (carrousel offres) ─
            # SerpApi retourne les sitelinks dans organic_results[n]["sitelinks"]
            # sous la clé "inline" (liste de dicts avec "title" et "link")
            sitelinks_data = result.get("sitelinks", {})

            if isinstance(sitelinks_data, dict):
                inline_links = sitelinks_data.get("inline", [])
            elif isinstance(sitelinks_data, list):
                inline_links = sitelinks_data
            else:
                inline_links = []

            for idx, sl in enumerate(inline_links, start=1):
                sl_link = sl.get("link", "")
                if VP_DOMAIN in sl_link:
                    rows.append(_row(
                        query   = query,
                        type_   = "🎠 Sitelink carrousel",
                        position= f"{pos}.{idx}",
                        titre   = sl.get("title", "—"),
                        url     = sl_link,
                        snippet = sl.get("snippet", ""),
                    ))

    if not rows:
        rows.append(_row(query, "❌ Absent", "—", "Voyage Privé absent des résultats", "", ""))

    return rows


def _row(query, type_, position, titre, url, snippet) -> dict:
    return {
        "Requête" : query,
        "Type"    : type_,
        "Position": position,
        "Titre"   : titre,
        "URL"     : url,
        "Snippet" : snippet,
    }


@st.cache_data(ttl=3_600, show_spinner=False)
def run_all(queries_tuple: tuple, hl: str, gl: str, num: int, workers: int) -> pd.DataFrame:
    rows = []
    progress = st.progress(0.0, text="🔄 Analyse des SERP…")
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

    df = run_all(tuple(queries), hl, gl, num_results, max_workers)

    # ── KPIs ─────────────────────────────────────────
    present   = df[~df["Type"].isin(["❌ Absent", "⚠️ Erreur API"])]
    principal = df[df["Type"] == "🔵 Lien principal"]
    sitelinks = df[df["Type"] == "🎠 Sitelink carrousel"]
    absent    = df[df["Type"] == "❌ Absent"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requêtes analysées",     len(queries))
    c2.metric("🔵 Liens principaux VP", len(principal))
    c3.metric("🎠 Sitelinks offres VP", len(sitelinks))
    c4.metric("❌ Sans présence VP",    len(absent))

    st.markdown("---")

    # ── Tableau principal ────────────────────────────
    st.subheader("📊 Résultats détaillés")

    type_filter = st.multiselect(
        "Filtrer par type",
        options=df["Type"].unique().tolist(),
        default=df["Type"].unique().tolist(),
    )
    df_filtered = df[df["Type"].isin(type_filter)]

    st.dataframe(
        df_filtered,
        use_container_width=True,
        height=450,
        column_config={
            "URL": st.column_config.LinkColumn("URL", display_text="🔗 Voir"),
        },
        column_order=["Requête", "Type", "Position", "Titre", "URL", "Snippet"],
    )

    # ── Vue groupée par requête ──────────────────────
    st.markdown("---")
    st.subheader("🔍 Détail par requête")

    for query in df["Requête"].unique():
        subset  = df[df["Requête"] == query]
        nb_sl   = len(subset[subset["Type"] == "🎠 Sitelink carrousel"])
        has_main= len(subset[subset["Type"] == "🔵 Lien principal"]) > 0
        has_vp  = has_main or nb_sl > 0

        badge_main = "🔵 lien principal" if has_main else ""
        badge_sl   = f"+ 🎠 {nb_sl} sitelink(s)" if nb_sl else ""
        label = f"{'✅' if has_vp else '❌'} {query} — {badge_main} {badge_sl}".strip(" —")

        with st.expander(label):
            if not has_vp:
                st.info("Voyage Privé n'apparaît pas dans les résultats analysés.")
                continue

            # Lien principal
            main_rows = subset[subset["Type"] == "🔵 Lien principal"]
            if not main_rows.empty:
                r = main_rows.iloc[0]
                st.markdown(f"**🔵 Résultat principal — Position `{r['Position']}`**")
                st.markdown(f"**{r['Titre']}**")
                st.markdown(f"[{r['URL']}]({r['URL']})")
                if r["Snippet"] and r["Snippet"] not in ("—", ""):
                    st.caption(r["Snippet"])
                st.markdown("---")

            # Sitelinks carrousel
            sl_rows = subset[subset["Type"] == "🎠 Sitelink carrousel"]
            if not sl_rows.empty:
                st.markdown(f"**🎠 Sitelinks carrousel — {len(sl_rows)} offre(s)**")
                for _, r in sl_rows.iterrows():
                    col_a, col_b = st.columns([3, 7])
                    col_a.markdown(f"Position `{r['Position']}`")
                    col_b.markdown(f"**{r['Titre']}** → [{r['URL']}]({r['URL']})")

    # ── Exports ──────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    csv = df.to_csv(index=False).encode("utf-8")
    col1.download_button(
        "💾 Télécharger CSV",
        data=csv,
        file_name="vp_serp_sitelinks.csv",
        mime="text/csv",
    )

    xlsx_buffer = BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Détail")
        summary = (
            df[~df["Type"].isin(["❌ Absent", "⚠️ Erreur API"])]
            .groupby(["Requête", "Type"])
            .size()
            .reset_index(name="Nombre URLs VP")
        )
        summary.to_excel(writer, index=False, sheet_name="Résumé")
    xlsx_buffer.seek(0)
    col2.download_button(
        "📊 Télécharger XLSX",
        data=xlsx_buffer,
        file_name="vp_serp_sitelinks.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
