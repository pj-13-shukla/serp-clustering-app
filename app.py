import streamlit as st
import pandas as pd
import openai
import requests
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI-Based Keyword Clustering Tool", layout="wide")
st.title("🧠 AI-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV and get semantic, search-intent-based clusters with smart labels.")

# -----------------------------
# File Upload and API Inputs
# -----------------------------
uploaded_file = st.file_uploader("\ud83d\udcc4 Upload your keywords.csv file", type="csv")
openai_api = st.text_input("\ud83d\udd11 OpenAI API Key", type="password")
threshold = st.slider("\ud83e\uddd0 Cosine Similarity Threshold", 70, 95, 80)

if "final_df" not in st.session_state:
    st.session_state.final_df = None

# -----------------------------
# Embedding Function
# -----------------------------
def get_embedding(text):
    try:
        res = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        return res["data"][0]["embedding"]
    except Exception:
        return None

# -----------------------------
# GPT Label Generator
# -----------------------------
def generate_cluster_label(keywords, api_key):
    openai.api_key = api_key
    prompt = f"""
You're an SEO assistant. Given the following keywords:
{keywords}
Return a short, capitalized, 2–4 word label describing the overall search intent or topic.
Avoid long tails, duplicates, or overly generic terms. Use acronyms when relevant.
Just return the label. No other text.
"""
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message["content"].strip().title()
    except Exception:
        return "Unlabeled Cluster"

# -----------------------------
# Main Logic
# -----------------------------
if st.button("\ud83d\ude80 Run Clustering") and uploaded_file and openai_api:
    st.info("Embedding and clustering... hold tight.")
    df = pd.read_csv(uploaded_file)

    # Detect keyword column
    possible_cols = ['keyword', 'keywords', 'query', 'queries']
    df_columns_lower = [col.lower().strip() for col in df.columns]
    keyword_col = None
    for col in possible_cols:
        if col in df_columns_lower:
            keyword_col = df.columns[df_columns_lower.index(col)]
            break
    if not keyword_col:
        for col in df.columns:
            if df[col].dtype == 'object':
                keyword_col = col
                break
    if not keyword_col:
        st.error("\u274c Could not detect a valid keyword column in the uploaded file.")
        st.stop()

    keywords = df[keyword_col].dropna().unique().tolist()
    openai.api_key = openai_api
    embeddings = []
    valid_keywords = []
    for kw in keywords:
        emb = get_embedding(kw.lower())
        if emb:
            embeddings.append(emb)
            valid_keywords.append(kw)

    if len(embeddings) < 2:
        st.error("Could not generate enough embeddings to cluster. Please try again.")
        st.stop()

    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - sim_matrix

    clustering = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        distance_threshold=1 - (threshold / 100),
        n_clusters=None
    ).fit(distance_matrix)

    cluster_labels = clustering.labels_

    clustered_data = pd.DataFrame({
        "Keyword": valid_keywords,
        "Cluster ID": cluster_labels
    })

    final_rows = []
    for cluster_id in sorted(clustered_data["Cluster ID"].unique()):
        kws = clustered_data[clustered_data["Cluster ID"] == cluster_id]["Keyword"].tolist()
        hub = kws[0]
        label = generate_cluster_label(kws, openai_api)
        size = len(kws)
        for kw in kws:
            final_rows.append({
                "Cluster Label": label,
                "Cluster Size": size,
                "Hub": hub,
                "Keyword": kw
            })

    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sort_values(by=["Cluster Size", "Cluster Label"], ascending=[False, True])
    st.session_state.final_df = final_df
    st.success(f"\u2705 Done! {len(final_df['Cluster Label'].unique())} clusters, {len(final_df)} keywords.")

# -----------------------------
# Preview & Export
# -----------------------------
if st.session_state.final_df is not None:
    csv = st.session_state.final_df.to_csv(index=False, encoding="utf-8")
    st.download_button("\ud83d\udcc5 Download Clustered CSV", data=csv, file_name="final_clustered_keywords.csv", mime="text/csv")
    st.dataframe(st.session_state.final_df, use_container_width=True)
