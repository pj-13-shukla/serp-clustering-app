import streamlit as st
import pandas as pd
import openai
import requests
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import re

# -----------------------------
# Setup & Config
# -----------------------------
st.set_page_config(page_title="Semantic Keyword Clustering", layout="wide")
st.title("🧠 AI-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV and get semantic, search-intent-based clusters with smart labels.")

# Upload CSV + API inputs
uploaded_file = st.file_uploader("📤 Upload your keywords.csv file", type="csv")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")
threshold = st.slider("🧠 Cosine Similarity Threshold", 70, 95, 80)

# Initialize session state
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
# Cluster Label Generator (GPT)
# -----------------------------
def generate_cluster_label(keywords, api_key):
    openai.api_key = api_key
    prompt = f"""
You're an SEO assistant. Given the following keywords:

{keywords}

Return a short, generalized 2–4 word label that describes the group. Avoid using long-tails or exact matches. Just return the label, nothing else.
"""
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message["content"].strip()
    except Exception:
        return "Unlabeled Cluster"

# -----------------------------
# Main Clustering Logic
# -----------------------------
if st.button("🚀 Run Clustering") and uploaded_file and openai_api:
    st.info("Generating embeddings and clustering keywords. This may take a minute...")
    df = pd.read_csv(uploaded_file)

    # Detect keyword column
    column_names = [col.lower().strip() for col in df.columns]
    keyword_col = next((col for col in ['keyword', 'keywords', 'query', 'queries'] if col in column_names), None)
    keyword_col = df.columns[column_names.index(keyword_col)] if keyword_col else df.columns[0]

    keywords = df[keyword_col].dropna().unique().tolist()

    # Get embeddings
    openai.api_key = openai_api
    with st.spinner("Embedding keywords..."):
        embeddings = []
        cleaned_keywords = []
        for kw in keywords:
            kw_clean = kw.lower().strip()
            embedding = get_embedding(kw_clean)
            if embedding:
                embeddings.append(embedding)
                cleaned_keywords.append(kw.strip())

    if len(embeddings) < 2:
        st.error("Could not generate enough embeddings to cluster. Please try again.")
    else:
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        norm_thresh = threshold / 100

        # Convert similarity matrix to distance matrix
        distance_matrix = 1 - similarity_matrix

        # Clustering
        clustering = AgglomerativeClustering(
            affinity='precomputed',
            linkage='average',
            distance_threshold=1 - norm_thresh,
            n_clusters=None
        ).fit(distance_matrix)

        cluster_labels = clustering.labels_

        # Group by clusters
        clustered_data = pd.DataFrame({
            "Keyword": cleaned_keywords,
            "Cluster ID": cluster_labels
        })

        final_rows = []
        for cluster_id in sorted(clustered_data["Cluster ID"].unique()):
            cluster_keywords = clustered_data[clustered_data["Cluster ID"] == cluster_id]["Keyword"].tolist()
            cluster_size = len(cluster_keywords)
            hub = cluster_keywords[0]
            label = generate_cluster_label(cluster_keywords, openai_api)

            for kw in cluster_keywords:
                final_rows.append({
                    "Cluster Label": label.title(),
                    "Cluster Size": cluster_size,
                    "Hub": hub,
                    "Keyword": kw
                })

        # Final output
        final_df = pd.DataFrame(final_rows)
        final_df = final_df.sort_values(by=["Cluster Size", "Cluster Label"], ascending=[False, True])
        st.session_state.final_df = final_df
        st.success(f"✅ Clustering complete! {len(final_df['Cluster Label'].unique())} clusters, {len(final_df)} keywords total.")

# -----------------------------
# Display Output
# -----------------------------
if st.session_state.final_df is not None:
    csv_data = st.session_state.final_df.to_csv(index=False, encoding="utf-8")
    st.download_button("📥 Download Clustered CSV", data=csv_data, file_name="clustered_keywords.csv", mime="text/csv")

    st.dataframe(st.session_state.final_df, use_container_width=True)
