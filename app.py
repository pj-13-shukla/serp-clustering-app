import streamlit as st
import pandas as pd
import openai
import requests
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Setup
st.set_page_config(page_title="AI-Based Keyword Clustering Tool", layout="wide")
st.title("AI-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV and get semantic, search-intent-based clusters with smart labels.")

# Inputs
uploaded_file = st.file_uploader("Upload your keywords.csv file", type="csv")
openai_api = st.text_input("OpenAI API Key", type="password")
threshold = st.slider("Cosine Similarity Threshold", 70, 95, 80)

if "final_df" not in st.session_state:
    st.session_state.final_df = None

# Embedding function
def get_embedding(text):
    try:
        res = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        return res["data"][0]["embedding"]
    except:
        return None

# GPT Labeling
def generate_cluster_label(keywords, api_key):
    openai.api_key = api_key
    prompt = f"""
You are an SEO assistant. Given the following keywords:

{keywords}

Return a short, clear 2-4 word cluster label that captures their shared search intent.
Avoid duplicating entire keyword phrases. Use abstract or categorical naming when possible.
"""
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message["content"].strip()
    except:
        return "Unlabeled Cluster"

# Main Logic
if st.button("Run Clustering") and uploaded_file and openai_api:
    df = pd.read_csv(uploaded_file)

    col_names = [col.lower().strip() for col in df.columns]
    keyword_col = next((col for col in ['keyword', 'keywords', 'query', 'queries'] if col in col_names), None)
    keyword_col = df.columns[col_names.index(keyword_col)] if keyword_col else df.columns[0]

    keywords = df[keyword_col].dropna().unique().tolist()
    openai.api_key = openai_api

    embeddings = []
    cleaned_keywords = []
    for kw in keywords:
        emb = get_embedding(kw.strip().lower())
        if emb:
            embeddings.append(emb)
            cleaned_keywords.append(kw.strip())

    if len(embeddings) < 2:
        st.error("Could not generate enough embeddings to cluster. Please try again.")
    else:
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        clustering = AgglomerativeClustering(
            affinity='precomputed',
            linkage='average',
            distance_threshold=1 - (threshold / 100),
            n_clusters=None
        ).fit(distance_matrix)

        clustered_data = pd.DataFrame({
            "Keyword": cleaned_keywords,
            "Cluster ID": clustering.labels_
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

        final_df = pd.DataFrame(final_rows)
        final_df = final_df.sort_values(by=["Cluster Size", "Cluster Label"], ascending=[False, True])
        st.session_state.final_df = final_df
        st.success(f"Clustering complete! {len(final_df['Cluster Label'].unique())} clusters, {len(final_df)} keywords.")

# Display Output
if st.session_state.final_df is not None:
    csv_data = st.session_state.final_df.to_csv(index=False, encoding="utf-8")
    st.download_button("Download Clustered CSV", data=csv_data, file_name="clustered_keywords.csv", mime="text/csv")
    st.dataframe(st.session_state.final_df, use_container_width=True)
