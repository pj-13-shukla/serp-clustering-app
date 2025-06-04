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

# ----------------------------- Setup
st.set_page_config(page_title="AI-Based Keyword Clustering", layout="wide")
st.title("🧠 AI Keyword Clustering (Roger Optimized)")
st.markdown("Upload keyword CSV and generate accurate, search-intent-based clusters with smart GPT labeling.")

# ----------------------------- Inputs
uploaded_file = st.file_uploader("📥 Upload keyword file (.csv)", type="csv")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")
threshold = st.slider("🔗 Similarity Threshold", 70, 95, 80)

if "final_df" not in st.session_state:
    st.session_state.final_df = None

# ----------------------------- Clean & Capitalize
def clean_label(text):
    acronyms = ['thc', 'thca', 'cbd', 'cbg']
    for word in acronyms:
        pattern = re.compile(rf'\b{word}\b', re.IGNORECASE)
        text = pattern.sub(word.upper(), text)
    return text.title()

# ----------------------------- Embedding Fetch
def get_embedding(text):
    try:
        res = openai.Embedding.create(model="text-embedding-3-small", input=text)
        return res["data"][0]["embedding"]
    except Exception:
        return None

# ----------------------------- GPT Label
def generate_cluster_label(keywords, api_key):
    openai.api_key = api_key
    prompt = f"""
You're an SEO assistant. Given these keywords:

{keywords}

Return a short, general 2–4 word label describing the shared topic. Do NOT use full long-tails. Just return the label.
"""
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return res.choices[0].message["content"].strip()
    except:
        return ""

# ----------------------------- Run Clustering
if st.button("🚀 Run Clustering") and uploaded_file and openai_api:
    st.info("Embedding and clustering... hold tight.")
    df = pd.read_csv(uploaded_file)

    # Detect keyword column
    column_names = [col.lower().strip() for col in df.columns]
    keyword_col = next((col for col in ['keyword', 'keywords', 'query', 'queries'] if col in column_names), df.columns[0])
    keywords = df[keyword_col].dropna().unique().tolist()

    # Generate embeddings
    openai.api_key = openai_api
    embeddings, valid_keywords = [], []
    for kw in keywords:
        emb = get_embedding(kw.strip())
        if emb:
            embeddings.append(emb)
            valid_keywords.append(kw.strip())
        time.sleep(0.5)

    if len(embeddings) < 2:
        st.error("❌ Not enough keywords with valid embeddings.")
    else:
        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = 1 - sim_matrix
        threshold_val = 1 - (threshold / 100)

        model = AgglomerativeClustering(
            affinity="precomputed",
            linkage="average",
            distance_threshold=threshold_val,
            n_clusters=None
        ).fit(dist_matrix)

        clustered = pd.DataFrame({
            "Keyword": valid_keywords,
            "Cluster ID": model.labels_
        })

        final_data = []
        for cluster_id in sorted(clustered["Cluster ID"].unique()):
            kws = clustered[clustered["Cluster ID"] == cluster_id]["Keyword"].tolist()
            hub = kws[0]
            gpt_label = generate_cluster_label(kws, openai_api)
            label = clean_label(gpt_label) if gpt_label else clean_label(hub)

            for kw in kws:
                final_data.append({
                    "Cluster ID": cluster_id,
                    "Cluster Size": len(kws),
                    "Cluster Label": f"**{label}**",
                    "Hub": hub,
                    "Keyword": kw
                })

        final_df = pd.DataFrame(final_data)
        final_df = final_df.sort_values(by=["Cluster Size", "Cluster Label"], ascending=[False, True])
        st.session_state.final_df = final_df
        st.success(f"✅ Done! {len(final_df['Cluster Label'].unique())} clusters created.")

# ----------------------------- Output
if st.session_state.final_df is not None:
    csv = st.session_state.final_df.to_csv(index=False, encoding="utf-8")
    st.download_button("📥 Download Final CSV", data=csv, file_name="clustered_keywords.csv", mime="text/csv")
    st.markdown("### 🧾 Clustered Keywords Preview")
    st.dataframe(st.session_state.final_df, use_container_width=True)
