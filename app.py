import streamlit as st
import pandas as pd
import openai
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import time

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(page_title="Semantic Keyword Clustering", layout="wide")
st.title("🧠 AI-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV and get semantic, search-intent-based clusters with smart labels.")

uploaded_file = st.file_uploader("📄 Upload your keywords.csv file", type="csv")
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
similarity_threshold = st.slider("🔍 Cosine Similarity Threshold", 70, 95, 80)

# -----------------------------
# Helper Functions
# -----------------------------
def get_embedding(text, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error for '{text}': {e}")
        return None

def generate_cluster_label(keywords, api_key):
    prompt = f"""
You're an SEO assistant. Given the following keywords:

{keywords}

Return a short, capitalized 2–4 word label that describes the group. Avoid exact matches. Use clear and generalized wording. For example:
- For ['how to clean a bowl', 'cleaning bong bowls'], use "Bowl Cleaning"
- For ['thca to thc', 'convert thca'], use "THCA Conversion"
"""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Labeling error: {e}")
        return "Unlabeled Cluster"

# -----------------------------
# Clustering Process
# -----------------------------
if st.button("🚀 Run Clustering") and uploaded_file and openai_api_key:
    df = pd.read_csv(uploaded_file)
    keyword_column = df.columns[0]
    keywords = df[keyword_column].dropna().unique().tolist()

    st.info("🔄 Generating embeddings, please wait...")
    embeddings = []
    valid_keywords = []
    for kw in keywords:
        embedding = get_embedding(kw, openai_api_key)
        if embedding:
            embeddings.append(embedding)
            valid_keywords.append(kw)
        time.sleep(0.2)

    if len(embeddings) < 2:
        st.error("❌ Not enough valid embeddings to proceed.")
    else:
        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = 1 - sim_matrix
        clustering = AgglomerativeClustering(
            affinity='precomputed',
            linkage='average',
            distance_threshold=1 - (similarity_threshold / 100),
            n_clusters=None
        )
        cluster_ids = clustering.fit_predict(dist_matrix)

        clustered_df = pd.DataFrame({
            "Keyword": valid_keywords,
            "Cluster ID": cluster_ids
        })

        output_rows = []
        for cid in sorted(clustered_df["Cluster ID"].unique()):
            group = clustered_df[clustered_df["Cluster ID"] == cid]
            kws = group["Keyword"].tolist()
            label = generate_cluster_label(kws, openai_api_key)
            for kw in kws:
                output_rows.append({
                    "Topic Cluster": label,
                    "Cluster Size": len(kws),
                    "Keyword": kw
                })

        final_output = pd.DataFrame(output_rows)
        final_output = final_output.sort_values(by=["Topic Cluster", "Keyword"]).reset_index(drop=True)

        csv = final_output.to_csv(index=False).encode("utf-8")
        st.success(f"✅ Done! {len(final_output)} keywords clustered into {final_output['Topic Cluster'].nunique()} clusters.")
        st.download_button("📥 Download Result CSV", csv, "final_clustered_keywords.csv", "text/csv")
        st.dataframe(final_output, use_container_width=True)
