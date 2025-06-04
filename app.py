import streamlit as st
import pandas as pd
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# --- Page Config ---
st.set_page_config(page_title="AI-Based Keyword Clustering Tool", layout="wide")
st.title("\U0001F9E0 AI-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV and get semantic, search-intent-based clusters with smart labels.")

# --- Upload Section ---
uploaded_file = st.file_uploader("📄 Upload your keywords.csv file", type="csv")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")
threshold = st.slider("✏️ Cosine Similarity Threshold", 70, 95, 80)

# --- Helper Functions ---
def get_embedding(text, client):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def generate_cluster_label(keywords, client):
    try:
        prompt = f"""
        You're an SEO assistant. Given the following keywords:
        {keywords}

        Return a short, generalized 2–4 word label that describes the group. Avoid long-tails. Just the label.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Unlabeled Cluster"

# --- Clustering Logic ---
if st.button("🚀 Run Clustering") and uploaded_file and openai_api:
    st.info("Embedding and clustering... hold tight.")
    df = pd.read_csv(uploaded_file)

    keyword_col = df.columns[0]
    keywords = df[keyword_col].dropna().unique().tolist()

    import openai as openai_sdk
    client = openai_sdk.OpenAI(api_key=openai_api)

    embeddings = []
    valid_keywords = []
    progress = st.progress(0, text="🔄 Generating embeddings...")

    for i, kw in enumerate(keywords):
        emb = get_embedding(kw, client)
        if emb:
            embeddings.append(emb)
            valid_keywords.append(kw)
        progress.progress((i + 1) / len(keywords))

    progress.empty()

    if len(embeddings) < 2:
        st.error("Could not generate enough embeddings to cluster. Please try again.")
    else:
        sim_matrix = cosine_similarity(embeddings)
        dist = 1 - sim_matrix

        clustering = AgglomerativeClustering(
            metric='precomputed',
            linkage='average',
            distance_threshold=1 - (threshold / 100),
            n_clusters=None
        ).fit(dist)

        labels = clustering.labels_
        df_result = pd.DataFrame({"Keyword": valid_keywords, "Cluster ID": labels})

        final_data = []
        for cid in sorted(df_result["Cluster ID"].unique()):
            cluster_keywords = df_result[df_result["Cluster ID"] == cid]["Keyword"].tolist()
            cluster_size = len(cluster_keywords)
            label = generate_cluster_label(cluster_keywords, client).title()

            for kw in cluster_keywords:
                final_data.append({
                    "Topic Cluster": label,
                    "Cluster Size": cluster_size,
                    "Keyword": kw
                })

        final_df = pd.DataFrame(final_data)
        final_df = final_df.sort_values(by=["Topic Cluster", "Keyword"])

        csv = final_df.to_csv(index=False, encoding="utf-8")
        st.download_button("📥 Download Clustered CSV", data=csv, file_name="clustered_keywords.csv", mime="text/csv")

        st.markdown("### 🔍 Clustered Keywords Preview")
        st.dataframe(final_df, use_container_width=True)
