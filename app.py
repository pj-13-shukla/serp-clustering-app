import streamlit as st
import pandas as pd
import openai
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------
# Setup
# ----------------------------
st.set_page_config(page_title="AI-Based Keyword Clustering Tool", layout="wide")
st.title("🧠 AI-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV and get semantic, search-intent-based clusters with smart labels.")

uploaded_file = st.file_uploader("📄 Upload your keywords.csv file", type="csv")
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
threshold = st.slider("📏 Cosine Similarity Threshold", 70, 95, 80)

# ----------------------------
# OpenAI Clients
# ----------------------------
client = openai.OpenAI(api_key=openai_api_key)

def get_embedding(text):
    try:
        res = client.embeddings.create(model="text-embedding-3-small", input=text)
        return res.data[0].embedding
    except:
        return None

def generate_label(keywords):
    prompt = f"""
You are an SEO assistant. Given the following keywords:

{keywords}

Return a short, generalized 2–4 word label that describes the group. Avoid using long-tails or exact matches. Use proper casing (like "THCA Conversion" or "Bong Size") and avoid duplicates.
Only return the label — no intro, no explanation.
"""
    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content.strip().title()
    except:
        return "Unlabeled Cluster"

# ----------------------------
# Main Clustering
# ----------------------------
if st.button("🚀 Run Clustering") and uploaded_file and openai_api_key:
    df = pd.read_csv(uploaded_file)
    colnames = [col.lower().strip() for col in df.columns]
    keyword_col = next((col for col in ['keyword', 'keywords', 'query', 'queries'] if col in colnames), None)
    keyword_col = df.columns[colnames.index(keyword_col)] if keyword_col else df.columns[0]

    raw_keywords = df[keyword_col].dropna().unique().tolist()

    with st.spinner("🔍 Generating embeddings..."):
        keywords = []
        vectors = []
        for kw in raw_keywords:
            emb = get_embedding(kw.lower().strip())
            if emb:
                keywords.append(kw.strip())
                vectors.append(emb)

    if len(keywords) < 2:
        st.error("❌ Could not generate enough embeddings. Try again.")
    else:
        similarity = cosine_similarity(vectors)
        dist = 1 - similarity

        clustering = AgglomerativeClustering(
            affinity='precomputed',
            linkage='average',
            distance_threshold=1 - (threshold / 100),
            n_clusters=None
        ).fit(dist)

        labels = clustering.labels_
        result_df = pd.DataFrame({"Keyword": keywords, "Cluster ID": labels})

        final_rows = []
        for cid in sorted(result_df["Cluster ID"].unique()):
            cluster_keywords = result_df[result_df["Cluster ID"] == cid]["Keyword"].tolist()
            cluster_size = len(cluster_keywords)
            label = generate_label(cluster_keywords)

            for kw in cluster_keywords:
                final_rows.append({
                    "Topic Cluster": label,
                    "Cluster Size": cluster_size,
                    "Keyword": kw
                })

        final_df = pd.DataFrame(final_rows)
        final_df = final_df.sort_values(by=["Topic Cluster", "Keyword"])
        csv = final_df.to_csv(index=False)

        st.success(f"✅ Done! {final_df['Topic Cluster'].nunique()} clusters found.")
        st.download_button("📥 Download Clustered Keywords CSV", data=csv, file_name="clustered_keywords.csv", mime="text/csv")
        st.dataframe(final_df, use_container_width=True)
