import streamlit as st
import pandas as pd
import json
import openai
import requests
import time

st.set_page_config(page_title="SERP Keyword Clustering", layout="wide")
st.title("🔍 SERP-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV, enter your API keys, and generate simplified keyword clusters based on SERP overlap.")

# Upload CSV
uploaded_file = st.file_uploader("📥 Upload your keywords.csv file", type="csv")

# API keys
serper_api = st.text_input("🔑 Serper API Key", type="password")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")

# Similarity threshold
threshold = st.slider("🔧 SERP Similarity Threshold (%)", min_value=10, max_value=100, value=30) / 100

# Helper function: detect keyword column
def detect_keyword_column(df):
    for col in df.columns:
        if col.strip().lower() in ["keyword", "keywords", "query", "queries"]:
            return col
    return df.columns[0]  # fallback to first column

# Clustering logic
def jaccard(set1, set2):
    return len(set(set1) & set(set2)) / len(set(set1) | set(set2)) if set1 or set2 else 0

# Run
if st.button("🚀 Run Clustering") and uploaded_file and serper_api and openai_api:
    st.info("Processing... Please wait.")
    keywords_df = pd.read_csv(uploaded_file)
    keyword_col = detect_keyword_column(keywords_df)
    keywords = keywords_df[keyword_col].dropna().unique().tolist()

    headers = {"X-API-KEY": serper_api, "Content-Type": "application/json"}
    serp_data = {}

    progress = st.progress(0)
    for i, kw in enumerate(keywords):
        st.text(f"🔍 Fetching SERP for: {kw}")
        response = requests.post("https://google.serper.dev/search", headers=headers, json={"q": kw})
        urls = [r.get("link") for r in response.json().get("organic", [])][:10] if response.status_code == 200 else []
        serp_data[kw] = urls
        time.sleep(1)
        progress.progress((i+1)/len(keywords))

    clusters = []
    unclustered = set(serp_data)
    while unclustered:
        hub = unclustered.pop()
        cluster = [hub]
        for other in list(unclustered):
            if jaccard(serp_data[hub], serp_data[other]) >= threshold:
                cluster.append(other)
                unclustered.remove(other)
        clusters.append(cluster)

    openai.api_key = openai_api
    labeled_rows = []
    for cluster in clusters:
        hub = cluster[0]
        prompt = f"Group the following search keywords into a single concise, high-level SEO content theme. Give a short and clean title for this group: {cluster}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            label = response.choices[0].message['content'].strip().replace('\n', ' ')
        except Exception:
            label = "Unnamed Cluster"

        for kw in cluster:
            labeled_rows.append({
                "Cluster Label": label,
                "Hub": hub,
                "Keyword": kw
            })

    final_df = pd.DataFrame(labeled_rows)
    st.success("✅ Clustering completed!")
    st.download_button("📥 Download Clustered CSV", final_df.to_csv(index=False, encoding="utf-8"), file_name="final_clustered_keywords.csv", mime="text/csv")
    st.dataframe(final_df)
