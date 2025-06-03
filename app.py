import streamlit as st
import pandas as pd
import json
import openai
import requests
import time

st.set_page_config(page_title="SERP Keyword Clustering", layout="wide")

st.title("🔍 SERP-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV, enter your API keys, and generate clean keyword clusters based on SERP overlap.")

# Upload keywords CSV
uploaded_file = st.file_uploader("📥 Upload your keywords.csv file", type="csv")

# API keys
serper_api = st.text_input("🔑 Serper API Key", type="password")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")

# Similarity threshold slider
threshold = st.slider("📊 SERP Similarity Threshold (%)", min_value=10, max_value=100, value=30) / 100

# Maintain session state
if "final_df" not in st.session_state:
    st.session_state.final_df = None

# Process button
if st.button("🚀 Run Clustering") and uploaded_file and serper_api and openai_api:
    st.info("Processing... Please wait.")
    keywords_df = pd.read_csv(uploaded_file)

    # Smarter column detection
    possible_columns = ["keyword", "Keyword", "query", "Query", "queries", "Queries"]
    keyword_col = None
    for col in keywords_df.columns:
        if col.strip() in possible_columns:
            keyword_col = col
            break
    if not keyword_col:
        keyword_col = keywords_df.columns[0]  # fallback to first column

    keywords = keywords_df[keyword_col].dropna().unique().tolist()

    headers = {"X-API-KEY": serper_api, "Content-Type": "application/json"}
    serp_data = {}

    progress = st.progress(0)
    for i, keyword in enumerate(keywords):
        response = requests.post("https://google.serper.dev/search", headers=headers, json={"q": keyword})
        if response.status_code == 200:
            data = response.json().get("organic", [])
            urls = [item.get("link") for item in data][:10]
            serp_data[keyword] = urls
        else:
            serp_data[keyword] = []
        progress.progress((i + 1) / len(keywords))
        time.sleep(1)

    # Jaccard similarity
    def jaccard(set1, set2):
        return len(set(set1) & set(set2)) / len(set(set1) | set(set2)) if set1 or set2 else 0

    # Clustering logic
    clusters = []
    unclustered = set(serp_data.keys())
    while unclustered:
        hub = unclustered.pop()
        cluster = [hub]
        to_compare = list(unclustered)
        for kw in to_compare:
            if jaccard(serp_data[hub], serp_data[kw]) >= threshold:
                cluster.append(kw)
                unclustered.remove(kw)
        clusters.append(cluster)

    # GPT labeling
    openai.api_key = openai_api
    labeled_rows = []
    for cluster in clusters:
        hub = cluster[0]
        prompt = f"""You are an SEO expert.

Given this list of keywords:
{cluster}

Group them under a simple, short and clear cluster label (2–5 words max). Do not repeat the keywords, just summarize their common theme.

Respond only with the label:"""
        try:
            res = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            label = res.choices[0].message['content'].strip()
        except Exception:
            label = f"Cluster: {hub}"

        for kw in cluster:
            labeled_rows.append({
                "Cluster Label": label,
                "Hub": hub,
                "Keyword": kw
            })

    final_df = pd.DataFrame(labeled_rows)
    st.session_state.final_df = final_df

    st.success("✅ Clustering completed!")
    st.download_button("📁 Download Clustered CSV", final_df.to_csv(index=False, encoding="utf-8"), file_name="final_clustered_keywords.csv", mime="text/csv")
    st.dataframe(final_df)

# Redisplay after rerun
if st.session_state.final_df is not None:
    st.dataframe(st.session_state.final_df)
