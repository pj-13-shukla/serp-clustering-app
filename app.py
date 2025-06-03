import streamlit as st
import pandas as pd
import json
import openai
import requests
import time

st.set_page_config(page_title="SERP Keyword Clustering", layout="wide")

st.title("🔍 SERP-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV, enter your API keys, and generate content-ready keyword clusters based on SERP overlap.")

# Upload CSV file
uploaded_file = st.file_uploader("📅 Upload your keywords.csv file", type="csv")

# API Keys
serper_api = st.text_input("🔑 Serper API Key", type="password")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")

# Threshold slider
threshold = st.slider("📊 SERP Similarity Threshold (%)", min_value=10, max_value=100, value=30) / 100

# Session state
if "final_df" not in st.session_state:
    st.session_state.final_df = None

# Run Clustering
if st.button("🚀 Run Clustering") and uploaded_file and serper_api and openai_api:
    st.info("Processing... Please wait.")
    openai.api_key = openai_api
    df = pd.read_csv(uploaded_file)

    # Flexible column detection
    colnames = df.columns.str.lower().tolist()
    keyword_col = None
    for option in ["keyword", "query", "search term", "term"]:
        if option in colnames:
            keyword_col = df.columns[colnames.index(option)]
            break
    if keyword_col is None:
        keyword_col = df.columns[0]

    keywords = df[keyword_col].dropna().astype(str).unique().tolist()
    headers = {"X-API-KEY": serper_api, "Content-Type": "application/json"}
    serp_data = {}

    progress = st.progress(0)
    for i, kw in enumerate(keywords):
        progress.progress((i + 1) / len(keywords), f"Fetching SERP for: {kw}")
        try:
            res = requests.post("https://google.serper.dev/search", headers=headers, json={"q": kw})
            data = res.json().get("organic", [])
            urls = [item.get("link") for item in data][:10]
        except:
            urls = []
        serp_data[kw] = urls
        time.sleep(1)

    # Jaccard clustering
    def jaccard(a, b):
        return len(set(a) & set(b)) / len(set(a) | set(b)) if a or b else 0

    clusters = []
    unclustered = set(keywords)
    while unclustered:
        hub = unclustered.pop()
        group = [hub]
        for kw in list(unclustered):
            if jaccard(serp_data[hub], serp_data[kw]) >= threshold:
                group.append(kw)
                unclustered.remove(kw)
        clusters.append(group)

    labeled = []
    for group in clusters:
        prompt = f"""
You're an SEO expert. Given the following list of keywords:
{group}

Group them under ONE short, general cluster name that reflects their common intent.
Avoid repetition, locations, or overly long phrases. Do not use the words 'cluster', 'near me', or any location names.

Return only the concise label.
"""
        try:
            result = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            label = result.choices[0].message['content'].strip()
        except:
            label = "General Topic"

        for kw in group:
            labeled.append({
                "Cluster Label": label,
                "Hub": group[0],
                "Keyword": kw
            })

    st.session_state.final_df = pd.DataFrame(labeled)
    st.success("✅ Clustering completed!")

# Show results and allow download
if st.session_state.final_df is not None:
    st.download_button("📅 Download Clustered CSV", st.session_state.final_df.to_csv(index=False, encoding="utf-8"), "final_clustered_keywords.csv", "text/csv")
    st.dataframe(st.session_state.final_df)
