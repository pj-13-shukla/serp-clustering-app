import streamlit as st
import pandas as pd
import json
import openai
import requests
import time
import re
from collections import defaultdict

st.set_page_config(page_title="SERP Keyword Clustering", layout="wide")

st.title("🔍 SERP-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV, enter your API keys, and generate clean, content-ready keyword clusters based on SERP overlap.")

# Upload keywords CSV
uploaded_file = st.file_uploader("📥 Upload your keywords.csv file", type="csv")

# API keys
serper_api = st.text_input("🔑 Serper API Key", type="password")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")

# Similarity threshold slider
threshold = st.slider("🔧 SERP Similarity Threshold (%)", min_value=10, max_value=100, value=30) / 100

# Progress text
progress_bar = st.progress(0)
progress_text = st.empty()

# Main Clustering Process
if st.button("🚀 Run Clustering") and uploaded_file and serper_api and openai_api:
    st.info("Processing... Please wait.")
    openai.api_key = openai_api

    # Read keywords CSV
    df = pd.read_csv(uploaded_file)
    colname = df.columns[0]
    keywords = df[colname].dropna().unique().tolist()

    # Fetch SERP data
    headers = {"X-API-KEY": serper_api, "Content-Type": "application/json"}
    serp_data = {}

    for i, kw in enumerate(keywords):
        response = requests.post("https://google.serper.dev/search", headers=headers, json={"q": kw})
        if response.status_code == 200:
            serp_data[kw] = [r.get("link") for r in response.json().get("organic", [])[:10]]
        else:
            serp_data[kw] = []
        progress_text.text(f"🔎 Fetching SERP for: {kw}")
        progress_bar.progress((i + 1) / len(keywords))
        time.sleep(1)

    # Clustering using Jaccard
    def jaccard(set1, set2):
        return len(set(set1) & set(set2)) / len(set(set1) | set(set2)) if set1 or set2 else 0

    clusters = []
    ungrouped = set(keywords)
    while ungrouped:
        hub = ungrouped.pop()
        cluster = [hub]
        for other in list(ungrouped):
            if jaccard(serp_data[hub], serp_data[other]) >= threshold:
                cluster.append(other)
                ungrouped.remove(other)
        clusters.append(cluster)

    # Generate cluster labels via GPT and deduplicate similar ones
    def normalize_label(label):
        label = label.lower()
        label = re.sub(r'[^a-z\s]', '', label)
        label = re.sub(r'\s+', ' ', label).strip()
        return label

    label_map = {}
    final_rows = []

    for cluster in clusters:
        hub = cluster[0]
        try:
            res = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Generate one short, meaningful cluster name for these keywords:\n{cluster}"}],
                temperature=0.3
            )
            label = res.choices[0].message['content'].strip()
        except Exception:
            label = "Unnamed Cluster"

        norm_label = normalize_label(label)
        if norm_label in label_map:
            label = label_map[norm_label]
        else:
            label_map[norm_label] = label

        for kw in cluster:
            final_rows.append({
                "Cluster Label": label,
                "Hub": hub,
                "Keyword": kw
            })

    # Final Output
    final_df = pd.DataFrame(final_rows)
    st.success("✅ Clustering completed!")
    st.dataframe(final_df)

    csv = final_df.to_csv(index=False, encoding="utf-8")
    st.download_button("📥 Download Clustered CSV", csv, file_name="final_clustered_keywords.csv", mime="text/csv")
