import streamlit as st
import pandas as pd
import json
import openai
import requests
import time

st.set_page_config(page_title="SERP Keyword Clustering", layout="wide")
st.title("🔍 SERP-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV, enter your API keys, and generate content-ready keyword clusters based on SERP overlap.")

uploaded_file = st.file_uploader("📥 Upload your keywords.csv file", type="csv")
serper_api = st.text_input("🔑 Serper API Key", type="password")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")
threshold = st.slider("🛠 SERP Similarity Threshold (%)", min_value=10, max_value=100, value=30) / 100

progress_bar = st.progress(0)
session_state = st.session_state

if 'final_df' not in session_state:
    session_state.final_df = None

if st.button("🚀 Run Clustering") and uploaded_file and serper_api and openai_api:
    st.info("Processing... Please wait.")
    keywords_df = pd.read_csv(uploaded_file)
    column_names = [col.lower().strip() for col in keywords_df.columns]

    keyword_col = None
    for name in ['keyword', 'keywords', 'query', 'queries']:
        if name in column_names:
            keyword_col = keywords_df.columns[column_names.index(name)]
            break
    if not keyword_col:
        keyword_col = keywords_df.columns[0]

    keywords = keywords_df[keyword_col].dropna().unique().tolist()
    headers = {"X-API-KEY": serper_api, "Content-Type": "application/json"}
    serp_data = {}

    for i, keyword in enumerate(keywords):
        progress_bar.progress((i + 1) / len(keywords))
        response = requests.post("https://google.serper.dev/search", headers=headers, json={"q": keyword})
        if response.status_code == 200:
            urls = [item.get("link") for item in response.json().get("organic", [])][:10]
            serp_data[keyword] = urls
        else:
            serp_data[keyword] = []
        time.sleep(1)

    def jaccard(set1, set2):
        return len(set(set1) & set(set2)) / len(set(set1) | set(set2)) if set1 or set2 else 0

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

    openai.api_key = openai_api
    labeled_rows = []
    for i, cluster in enumerate(clusters):
        try:
            prompt = f"""
You are an SEO expert.

Here is a list of keywords:
{cluster}

Group them under one general, short, meaningful topic label that best describes their shared search intent (e.g., “Asphalt Services” or “Concrete Contractors”).

Do NOT include specific brand names or locations. Just respond with the label only.
"""
            res = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            label = res.choices[0].message['content'].strip()
        except Exception:
            label = f"Cluster {i+1}"

        hub = cluster[0]
        for kw in cluster:
            labeled_rows.append({
                "Cluster Label": label,
                "Hub": hub,
                "Keyword": kw
            })

    final_df = pd.DataFrame(labeled_rows)
    session_state.final_df = final_df
    st.success("✅ Clustering completed!")

if session_state.final_df is not None:
    csv_data = session_state.final_df.to_csv(index=False, encoding="utf-8")
    st.download_button("📥 Download Clustered CSV", csv_data, file_name="final_clustered_keywords.csv", mime="text/csv")
    st.dataframe(session_state.final_df)
