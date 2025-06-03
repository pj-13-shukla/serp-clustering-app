import streamlit as st
import pandas as pd
import json
import openai
import requests
import time

st.set_page_config(page_title="SERP Keyword Clustering", layout="wide")

st.title("\U0001F50D SERP-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV, enter your API keys, and generate content-ready keyword clusters based on SERP overlap.")

uploaded_file = st.file_uploader("\U0001F4E5 Upload your keywords.csv file", type="csv")
serper_api = st.text_input("\U0001F511 Serper API Key", type="password")
openai_api = st.text_input("\U0001F511 OpenAI API Key", type="password")
threshold = st.slider("\U0001F527 SERP Similarity Threshold (%)", min_value=10, max_value=100, value=30) / 100

progress_bar = st.progress(0)

if st.button("\U0001F680 Run Clustering") and uploaded_file and serper_api and openai_api:
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
You are an SEO expert. Given the following list of keywords:
{cluster}
Generate a short, general, meaningful label that describes their common intent or category. Avoid long-tail keywords or overly specific phrases. Respond with only the label text, nothing else.
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
    st.success("\u2705 Clustering completed!")

    csv_data = final_df.to_csv(index=False, encoding="utf-8")
    st.download_button("\U0001F4E5 Download Clustered CSV", csv_data, file_name="final_clustered_keywords.csv", mime="text/csv")
    st.dataframe(final_df)
