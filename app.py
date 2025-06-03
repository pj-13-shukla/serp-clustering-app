import streamlit as st
import pandas as pd
import openai
import requests
import time

# Page setup
st.set_page_config(page_title="SERP Keyword Clustering", layout="wide")
st.title("🔍 SERP-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV, enter your API keys, and generate content-ready keyword clusters based on SERP overlap.")

# Upload + API Inputs
uploaded_file = st.file_uploader("📤 Upload your keywords.csv file", type="csv")
serper_api = st.text_input("🔑 Serper API Key", type="password")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")
threshold = st.slider("🛠️ SERP Similarity Threshold (%)", 10, 100, 30) / 100

# Initialize session state
if 'final_df' not in st.session_state:
    st.session_state.final_df = None

progress_bar = st.progress(0)

# Main clustering logic
if st.button("🚀 Run Clustering") and uploaded_file and serper_api and openai_api:
    st.info("Processing... Please wait.")
    keywords_df = pd.read_csv(uploaded_file)

    # Detect correct keyword column
    column_names = [col.lower().strip() for col in keywords_df.columns]
    keyword_col = next((col for col in ['keyword', 'keywords', 'query', 'queries'] if col in column_names), None)
    keyword_col = keywords_df.columns[column_names.index(keyword_col)] if keyword_col else keywords_df.columns[0]

    keywords = keywords_df[keyword_col].dropna().unique().tolist()

    # Fetch SERP results
    headers = {"X-API-KEY": serper_api, "Content-Type": "application/json"}
    serp_data = {}
    for i, keyword in enumerate(keywords):
        progress_bar.progress((i + 1) / len(keywords))
        try:
            response = requests.post("https://google.serper.dev/search", headers=headers, json={"q": keyword})
            urls = [item.get("link") for item in response.json().get("organic", [])][:10]
        except Exception:
            urls = []
        serp_data[keyword] = urls
        time.sleep(1)

    # Jaccard similarity
    def jaccard(set1, set2):
        return len(set(set1) & set(set2)) / len(set(set1) | set(set2)) if set1 or set2 else 0

    # Cluster keywords
    clusters = []
    unclustered = set(keywords)
    while unclustered:
        hub = unclustered.pop()
        cluster = [hub]
        to_compare = list(unclustered)
        for kw in to_compare:
            if jaccard(serp_data[hub], serp_data[kw]) >= threshold:
                cluster.append(kw)
                unclustered.remove(kw)
        clusters.append(cluster)

    # GPT-4o for labeling
    openai.api_key = openai_api
    labeled_rows = []
    for i, cluster in enumerate(clusters):
        prompt = f"""
You are an expert in SEO and keyword categorization.

Given this list of keywords:
{cluster}

Return a short, generalized label (2-4 words) that best describes the group. Avoid repeating full keywords or using long-tail terms like "near me", "best", or "top". Keep it clean and useful for content planning.

Only return the label, no punctuation or explanation.
"""
        try:
            res = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            label = res.choices[0].message["content"].strip()
            if not label:
                label = f"Cluster {i+1}"
        except Exception:
            label = f"Cluster {i+1}"

        hub = cluster[0]
        for kw in cluster:
            labeled_rows.append({
                "Cluster Label": label,
                "Hub": hub,
                "Keyword": kw
            })

    # Final DataFrame
    final_df = pd.DataFrame(labeled_rows)
    st.session_state.final_df = final_df  # Save for reuse

    st.success("✅ Clustering complete!")

# Display and download section
if st.session_state.final_df is not None:
    # Prepare download data
    csv_data = st.session_state.final_df.to_csv(index=False, encoding="utf-8")

    # Download button
    st.download_button(
        label="📥 Download Clustered CSV",
        data=csv_data,
        file_name="final_clustered_keywords.csv",
        mime="text/csv"
    )

    # Show clean table (bold cluster label for visual appeal)
    display_df = st.session_state.final_df.copy()
    display_df["Cluster Label"] = display_df["Cluster Label"].apply(lambda x: f"**{x}**")
    st.markdown("### 📊 Preview of Clustered Keywords")
    st.dataframe(display_df, use_container_width=True)
