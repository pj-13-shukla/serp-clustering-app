import json
import pandas as pd
from itertools import combinations

# Configurable threshold (can be user input in the future)
SIMILARITY_THRESHOLD = 0.3

# Load SERP data
with open("serp_results.json", "r") as f:
    serp_data = json.load(f)

# Clean out keywords with no URLs
serp_data = {k: v for k, v in serp_data.items() if isinstance(v, list) and len(v) > 0}

# Initialize clustering
unclustered = set(serp_data.keys())
clusters = []

def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set2))
    union = len(set(set1).union(set2))
    return intersection / union if union != 0 else 0

while unclustered:
    hub = unclustered.pop()
    cluster = [hub]

    to_check = list(unclustered)
    for keyword in to_check:
        sim = jaccard_similarity(serp_data[hub], serp_data[keyword])
        if sim >= SIMILARITY_THRESHOLD:
            cluster.append(keyword)
            unclustered.remove(keyword)

    clusters.append(cluster)

# Prepare output rows
output_rows = []
for cluster in clusters:
    hub = cluster[0]
    for keyword in cluster:
        urls = serp_data.get(keyword, [])
        output_rows.append({
            "Hub": hub,
            "Keyword": keyword,
            "URLs": "\n".join(urls)
        })

# Convert to DataFrame and save
df = pd.DataFrame(output_rows)
df.to_csv("clustered_keywords.csv", index=False)

print("✅ Clustering completed. Output saved to clustered_keywords.csv")
