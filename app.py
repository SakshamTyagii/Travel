import pandas as pd
import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Update file paths to use correct directory structure and OS-agnostic paths
data_dir = os.path.join(os.path.dirname(__file__), "data")
appsheet_csv = os.path.join(data_dir, "appsheet.csv")

# Load the appsheet.csv file
appsheet_df = pd.read_csv(appsheet_csv)

# Initialize the NLP model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare a list of searchable terms (Location + synonyms)
searchable_items = []
search_terms = []
for _, row in appsheet_df.iterrows():
    city = row["City"]
    location = row["Location"]
    synonyms = row["synonyms"].split(", ") if pd.notna(row["synonyms"]) and row["synonyms"] else []
    
    # Add the correct location name
    search_terms.append(location.lower())
    searchable_items.append({
        "display": f"{location} ({city})",
        "value": location,
        "city": city.lower()
    })
    # Add each synonym as a searchable term
    for synonym in synonyms:
        search_terms.append(synonym.lower())
        searchable_items.append({
            "display": f"{location} ({city})",
            "value": location,
            "city": city.lower()
        })

# Compute embeddings for all search terms
embeddings = model.encode(search_terms, convert_to_tensor=True)

@app.route('/')
def serve_index():
    return send_file("index.html")

@app.route('/get_locations', methods=['GET'])
def get_locations():
    locations_data = appsheet_df.to_dict('records')
    return jsonify(locations_data)

@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('query', '').strip().lower()
    if not query:
        return jsonify([])

    # Compute embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarities between the query and all search terms
    similarities = util.cos_sim(query_embedding, embeddings)[0]

    # Combine similarities with searchable items
    matches = []
    for i, item in enumerate(searchable_items):
        score = similarities[i].item()
        # Boost score if the city matches the query
        if item["city"] in query:
            score += 0.2  # Add a boost for city match
        matches.append({
            "display": item["display"],
            "value": item["value"],
            "score": score
        })

    # Filter matches with score > 0.5, sort by score, limit to top 5
    matches = [match for match in matches if match["score"] > 0.5]
    matches.sort(key=lambda x: x["score"], reverse=True)
    matches = matches[:5]

    # Remove duplicates by display value
    seen = set()
    unique_matches = []
    for match in matches:
        if match["display"] not in seen:
            seen.add(match["display"])
            unique_matches.append({"display": match["display"], "value": match["value"]})

    return jsonify(unique_matches)

if __name__ == "__main__":
    app.run(debug=True, port=5000)