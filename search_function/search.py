import pandas as pd
import os
import pickle
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import torch
from fuzzywuzzy import fuzz
from pytrie import StringTrie

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Update file paths to use correct directory structure and OS-agnostic paths
# For GCP deployment, we need to ensure data files are in the same directory as the app
is_cloud_run = os.environ.get('K_SERVICE') is not None
if is_cloud_run:
    # In Cloud Run, we'll use the absolute path to the directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        # If data directory doesn't exist in the expected location, 
        # try to find it in the parent directory
        parent_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        if os.path.exists(parent_data_dir):
            data_dir = parent_data_dir
else:
    # Local development
    data_dir = os.path.join(os.path.dirname(__file__), "data")

appsheet_csv = os.path.join(data_dir, "appsheet.csv")
embeddings_file = os.path.join(data_dir, "embeddings.pkl")

# Print paths for debugging
print(f"Data directory: {data_dir}")
print(f"Appsheet CSV: {appsheet_csv}")
print(f"Embeddings file: {embeddings_file}")

# Load the appsheet.csv file
appsheet_df = pd.read_csv(appsheet_csv)

# Initialize the NLP model with all-mpnet-base-v2
model = SentenceTransformer("all-mpnet-base-v2")

# Airport codes mapping (simplified for demonstration; you can expand this)
airport_codes = {
    "indira gandhi international airport": "DEL",
    "kempegowda international airport": "BLR",
    "sri guru ram dass jee international airport": "ATQ",
    "maharishi valmiki international airport ayodhya dham": "AYJ"
}

# Function to generate additional synonyms (e.g., transliterations, category variations)
def generate_additional_synonyms(location, city, layer):
    synonyms = []
    location_lower = location.lower()
    city_lower = city.lower()
    
    # Add city + category variation (e.g., "Delhi Airport")
    if "airports" in layer.lower():
        synonyms.append(f"{city_lower} airport")
    elif "religious" in layer.lower():
        synonyms.append(f"{city_lower} temple")
    
    # Add transliteration-like variations (simplified)
    if "indira gandhi" in location_lower:
        synonyms.append(location_lower.replace("indira gandhi", "indra ghandi"))
    
    # Add partial name (e.g., "Indira Gandhi" for "Indira Gandhi International Airport")
    parts = location_lower.split()
    if len(parts) > 2:
        synonyms.append(" ".join(parts[:2]))  # First two words
    
    return synonyms

# Prepare a list of searchable terms (Location + synonyms)
searchable_items = []  # Deduplicated list of items
search_terms = []  # List of terms (locations + synonyms)
term_to_item_idx = []  # Maps indices in search_terms to indices in searchable_items
trie = StringTrie()  # For prefix matching

# N-gram dictionary for partial matching (trigrams)
ngram_dict = {}

def generate_ngrams(text, n=3):
    text = text.lower()
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]

# Build searchable items and terms
for idx, row in appsheet_df.iterrows():
    city = row["City"]
    location = row["Location"]
    layer = row["Layer"] if "Layer" in row and pd.notna(row["Layer"]) else ""
    synonyms = row["synonyms"].split(", ") if pd.notna(row["synonyms"]) and row["synonyms"] else []
    
    # Add specific synonyms for airports
    if location.lower() in airport_codes:
        code = airport_codes[location.lower()]
        if code.lower() not in [s.lower() for s in synonyms]:
            synonyms.append(code)
    
    # Generate additional synonyms
    additional_synonyms = generate_additional_synonyms(location, city, layer)
    synonyms.extend(additional_synonyms)
    
    # Remove duplicates in synonyms
    synonyms = list(set(s.lower() for s in synonyms if s))
    
    # Add the item to searchable_items (only once per location)
    item_idx = len(searchable_items)
    item = {
        "display": f"{location} ({city})",
        "value": location,
        "city": city.lower(),
        "layer": layer.lower(),
        "synonyms": synonyms
    }
    searchable_items.append(item)
    
    # Add the location name to search terms
    search_terms.append(location.lower())
    term_to_item_idx.append(item_idx)
    trie[location.lower()] = item_idx  # Add to trie
    
    # Add n-grams for the location name
    for ngram in generate_ngrams(location):
        if ngram not in ngram_dict:
            ngram_dict[ngram] = set()
        ngram_dict[ngram].add(item_idx)
    
    # Add each synonym to search terms
    for synonym in synonyms:
        search_terms.append(synonym.lower())
        term_to_item_idx.append(item_idx)
        trie[synonym.lower()] = item_idx  # Add to trie
        # Add n-grams for synonyms
        for ngram in generate_ngrams(synonym):
            if ngram not in ngram_dict:
                ngram_dict[ngram] = set()
            ngram_dict[ngram].add(item_idx)

# Precompute embeddings and save to file
if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)
else:
    embeddings = model.encode(search_terms, convert_to_tensor=True, batch_size=16)
    with open(embeddings_file, "wb") as f:
        pickle.dump(embeddings, f)

@app.route('/')
def serve_index():
    return send_file(os.path.join(os.path.dirname(__file__), "index.html"))

@app.route('/get_locations', methods=['GET'])
def get_locations():
    locations_data = appsheet_df.to_dict('records')
    return jsonify(locations_data)

@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('query', '').strip().lower()
    if not query:
        return jsonify([])

    # Split query into components (e.g., "Delhi Airport" -> ["delhi", "airport"])
    query_parts = query.split()
    query_ngrams = generate_ngrams(query)

    # Prefix matching with trie
    prefix_matches = set()
    for part in query_parts:
        for key, idx in trie.items(prefix=part):
            prefix_matches.add(idx)

    # N-gram matching
    ngram_matches = set()
    for ngram in query_ngrams:
        if ngram in ngram_dict:
            ngram_matches.update(ngram_dict[ngram])

    # Compute embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarities between the query and all search terms
    similarities = util.cos_sim(query_embedding, embeddings)[0]

    # Combine similarities with searchable items
    # Use a dictionary to aggregate scores for each unique item
    item_scores = {}
    for term_idx, similarity in enumerate(similarities):
        item_idx = term_to_item_idx[term_idx]
        item = searchable_items[item_idx]
        score = similarity.item()
        
        # Boost score if the city matches any query part
        if any(part in item["city"] for part in query_parts):
            score += 0.3
        # Boost score based on query intent (category matching)
        if "airport" in query and "airports" in item["layer"]:
            score += 0.6  # Increased boost for airports
        elif "temple" in query and "religious" in item["layer"]:
            score += 0.4
        # Boost score significantly if the query exactly matches a synonym
        if query in item["synonyms"]:
            score += 0.7
        # Boost score for prefix matches
        if item_idx in prefix_matches:
            score += 0.1
        # Boost score for n-gram matches
        if item_idx in ngram_matches:
            score += 0.15
        # Additional boost for "Indira Gandhi" + "airport" in query for Delhi
        if "indira gandhi" in query and "airport" in query and "delhi" in item["city"]:
            score += 0.5
        
        # Aggregate the highest score for each item
        if item_idx not in item_scores or score > item_scores[item_idx]["score"]:
            item_scores[item_idx] = {
                "display": item["display"],
                "value": item["value"],
                "score": score
            }

    # Add fuzzy matching for location names to handle misspellings
    for item_idx, item in enumerate(searchable_items):
        # Fuzzy match on the location name itself
        fuzzy_score = fuzz.partial_ratio(query, item["value"].lower())
        if fuzzy_score > 85:  # High threshold for near matches
            score = fuzzy_score / 100.0 + 0.5  # Normalize and boost
            # Apply category and city boosts
            if any(part in item["city"] for part in query_parts):
                score += 0.3
            if "airport" in query and "airports" in item["layer"]:
                score += 0.6
            elif "temple" in query and "religious" in item["layer"]:
                score += 0.4
            if "indira gandhi" in query and "airport" in query and "delhi" in item["city"]:
                score += 0.5
            if item_idx in item_scores:
                item_scores[item_idx]["score"] = max(item_scores[item_idx]["score"], score)
            else:
                item_scores[item_idx] = {
                    "display": item["display"],
                    "value": item["value"],
                    "score": score
                }

    # Convert aggregated scores to list and sort
    matches = list(item_scores.values())
    
    # Fallback to fuzzy matching for short queries (e.g., abbreviations like "IGI")
    if len(query) <= 3 and not any(match["score"] > 0.6 for match in matches):
        for item_idx, item in enumerate(searchable_items):
            for synonym in item["synonyms"]:
                fuzzy_score = fuzz.ratio(query, synonym)
                if fuzzy_score > 80:
                    matches.append({
                        "display": item["display"],
                        "value": item["value"],
                        "score": fuzzy_score / 100.0 + 0.5
                    })

    # Filter matches with score > 0.5, sort by score, limit to top 5
    matches = [match for match in matches if match["score"] > 0.5]
    matches.sort(key=lambda x: x["score"], reverse=True)
    matches = matches[:5]

    # Remove duplicates by display value (already handled by item_scores)
    return jsonify(matches)

@app.route('/', methods=['GET'])
def index():
    return "Search API is running!", 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)