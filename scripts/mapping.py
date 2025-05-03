import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
import os

# Function to generate additional synonyms (e.g., common misspellings)
def generate_synonyms(location):
    synonyms = []
    # 1. Remove spaces (e.g., "San Francisco" -> "SanFrancisco")
    synonyms.append(location.replace(" ", ""))
    # 2. Common abbreviation (e.g., "San Francisco" -> "SF")
    if len(location.split()) > 1:
        initials = "".join(word[0] for word in location.split())
        synonyms.append(initials)
    # 3. Replace vowels (e.g., "San" -> "Sen")
    vowel_swaps = {"a": "e", "e": "a", "i": "y", "o": "u", "u": "o"}
    swapped = location
    for v1, v2 in vowel_swaps.items():
        swapped = re.sub(f"(?i){v1}", v2, swapped)
    if swapped != location:
        synonyms.append(swapped)
    # Remove duplicates and original location
    synonyms = [s for s in set(synonyms) if s.lower() != location.lower()]
    return synonyms

# Update file paths to use correct directory structure and OS-agnostic paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
appsheet_csv = os.path.join(data_dir, "appsheet.csv")
mongodb_csv = os.path.join(data_dir, "mongodb.csv")
mappings_csv = os.path.join(os.path.dirname(__file__), "location_mappings.csv")

# Load CSVs from the data folder
appsheet_df = pd.read_csv(appsheet_csv)
mongodb_df = pd.read_csv(mongodb_csv)

# Initialize NLP model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize mappings and synonym dictionary
mappings = {}  # MongoDB location -> AppSheet location
synonym_dict = {loc: [] for loc in appsheet_df["Location"].unique()}  # AppSheet location -> list of synonyms

# Map locations by city
for city in appsheet_df["City"].unique():
    # Extract locations for the current city
    appsheet_locs = appsheet_df[appsheet_df["City"] == city]["Location"].tolist()
    mongodb_locs = mongodb_df[mongodb_df["City"] == city]["Location"].tolist()
    
    if not appsheet_locs or not mongodb_locs:
        continue  # Skip if no locations in either dataset for this city
    
    # Compute embeddings
    app_embeddings = model.encode(appsheet_locs, convert_to_tensor=True)
    mongo_embeddings = model.encode(mongodb_locs, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.cos_sim(mongo_embeddings, app_embeddings)
    
    # Map each MongoDB location to the best AppSheet match
    for i, mongo_loc in enumerate(mongodb_locs):
        max_sim = similarities[i].max().item()
        if max_sim > 0.8:  # Cosine similarity threshold
            best_idx = similarities[i].argmax().item()
            best_match = appsheet_locs[best_idx]
            mappings[mongo_loc] = best_match
            # Add MongoDB location as a synonym if it's not the correct name
            if mongo_loc.lower() != best_match.lower():
                synonym_dict[best_match].append(mongo_loc)

# Add generated synonyms to synonym_dict
for loc in synonym_dict:
    synonym_dict[loc].extend(generate_synonyms(loc))
    # Remove duplicates and ensure unique synonyms
    synonym_dict[loc] = list(set(synonym_dict[loc]))

# Add synonyms column to AppSheet DataFrame
appsheet_df["synonyms"] = appsheet_df["Location"].map(lambda x: ", ".join(synonym_dict.get(x, [])))

# Overwrite the original appsheet.csv with the updated DataFrame
appsheet_df.to_csv(appsheet_csv, index=False)

# Save mappings for reference in the current directory
pd.DataFrame(list(mappings.items()), columns=["MongoDB_Location", "AppSheet_Location"]).to_csv(mappings_csv, index=False)
