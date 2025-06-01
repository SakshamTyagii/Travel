import pandas as pd
import os
import pickle
import sys
import traceback
import logging
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
from pytrie import StringTrie
import requests  # Add this import for the geocoding functions
from dotenv import load_dotenv  # Add this to load your .env file

from flask import Blueprint
import json  # Import json for loading enhanced places data
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Simple in-memory storage for user history and popularity tracking
# For production, use a proper database
user_search_history = {}
popular_searches = {}

# Create a Blueprint instead of a Flask app
app = Blueprint('search', __name__)

# Set up logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Google Maps geocoding functions
def geocode(address):
    """Convert address to coordinates"""
    api_key = os.environ.get('GOOGLE_MAPS_API_KEY', '')
    if not api_key:
        logger.warning("No Google Maps API key found in environment variables")
        return None
        
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                location = data['results'][0]['geometry']['location']
                return {
                    'lat': location['lat'],
                    'lng': location['lng'],
                    'formatted_address': data['results'][0]['formatted_address']
                }
            else:
                logger.warning(f"Geocoding error: {data['status']}")
        else:
            logger.warning(f"Geocoding API returned status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error in geocoding: {str(e)}")
    return None

def reverse_geocode(lat, lng):
    """Convert coordinates to address"""
    api_key = os.environ.get('GOOGLE_MAPS_API_KEY', '')
    if not api_key:
        logger.warning("No Google Maps API key found in environment variables")
        return None
        
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                return data['results'][0]['formatted_address']
            else:
                logger.warning(f"Reverse geocoding error: {data['status']}")
        else:
            logger.warning(f"Reverse geocoding API returned status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error in reverse geocoding: {str(e)}")
    return None

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
    logger.info(f"Running in Cloud Run environment, data_dir={data_dir}")
else:
    # Local development
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    logger.info(f"Running in local environment, data_dir={data_dir}")

appsheet_csv = os.path.join(data_dir, "appsheet.csv")
embeddings_file = os.path.join(data_dir, "embeddings.pkl")
enhanced_places_file = os.path.join(data_dir, "enhanced_places.json")

# Print paths for debugging
logger.info(f"Data directory: {data_dir}")
logger.info(f"Appsheet CSV: {appsheet_csv}")
logger.info(f"Embeddings file: {embeddings_file}")

try:
    # Check if files exist
    if not os.path.exists(appsheet_csv):
        logger.error(f"Appsheet CSV file not found: {appsheet_csv}")
        # List available files in the data directory for debugging
        if os.path.exists(data_dir):
            logger.info(f"Files in data directory: {os.listdir(data_dir)}")
    
    # Load the appsheet.csv file
    appsheet_df = pd.read_csv(appsheet_csv)
    logger.info(f"Successfully loaded appsheet.csv with {len(appsheet_df)} entries")
except Exception as e:
    logger.error(f"Error loading appsheet.csv: {str(e)}")
    logger.error(traceback.format_exc())
    # Create a fallback dataframe with example data
    appsheet_df = pd.DataFrame({
        "City": ["Delhi", "Delhi"],
        "Location": ["Indira Gandhi International Airport", "Red Fort"],
        "Layer": ["Airports", "Monuments"],
        "synonyms": ["DEL, IGI, Delhi Airport", "Lal Qila"]
    })
    logger.warning(f"Created fallback dataframe with {len(appsheet_df)} entries")

try:
    # Initialize the NLP model with all-mpnet-base-v2
    model = SentenceTransformer("all-mpnet-base-v2")
    logger.info("Successfully initialized SentenceTransformer model")
except Exception as e:
    logger.error(f"Error initializing SentenceTransformer: {str(e)}")
    logger.error(traceback.format_exc())
    # Create a fallback model - this is a stub and won't work properly
    class FallbackModel:
        def encode(self, sentences, convert_to_tensor=False, batch_size=16):
            import numpy as np
            # Return dummy embeddings for testing
            if isinstance(sentences, str):
                return torch.tensor(np.zeros(384))
            return torch.tensor(np.zeros((len(sentences), 384)))
    model = FallbackModel()
    logger.warning("Created fallback model")

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
def get_keyboard_neighbors():
    """Return a dictionary of neighboring keys on a QWERTY keyboard"""
    return {
        'a': ['q', 'w', 's', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 's', 'd', 'r'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['g', 'y', 'u', 'j', 'n', 'b'],
        'i': ['u', 'j', 'k', 'o'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p', ';'],
        'm': ['n', 'j', 'k', ','],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'k', 'l', 'p'],
        'p': ['o', 'l', '[', ';'],
        'q': ['1', '2', 'w', 'a'],
        'r': ['e', 'd', 'f', 't'],
        's': ['a', 'w', 'e', 'd', 'x', 'z'],
        't': ['r', 'f', 'g', 'y'],
        'u': ['y', 'h', 'j', 'i'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'a', 's', 'e'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'g', 'h', 'u'],
        'z': ['a', 's', 'x'],
        '0': ['9', '-', '='],
        '1': ['`', 'q', '2'],
        '2': ['1', 'q', 'w', '3'],
        '3': ['2', 'w', 'e', '4'],
        '4': ['3', 'e', 'r', '5'],
        '5': ['4', 'r', 't', '6'],
        '6': ['5', 't', 'y', '7'],
        '7': ['6', 'y', 'u', '8'],
        '8': ['7', 'u', 'i', '9'],
        '9': ['8', 'i', 'o', '0'],
        ' ': [',', '.', '/']  # Space bar neighboring keys
    }

def generate_keyboard_variations(word):
    """Generate variations of a word with possible keyboard typos"""
    if not word or len(word) < 2:  # Too short for variations
        return set()
        
    neighbors = get_keyboard_neighbors()
    variations = set()
    
    # Single character typos
    for i in range(len(word)):
        # Character replacement (pressing a neighboring key)
        char = word[i]
        if char in neighbors:
            for neighbor in neighbors[char]:
                variations.add(word[:i] + neighbor + word[i+1:])
        
        # Character omission (not pressing a key)
        if len(word) > 2:  # Ensure we don't create too-short strings
            variations.add(word[:i] + word[i+1:])
        
        # Character insertion (pressing an extra key)
        if char in neighbors:
            for neighbor in neighbors[char]:
                variations.add(word[:i] + neighbor + word[i:])
        
        # Character transposition (pressing keys in wrong order)
        if i < len(word) - 1:
            variations.add(word[:i] + word[i+1] + word[i] + word[i+2:])
    
    # Double letter errors (pressing a key twice or not pressing twice)
    for i in range(len(word) - 1):
        if word[i] == word[i+1]:  # Double letter
            variations.add(word[:i] + word[i+1:])  # Remove duplicate
        else:
            variations.add(word[:i] + word[i] + word[i] + word[i+1:])  # Add duplicate
    
    return variations

def add_transliteration_variations(word):
    """Add common transliterations for Indian words"""
    if not word or len(word) < 3:  # Too short for meaningful transliterations
        return set()
        
    variations = set()
    
    # Common transliteration patterns
    transliterations = {
        'a': ['aa', 'ah'],
        'aa': ['a', 'ah'],
        'ee': ['i', 'ea'],
        'i': ['ee', 'ea'],
        'u': ['oo'],
        'oo': ['u'],
        'v': ['w'],
        'w': ['v'],
        'sh': ['s'],
        's': ['sh'],
        'k': ['c', 'q'],
        'q': ['k'],
        'c': ['k', 's'],
        'z': ['j'],
        'j': ['z'],
        'ph': ['f'],
        'f': ['ph'],
        'ksh': ['x'],
        'th': ['t'],
        'dh': ['d'],
        'fort': ['qila', 'kila'],
        'temple': ['mandir'],
        'road': ['marg', 'path']
    }
    
    # Generate variations based on transliteration patterns
    for pattern, replacements in transliterations.items():
        if pattern in word:
            for replacement in replacements:
                variations.add(word.replace(pattern, replacement))
    
    return variations

def enhance_search_query(query):
    """Generate keyboard-based variations of the search query"""
    if not query or len(query) < 3:  # Too short for meaningful variations
        return [query]
        
    words = query.split()
    all_variations = [query]  # Start with original query
    
    # For very short queries, be more selective to avoid noise
    if len(query) <= 5:
        max_variations = 3
    else:
        max_variations = 10
    
    # Generate variations for each word
    for word in words:
        if len(word) > 2:  # Only generate variations for words with 3+ chars
            # Get keyboard typo variations
            keyboard_variations = generate_keyboard_variations(word)
            # Get transliteration variations
            trans_variations = add_transliteration_variations(word)
            # Combine variations
            all_word_variations = list(keyboard_variations.union(trans_variations))
            
            # Limit the number of variations to avoid exponential growth
            if len(all_word_variations) > max_variations:
                all_word_variations = all_word_variations[:max_variations]
                
            # Add each variation to the list
            for variation in all_word_variations:
                all_variations.append(query.replace(word, variation))
    
    # Deduplicate
    return list(set(all_variations))

def ngram_similarity(text1, text2, n=3):
    """Calculate n-gram similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    
    if len(text1) < n or len(text2) < n:
        # For very short strings, fall back to direct comparison
        return 1.0 if text1 == text2 else 0.0
    
    ngrams1 = set(generate_ngrams(text1, n))
    ngrams2 = set(generate_ngrams(text2, n))
    
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    
    return len(intersection) / len(union)

import jellyfish

def get_phonetic_code(text):
    """Get phonetic representation of text using Soundex algorithm"""
    if not text:
        return ""
    
    words = text.split()
    if not words:
        return ""
        
    # For multi-word text, get phonetic codes for each word
    if len(words) > 1:
        return " ".join([jellyfish.soundex(word) for word in words])
    
    # For single word
    return jellyfish.soundex(text)

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

try:
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

    logger.info(f"Built search index with {len(searchable_items)} items and {len(search_terms)} search terms")
except Exception as e:
    logger.error(f"Error building search index: {str(e)}")
    logger.error(traceback.format_exc())

# Precompute embeddings and save to file
try:
    if os.path.exists(embeddings_file):
        with open(embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
            logger.info(f"Loaded embeddings from file with shape {embeddings.shape}")
    else:
        logger.info(f"Embeddings file not found, generating new embeddings for {len(search_terms)} terms")
        embeddings = model.encode(search_terms, convert_to_tensor=True, batch_size=16)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        with open(embeddings_file, "wb") as f:
            pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to {embeddings_file}")
except Exception as e:
    logger.error(f"Error processing embeddings: {str(e)}")
    logger.error(traceback.format_exc())
    # Create fallback embeddings - these will not work properly
    import numpy as np
    embeddings = torch.tensor(np.zeros((max(1, len(search_terms)), 384)))
    logger.warning(f"Created fallback embeddings with shape {embeddings.shape}")

# Try to load enhanced places data if available
places_data = []
try:
    if os.path.exists(enhanced_places_file):
        with open(enhanced_places_file, "r", encoding="utf-8") as f:
            places_data = json.load(f)
            logger.info(f"Loaded {len(places_data)} places from enhanced data file")
    else:
        logger.warning(f"Enhanced places file not found: {enhanced_places_file}")
except Exception as e:
    logger.error(f"Error loading enhanced places data: {str(e)}")

# CORS(main_app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.route('/')
def serve_index():
    # Log the attempt to serve the index.html file
    logger.info(f"Serving index.html from {os.path.join(os.path.dirname(__file__), 'index.html')}")
    try:
        return send_file(os.path.join(os.path.dirname(__file__), "index.html"))
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return f"Error serving index.html: {str(e)}", 500

@app.route('/get_locations', methods=['GET'])
def get_locations():
    locations_data = appsheet_df.to_dict('records')
    return jsonify(locations_data)

@app.route('/suggest', methods=['GET'])
@app.route('/api/query', methods=['GET'])
@app.route('/search/suggest', methods=['GET'])
def suggest():
    try:
        query = request.args.get('query', '').strip().lower()
        user_id = request.args.get('user_id')
        user_location = request.args.get('location')
        
        logger.info(f"Received search query: '{query}' via {request.path}")
        
        if not query:
            logger.info("Empty query, returning empty list")
            return jsonify([])

        # Track this search in user history if user_id is provided
        if user_id:
            if user_id not in user_search_history:
                user_search_history[user_id] = []
            
            user_search_history[user_id].append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "location": user_location
            })
        
        # Update search popularity
        update_search_popularity(query)
        
        # Split query into components (e.g., "Delhi Airport" -> ["delhi", "airport"])
        query_parts = query.split()
        logger.info(f"Query parts: {query_parts}")
        query_ngrams = generate_ngrams(query)

        # ENHANCEMENT: Generate keyboard and transliteration variations
        query_variations = enhance_search_query(query)
        logger.info(f"Generated {len(query_variations)} query variations")
        
        # Get phonetic code for query
        query_phonetic = get_phonetic_code(query)
        logger.info(f"Phonetic code for query: {query_phonetic}")

        # Prefix matching with trie
        prefix_matches = set()
        for part in query_parts:
            try:
                for key, idx in trie.items(prefix=part):
                    prefix_matches.add(idx)
            except Exception as e:
                logger.error(f"Error in prefix matching for part '{part}': {str(e)}")
        
        logger.info(f"Found {len(prefix_matches)} prefix matches")

        # N-gram matching
        ngram_matches = set()
        for ngram in query_ngrams:
            if ngram in ngram_dict:
                ngram_matches.update(ngram_dict[ngram])
        
        logger.info(f"Found {len(ngram_matches)} n-gram matches")

        # Compute embedding for the query
        try:
            query_embedding = model.encode(query, convert_to_tensor=True)
            # Compute cosine similarities between the query and all search terms
            similarities = util.cos_sim(query_embedding, embeddings)[0]
            logger.info(f"Computed embeddings and similarities")
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            # Fallback to dummy similarities
            import numpy as np
            similarities = torch.tensor(np.zeros(len(search_terms)))
            logger.warning(f"Using fallback similarities")

        # Combine similarities with searchable items
        # Use a dictionary to aggregate scores for each unique item
        item_scores = {}
        for term_idx, similarity in enumerate(similarities):
            if term_idx >= len(term_to_item_idx):
                continue
            
            item_idx = term_to_item_idx[term_idx]
            if item_idx >= len(searchable_items):
                continue
                
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

        logger.info(f"Computed scores for {len(item_scores)} items")

        # Add fuzzy matching for location names to handle misspellings
        for item_idx, item in enumerate(searchable_items):
            try:
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
                
                # Try variations of the query for fuzzy matching
                for variation in query_variations:
                    if variation != query:  # Skip the original query we already checked
                        var_score = fuzz.partial_ratio(variation, item["value"].lower())
                        if var_score > fuzzy_score:
                            fuzzy_score = var_score
                            logger.info(f"Better match with variation '{variation}': {fuzzy_score}")
                
                # Check for phonetic matches
                item_phonetic = get_phonetic_code(item["value"])
                if query_phonetic == item_phonetic and item_phonetic:
                    # Boost score for phonetic matches
                    phonetic_boost = 0.3
                    score = max(score, fuzzy_score / 100.0 + phonetic_boost)
                    logger.info(f"Phonetic match for '{query}' and '{item['value']}'")
                
                # Also check phonetic matches against synonyms
                for syn in item["synonyms"]:
                    syn_phonetic = get_phonetic_code(syn)
                    if query_phonetic == syn_phonetic and syn_phonetic:
                        phonetic_boost = 0.4  # Higher boost for synonym phonetic matches
                        score = max(score, fuzzy_score / 100.0 + phonetic_boost)
                        logger.info(f"Phonetic match for '{query}' and synonym '{syn}'")
            except Exception as e:
                logger.error(f"Error in fuzzy matching for item {item_idx}: {str(e)}")

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

        # ---- Add this at the end of the function, right before sorting matches ----
        
        # Apply personalization if user_id is provided
        if user_id and user_id in user_search_history and len(user_search_history[user_id]) > 1:
            # Get user's recent searches
            recent_searches = user_search_history[user_id][-10:]
            recent_queries = [s.get("query", "").lower() for s in recent_searches]
            
            # Use query history for personalization
            for idx, match in enumerate(matches):
                # Check if this result relates to previously searched items
                if any(previous_query in match["value"].lower() for previous_query in recent_queries):
                    # Boost based on recency and frequency
                    boost_factor = 0.1
                    match["score"] += boost_factor
                    logger.info(f"Boosted match {match['display']} based on user history")
        
        matches = [match for match in matches if match["score"] > 0.5]
        matches.sort(key=lambda x: x["score"], reverse=True)
        matches = matches[:5]
        
        logger.info(f"Returning {len(matches)} matches")
        for idx, match in enumerate(matches):
            logger.info(f"Match {idx+1}: {match['display']} (score: {match['score']:.2f})")
        
        # Allow CORS for this endpoint
        response = jsonify(matches)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        logger.error(f"Error in suggest endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response
        response = jsonify({"error": str(e)})
        response.status_code = 500
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/record_selection', methods=['POST'])
def record_selection():
    """Record when a user selects an item from search results"""
    try:
        user_id = request.args.get('user_id')
        query = request.args.get('query', '')
        selected_item = request.json if request.is_json else None
        
        logger.info(f"Recording selection for query '{query}' by user {user_id}")
        
        update_search_popularity(query, selected_item)
        if user_id and user_id in user_search_history:
            # Update the last search with selection
            if user_search_history[user_id]:
                user_search_history[user_id][-1]["selected_item"] = selected_item
        
        response = jsonify({"status": "success"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        logger.error(f"Error recording selection: {str(e)}")
        logger.error(traceback.format_exc())
        
        response = jsonify({"status": "error", "message": str(e)})
        response.status_code = 500
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/debug', methods=['GET'])
def debug():
    """Endpoint to provide debug information about the search configuration"""
    try:
        # Get environment information
        env_vars = {key: value for key, value in os.environ.items() 
                  if key.startswith('K_') or 'PORT' in key or 'PATH' in key}
        
        # Get request information
        request_info = {
            "path": request.path,
            "url": request.url,
            "base_url": request.base_url,
            "host_url": request.host_url,
            "host": request.host,
            "endpoint": request.endpoint,
            "headers": dict(request.headers),
        }
        
        # Get available routes
        available_routes = []
        for rule in app.url_map.iter_rules():
            available_routes.append({
                "endpoint": rule.endpoint,
                "methods": [m for m in rule.methods if m not in ['HEAD', 'OPTIONS']],
                "path": str(rule),
            })
            
        # Main debug info
        debug_info = {
            "is_cloud_run": os.environ.get('K_SERVICE') is not None,
            "data_dir": data_dir,
            "appsheet_csv_exists": os.path.exists(appsheet_csv),
            "embeddings_file_exists": os.path.exists(embeddings_file),
            "num_searchable_items": len(searchable_items),
            "num_search_terms": len(search_terms),
            "example_items": searchable_items[:3] if searchable_items else [],
            "environment": env_vars,
            "request_info": request_info,
            "available_routes": available_routes
        }
        
        # Add CORS headers
        response = jsonify(debug_info)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        response = jsonify({"error": str(e)})
        response.status_code = 500
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/healthz', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return "OK", 200

# Add more sophisticated phonetic matching
def advanced_phonetic_matching(query, text):
    """Get more sophisticated phonetic matching using multiple algorithms"""
    # Soundex matching
    soundex_score = 1.0 if jellyfish.soundex(query) == jellyfish.soundex(text) else 0.0
    
    # Metaphone matching (better for Indian place names)
    metaphone_score = 1.0 if jellyfish.metaphone(query) == jellyfish.metaphone(text) else 0.0
    
    # NYSIIS matching
    nysiis_score = 1.0 if jellyfish.nysiis(query) == jellyfish.nysiis(text) else 0.0
    
    # Jaro-Winkler distance (good for short strings)
    jaro_score = jellyfish.jaro_winkler(query, text)
    
    # Return the best score
    return max(soundex_score, metaphone_score, nysiis_score, jaro_score)

# Enhance with intent recognition
def detect_search_intent(query):
    """Detect search intent from query"""
    intents = {
        'transport': ['airport', 'station', 'bus', 'metro', 'train', 'railway'],
        'food': ['restaurant', 'food', 'cafe', 'eat', 'dining'],
        'accommodation': ['hotel', 'stay', 'hostel', 'resort'],
        'attraction': ['visit', 'see', 'attraction', 'monument', 'temple', 'fort'],
        'shopping': ['mall', 'market', 'shop', 'buy'],
    }
    
    query_words = set(query.lower().split())
    
    detected_intents = []
    for intent, keywords in intents.items():
        if any(keyword in query for keyword in keywords):
            detected_intents.append(intent)
    
    return detected_intents

def update_search_popularity(query, selected_item=None):
    """Update popularity of search terms and selected items"""
    query = query.lower()
    # Update query popularity
    if query in popular_searches:
        popular_searches[query] += 1
    else:
        popular_searches[query] = 1
    
    # Update selected item popularity if provided
    if selected_item:
        item_id = selected_item.get("value")  # Using value as ID since your items have this field
        if item_id:
            if "item_popularity" not in popular_searches:
                popular_searches["item_popularity"] = {}
            
            if item_id in popular_searches["item_popularity"]:
                popular_searches["item_popularity"][item_id] += 1
            else:
                popular_searches["item_popularity"][item_id] = 1

if __name__ == "__main__":
    # Create a Flask application
    from flask import Flask
    flask_app = Flask(__name__)
    
    # Register the blueprint
    flask_app.register_blueprint(app)
    
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    flask_app.run(debug=True, host='0.0.0.0', port=port)