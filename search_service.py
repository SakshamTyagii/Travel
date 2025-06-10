import os
import json
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Travel Assistant - Search API",
    description="Search for attractions, hotels, and more",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://travel-search-suggestion.web.app",
        "https://rag-frontend-e4375.web.app",
        "*"  # Remove this in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the places data
try:
    with open("search_function/data/enhanced_places.json", "r", encoding="utf-8") as f:
        places_data = json.load(f)
    logger.info(f"Loaded {len(places_data)} places from data file")
except Exception as e:
    logger.error(f"Error loading places data: {str(e)}")
    places_data = []

# Define models
class SearchResult(BaseModel):
    value: str
    display: str
    category: Optional[str] = None
    area: Optional[str] = None
    description: Optional[str] = None
    match_score: Optional[float] = 0.0  # Added to track match quality
    match_info: Optional[dict] = None   # Added to store match details for highlighting

@app.get("/")
async def root():
    return {"message": "Delhi Travel Assistant Search API is running"}

# Helper functions for better searching
def tokenize_query(query):
    """Break query into meaningful tokens"""
    return query.lower().split()

def get_ngrams(text, n=2):
    """Generate n-grams from text"""
    tokens = text.lower().split()
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def word_match_score(query_words, text):
    """Calculate how many words from the query match in the text"""
    if not text:
        return 0
        
    text_lower = text.lower()
    matches = 0
    
    for word in query_words:
        if word.lower() in text_lower:
            matches += 1
            
    return matches / len(query_words) if query_words else 0

# Add to your search service

# Add this import if not already present
import time
import os

# Add a function to reload data when needed
def reload_places_data():
    global places_data
    try:
        data_path = "search_function/data/enhanced_places.json"
        mod_time = os.path.getmtime(data_path)
        
        # Only reload if the file was modified recently
        if time.time() - mod_time < 600:  # 10 minutes
            with open(data_path, "r", encoding="utf-8") as f:
                places_data = json.load(f)
            logger.info(f"Reloaded {len(places_data)} places from data file")
    except Exception as e:
        logger.error(f"Error reloading places data: {str(e)}")

@app.get("/suggest", response_model=List[SearchResult])
async def get_suggestions(query: str = Query(..., min_length=3)):  # Enforce minimum length of 3
    # Check for data updates periodically
    reload_places_data()
    
    try:
        logger.info(f"Processing search query: '{query}'")
        query = query.lower().strip()
        
        # Double-check query length (defensive programming)
        if len(query) < 3:
            logger.info(f"Query too short, length: {len(query)}")
            return []
        
        # Tokenize the query
        query_words = tokenize_query(query)
        query_bigrams = get_ngrams(query, 2) if len(query_words) >= 2 else []
        
        # Filter places by query
        results = []
        seen_names = set()  # To avoid duplicates
        
        # Check for the city aliases in the query and add additional search terms
        city_aliases = {
            "chennai": "madras", 
            "madras": "chennai",
            "mumbai": "bombay", 
            "bombay": "mumbai",
            "kolkata": "calcutta",
            "calcutta": "kolkata",
            "bengaluru": "bangalore",
            "bangalore": "bengaluru",
            "kochi": "cochin",
            "cochin": "kochi",
            "thiruvananthapuram": "trivandrum",
            "trivandrum": "thiruvananthapuram",
            # "varanasi": ["banaras", "benares", "kashi"],
            "banaras": "varanasi",
            "benares": "varanasi",
            "kashi": "varanasi",
            "pune": "poona",
            "poona": "pune",
            "shimla": "simla",
            "simla": "shimla",
            
            # Add more as needed
        }
        
        additional_queries = [query]
        
        # Add queries with city name substitutions
        for city, alias in city_aliases.items():
            if city in query:
                additional_queries.append(query.replace(city, alias))
            if alias in query:
                additional_queries.append(query.replace(alias, city))
          # Search with all query variants
        for current_query in additional_queries:
            # Tokenize the query to match individual words as well
            query_words = current_query.lower().split()
            
            for place in places_data:
                name = place.get("name", "").lower()
                area = place.get("area", "").lower() if place.get("area") else ""
                synonyms = [s.lower() for s in place.get("synonyms", [])]
                
                # Only process each unique place once
                place_key = f"{name}_{area}"
                if place_key in seen_names:
                    continue
                  # Enhanced matching logic with priority for starting matches
                should_include = False
                
                # 1. Check if name or any word in name starts with query
                if (name.startswith(current_query) or 
                    any(word.startswith(current_query) for word in name.split())):
                    should_include = True
                
                # 2. Check if any synonym starts with query
                elif any(syn.startswith(current_query) or 
                        any(word.startswith(current_query) for word in syn.split()) 
                        for syn in synonyms):
                    should_include = True
                
                # 3. Fall back to contains matches
                elif (current_query in name or 
                      (area and current_query in area) or 
                      any(current_query in syn for syn in synonyms)):
                    should_include = True
                
                # 4. Multi-word match: all words from query should be present
                elif len(query_words) > 1:
                    all_content = f"{name} {area} {' '.join(synonyms)}"
                    if all(word in all_content for word in query_words):
                        should_include = True
                
                # 5. Partial word matches for single longer words
                elif len(query_words) == 1 and len(query_words[0]) >= 3:
                    word = query_words[0]
                    if (word in name or 
                        (area and word in area) or 
                        any(word in syn for syn in synonyms)):                        should_include = True
                
                if should_include:
                    # Calculate relevance score with new prioritization
                    score = calculate_relevance_score(current_query, name, area, synonyms)
                    
                    # Additional boost for multi-word matches
                    if len(query_words) > 1:
                        all_content = f"{name} {area} {' '.join(synonyms)}"
                        if all(word in all_content for word in query_words):
                            score += 2.0
                    
                    results.append(
                        SearchResult(
                            value=place.get("name", ""),
                            display=place.get("name", ""),
                            category=place.get("category", ""),
                            area=place.get("area", "Delhi"),
                            description=place.get("description", ""),
                            match_score=score,
                            match_info={"query": current_query}
                        )
                    )
                    
                    seen_names.add(place_key)
        
        # Sort results by match score (descending)
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        # Limit results but ensure we show at least 10 if available
        max_results = min(15, max(10, len(results)))
        results = results[:max_results]
        
        logger.info(f"Found {len(results)} results for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error processing search query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")

def calculate_relevance_score(query, name, area, synonyms):
    """Calculate a relevance score for search results ranking"""
    score = 0.0
    match_info = {}
    
    query_words = query.lower().split()
    query_lower = query.lower()
    name_lower = name.lower()
    area_lower = area.lower()
    
    # 1. HIGHEST PRIORITY: Words that START with the query
    if name_lower.startswith(query_lower):
        score += 10.0  # Highest score for names starting with query
        match_info["name_starts_with"] = True
    elif any(word.startswith(query_lower) for word in name_lower.split()):
        score += 8.0  # High score if any word in name starts with query
        match_info["name_word_starts_with"] = True
    
    # Check synonyms for starting matches
    for syn in synonyms:
        syn_lower = syn.lower()
        if syn_lower.startswith(query_lower):
            score += 7.0
            match_info["synonym_starts_with"] = True
            break
        elif any(word.startswith(query_lower) for word in syn_lower.split()):
            score += 6.0
            match_info["synonym_word_starts_with"] = True
            break
    
    # 2. MEDIUM PRIORITY: Exact substring matches (original logic)
    if query_lower in name_lower and not name_lower.startswith(query_lower):
        score += 5.0
        match_info["name_contains"] = True
    
    # Area match gets medium score
    if query_lower in area_lower:
        score += 3.0
        match_info["area_match"] = True
    
    # Synonym contains matches (lower priority than starts with)
    synonym_matches = [syn for syn in synonyms if query_lower in syn.lower() and not syn.lower().startswith(query_lower)]
    if synonym_matches:
        score += 4.0
        match_info["synonym_contains"] = synonym_matches[:3]  # Limit to 3 for brevity
    
    # Multi-word scoring: check if individual words appear
    if len(query_words) > 1:
        name_word_score = word_match_score(query_words, name)
        area_word_score = word_match_score(query_words, area)
        
        # Score based on how many query words match
        score += name_word_score * 3.0  # Name matches are more important
        score += area_word_score * 2.0  # Area matches are secondary
        
        # Check synonyms for multi-word matches
        best_synonym_score = 0
        for syn in synonyms:
            syn_word_score = word_match_score(query_words, syn)
            best_synonym_score = max(best_synonym_score, syn_word_score)
        
        score += best_synonym_score * 2.5
        
        # Bonus for exact word order in name
        if all(word in name for word in query_words):
            # Check if words appear in same order
            name_words = name.split()
            query_positions = []
            for qword in query_words:
                for i, nword in enumerate(name_words):
                    if qword in nword:
                        query_positions.append(i)
                        break
            
            if len(query_positions) == len(query_words) and query_positions == sorted(query_positions):
                score += 1.0  # Bonus for maintaining word order
      # Single word partial matching
    elif len(query_words) == 1:
        word = query_words[0]
        if len(word) >= 3:  # Reduced from 4 to 3 for better matching
            # Check if any word in the name starts with the query word
            name_words = name_lower.split()
            if any(nword.startswith(word) for nword in name_words):
                score += 4.0  # Higher score for word starting matches
            elif word in name_lower:
                score += 2.0  # Lower score for contains matches
            
            # Check synonyms
            for syn in synonyms:
                syn_words = syn.lower().split()
                if any(sword.startswith(word) for sword in syn_words):
                    score += 3.0
                    break
                elif word in syn.lower():
                    score += 1.5
                    break
    
    return score

# Health check endpoint for GCP
@app.get("/_ah/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)