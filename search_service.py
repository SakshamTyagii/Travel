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
            for place in places_data:
                name = place.get("name", "").lower()
                area = place.get("area", "").lower() if place.get("area") else ""
                synonyms = [s.lower() for s in place.get("synonyms", [])]
                
                # Only process each unique place once
                place_key = f"{name}_{area}"
                if place_key in seen_names:
                    continue
                
                # Check if query matches name, area, or any synonym
                if (current_query in name or 
                    (area and current_query in area) or 
                    any(current_query in syn for syn in synonyms)):
                    
                    # Calculate relevance score
                    score = calculate_relevance_score(current_query, name, area, synonyms)
                    
                    results.append(
                        SearchResult(
                            value=place.get("name", ""),
                            display=place.get("name", ""),
                            category=place.get("category", ""),
                            area=place.get("area", "Delhi"),
                            description=place.get("description", ""),
                            match_score=score,
                            match_info={"match_type": "query_match"}
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
    
    # Direct name match gets highest score
    if query in name:
        score += 5.0
        match_info["name_match"] = True
    
    # Area match gets medium score
    if query in area:
        score += 3.0
        match_info["area_match"] = True
    
    # Synonym matches
    synonym_matches = [syn for syn in synonyms if query in syn]
    if synonym_matches:
        score += 4.0
        match_info["synonym_matches"] = synonym_matches[:3]  # Limit to 3 for brevity
    
    # Word match boost
    query_words = query.split()
    if len(query_words) > 1:
        name_word_score = word_match_score(query_words, name)
        score += name_word_score * 2.0
        
        # Check synonyms for multi-word matches
        for syn in synonyms:
            syn_word_score = word_match_score(query_words, syn)
            if syn_word_score > 0.5:  # Only boost if more than half the words match
                score += 1.0
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