import os
import json
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

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
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the places data
try:
    with open("search_function/data/places.json", "r", encoding="utf-8") as f:
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

@app.get("/")
async def root():
    return {"message": "Delhi Travel Assistant Search API is running"}

@app.get("/suggest", response_model=List[SearchResult])
async def get_suggestions(query: str = Query(..., min_length=1)):
    try:
        logger.info(f"Processing search query: {query}")
        query = query.lower()
        
        # Filter places by query
        results = []
        for place in places_data:
            name = place.get("name", "").lower()
            category = place.get("category", "").lower()
            area = place.get("area", "").lower()
            
            # Also search in synonyms
            synonyms = [syn.lower() for syn in place.get("synonyms", [])]
            
            if (query in name or 
                query in category or 
                query in area or 
                any(query in syn for syn in synonyms)):
                
                results.append(
                    SearchResult(
                        value=place.get("name", ""),
                        display=place.get("name", ""),
                        category=place.get("category", ""),
                        area=place.get("area", "Delhi"),
                        description=place.get("description", "")
                    )
                )
        
        # Limit results
        results = results[:10]
        logger.info(f"Found {len(results)} results for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error processing search query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")

# Health check endpoint for GCP
@app.get("/_ah/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)