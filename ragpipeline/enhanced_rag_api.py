import os
import json
import logging
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Travel Assistant RAG API", version="2.0.0")

# ADD CORS MIDDLEWARE - THIS FIXES THE ERROR
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-frontend-e4375.web.app",
        "https://rag-frontend-e4375.firebaseapp.com",
        "http://localhost:5000",
        "http://localhost:3000",
        "http://127.0.0.1:5500",  # Your current frontend
        "http://127.0.0.1:5000",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class LocationRequest(BaseModel):
    location: str
    question: str = None

class EnhancedRAG:
    def __init__(self, json_path: str, gemini_api_key: str):
        self.json_path = json_path
        self.gemini_api_key = gemini_api_key
        
        # Configure Gemini API
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Load enhanced summaries
        self.summaries = self._load_summaries()
        logger.info(f"Loaded {len(self.summaries)} enhanced location summaries")
    
    def _load_summaries(self) -> Dict[str, Any]:
        """Load enhanced summaries from JSON file"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading summaries: {e}")
            return {}
    
    def find_relevant_location(self, query: str) -> str:
        """Find the most relevant location based on query"""
        query_lower = query.lower()
        
        # Direct match
        for location in self.summaries.keys():
            if location.lower() in query_lower:
                return location
        
        # Partial match
        for location in self.summaries.keys():
            location_words = location.lower().split()
            if any(word in query_lower for word in location_words if len(word) > 3):
                return location
        
        return None
    
    def get_enhanced_summary(self, location: str) -> Dict[str, Any]:
        """Get enhanced summary for a location"""
        if location in self.summaries:
            return self.summaries[location]
        
        # Try fuzzy match
        for loc_name in self.summaries.keys():
            if location.lower() in loc_name.lower() or loc_name.lower() in location.lower():
                return self.summaries[loc_name]
        
        return None
    
    def query(self, user_query: str) -> str:
        """Process user query and return enhanced response"""
        try:
            # Find relevant location
            location = self.find_relevant_location(user_query)
            
            if not location:
                # General query without specific location
                return self._handle_general_query(user_query)
            
            # Get location data
            location_data = self.get_enhanced_summary(location)
            if not location_data:
                return f"Sorry, I don't have information about {location}."
            
            # Create context-aware response
            context = f"""
Location: {location}
Enhanced Summary: {location_data.get('summary', '')}
Total Reviews Analyzed: {location_data.get('total_reviews', 0)}
Enhanced Data Available: {bool(location_data.get('enhanced_data'))}
"""
            
            prompt = f"""
Based on the following detailed information about {location}, answer the user's question comprehensively.

Context:
{context}

User Question: {user_query}

Provide a helpful, detailed response that:
1. Directly answers the question
2. Uses the enhanced summary data
3. Mentions specific details when available
4. Is conversational and helpful
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I'm sorry, I encountered an error processing your request. Please try again."
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general queries without specific location"""
        locations_list = "\n".join([f"- {loc}" for loc in self.summaries.keys()])
        
        return f"""I can help you with information about these locations in Delhi and Goa:

{locations_list}

Please specify which location you'd like to know about, or ask a question like:
- "Tell me about Aguada Fort"
- "What's the best time to visit Calangute Beach?"
- "Is Red Fort suitable for families?"
"""

# Initialize the RAG system
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found")
    
    json_path = os.path.join(os.path.dirname(__file__), "data", "enhanced_place_summaries.json")
    rag_system = EnhancedRAG(json_path=json_path, gemini_api_key=gemini_api_key)
    
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_system = None

@app.get("/")
async def root():
    return {
        "message": "Enhanced Travel Assistant RAG API",
        "version": "2.0.0",
        "status": "active" if rag_system else "error",
        "locations_available": len(rag_system.summaries) if rag_system else 0
    }

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Enhanced query endpoint with improved RAG"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        response = rag_system.query(request.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")

@app.post("/location-summary")
async def location_summary(request: LocationRequest):
    """Get enhanced summary for specific location"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        summary = rag_system.get_enhanced_summary(request.location)
        if summary:
            return {"location": request.location, "summary": summary}
        else:
            return {"error": f"Location '{request.location}' not found"}
    except Exception as e:
        logger.error(f"Location summary error: {e}")
        raise HTTPException(status_code=500, detail="Error getting location summary")

@app.get("/locations")
async def list_locations():
    """List all available locations"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    return {
        "locations": list(rag_system.summaries.keys()),
        "total": len(rag_system.summaries)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "locations_count": len(rag_system.summaries) if rag_system else 0
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
