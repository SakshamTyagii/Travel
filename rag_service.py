import os
import tempfile
import uvicorn
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our simplified RAG pipeline
from ragpipeline.simple_rag import SimpleRAG

# Get Gemini API key from environment
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    logger.warning("GEMINI_API_KEY not set. Responses will fallback to rule-based extraction.")

# Initialize the RAG service - try JSON first, fall back to CSV
json_path = os.path.join(os.path.dirname(__file__), "ragpipeline", "data", "place_summaries.json")
csv_path = os.path.join(os.path.dirname(__file__), "ragpipeline", "review_summaries.csv")

if os.path.exists(json_path):
    logger.info(f"Loading data from JSON: {json_path}")
    rag_pipeline = SimpleRAG(json_path=json_path, gemini_api_key=gemini_api_key)
else:
    logger.info(f"JSON not found, loading from CSV: {csv_path}")
    rag_pipeline = SimpleRAG(csv_path=csv_path, gemini_api_key=gemini_api_key)

KNOWN_PLACES = rag_pipeline.place_names

# Create FastAPI app
app = FastAPI(
    title="Delhi Travel Assistant - RAG API",
    description="AI-powered Q&A about Delhi attractions",
    version="1.0.0"
)

# Add CORS middleware
# Ensure your CORS settings allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_directory = Path(__file__).parent / "static"
if static_directory.exists():
    app.mount("/", StaticFiles(directory=str(static_directory), html=True), name="static")
else:
    logger.warning(f"Static directory not found at: {static_directory}")

# Define request model
class QueryRequest(BaseModel):
    query: str

# Define response model
class QueryResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {
        "message": "Welcome to Delhi Travel Assistant API",
        "endpoints": {
            "/places": "Get a list of all places in the database",
            "/query": "Post a query to get information about places",
            "/health": "Check if the API is healthy"
        },
        "version": "1.0.0"
    }

@app.get("/places")
def get_places():
    return {"places": KNOWN_PLACES}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        answer, place = rag_pipeline.generate_response(request.query)
        logger.info(f"Query: '{request.query}' → Place: '{place}'")
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ui")
async def serve_ui():
    """Serve the UI HTML file"""
    html_path = Path(__file__).parent / "ragpipeline" / "rag_test.html"
    if html_path.exists():
        return FileResponse(html_path)
    else:
        return {"error": "UI file not found", "path": str(html_path)}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "places_loaded": len(KNOWN_PLACES)
    }

if __name__ == "__main__":
    # Change port to something else that's likely not in use
    port = int(os.environ.get("PORT", 8090))  # Changed from 8080 to 8090
    print(f"Starting server on port {port}")
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")