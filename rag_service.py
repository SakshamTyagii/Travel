import os
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

# Force use of local model by setting environment variable
os.environ["USE_LOCAL_MODEL"] = "true"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local RAG pipeline that uses Llama2
try:
    # Use the local version explicitly, not the GCP version
    from ragpipeline.rag_pipeline import rag_pipeline
    logger.info("Using local Llama2-based RAG pipeline")
except ImportError as e:
    logger.error(f"Failed to import local RAG pipeline: {str(e)}")
    # Define a fallback function
    def rag_pipeline(query):
        return "RAG pipeline not available. Check logs for import errors."

# Create FastAPI app
app = FastAPI(
    title="Delhi Travel Assistant - RAG API",
    description="AI-powered Q&A about Delhi attractions",
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

# Define request model
class QueryRequest(BaseModel):
    query: str

# Define response model
class QueryResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Delhi Travel Assistant RAG API is running"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query}")
        answer = rag_pipeline(request.query)
        logger.info("Query processed successfully")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check endpoint for GCP
@app.get("/_ah/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)