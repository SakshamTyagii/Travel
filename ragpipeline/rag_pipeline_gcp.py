import os
import tempfile
from chromadb.utils import embedding_functions
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google.cloud import storage
import chromadb
import json

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "riitii-app-chroma-db")

# Initialize temp directory for ChromaDB
CHROMA_DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db")
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

def download_chromadb_from_gcs():
    """Download ChromaDB files from GCS bucket"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        # List all the files in the chroma_db folder
        blobs = list(bucket.list_blobs(prefix="chroma_db/"))
        print(f"Found {len(blobs)} files to download from GCS")
        
        # Download each file to the local temp directory
        for blob in blobs:
            if blob.name.endswith('/'):  # Skip directories
                continue
                
            # Create the local directory structure if needed
            local_path = os.path.join(CHROMA_DB_PATH, blob.name.replace("chroma_db/", ""))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")
        
        return True
    except Exception as e:
        print(f"Error downloading ChromaDB from GCS: {str(e)}")
        return False

# Try to download ChromaDB from GCS
download_success = download_chromadb_from_gcs()
if not download_success:
    print("Failed to download ChromaDB from GCS. Check GCS_BUCKET_NAME environment variable.")

# Initialize Gemini LLM
llm = GoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=GEMINI_API_KEY, temperature=0.3)

# Initialize ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
try:
    collection = client.get_collection(
        name="google_maps_summaries",
        embedding_function=embedding_function
    )
    print("Successfully loaded ChromaDB collection")
except Exception as e:
    print(f"Error loading ChromaDB collection: {str(e)}")
    # Try to create a new collection if it doesn't exist
    try:
        collection = client.create_collection(
            name="google_maps_summaries",
            embedding_function=embedding_function
        )
        print("Created new ChromaDB collection")
    except Exception as e:
        print(f"Error creating ChromaDB collection: {str(e)}")
        collection = None

# Define prompts
intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="Analyze the following user query and describe its intent in one sentence, focusing on what specific information the user is seeking (e.g., best time to visit, suitability for families, activities available). Also, identify the specific place name mentioned in the query, if any:\n\nQuery: {query}\n\nIntent:\nPlace Name:"
)

answer_prompt = PromptTemplate(
    input_variables=["query", "place_name", "summaries"],
    template="The user query is about {place_name}. Based on the following review summary for {place_name}, answer the user's query in 1-2 sentences, focusing on specific details like best visiting times, busy hours, suitability for specific groups, or unique features if mentioned. If the information is not available, provide a general response based on typical patterns for similar places in Delhi:\n\nQuery: {query}\n\nSummary for {place_name}:\n{summaries}\n\nAnswer:"
)

def extract_place_name(query):
    """Extract the place name from the query for filtering summaries"""
    known_places = [
        "India Gate", "Red Fort", "Qutub Minar", "Akshardham Temple", 
        "Lotus Temple", "Humayun's Tomb", "Jama Masjid", "Chandni Chowk",
        "Connaught Place", "Lodhi Gardens", "Delhi Zoo"
    ]
    query_lower = query.lower()
    for place in known_places:
        if place.lower() in query_lower:
            return place
    return None

def rag_pipeline(query):
    """Implement the RAG pipeline: Understand intent, retrieve summaries, generate answer"""
    if not collection:
        return "Sorry, I couldn't connect to the database. Please try again later."
    
    # Step 1: Understand query intent and extract place name
    try:
        intent_response = llm.invoke(intent_prompt.format(query=query))
        intent_lines = str(intent_response).split("\n")
        intent = intent_lines[0] if intent_lines else "General inquiry about the place."
        place_name = None
        for line in intent_lines:
            if line.startswith("Place Name:"):
                place_name = line.replace("Place Name:", "").strip()
                break
        if not place_name:
            place_name = extract_place_name(query)
    except Exception as e:
        print(f"Error extracting intent: {str(e)}")
        intent = "General inquiry about the place."
        place_name = extract_place_name(query)

    # Step 2: Retrieve relevant summaries from ChromaDB
    try:
        query_result = collection.query(query_texts=[query], n_results=3)
        retrieved_summaries = query_result['documents'][0]
        retrieved_metadata = query_result['metadatas'][0]
        
        # Filter summaries to only include the specific place, if a place name was identified
        if place_name:
            filtered_summaries = []
            filtered_metadata = []
            for summary, metadata in zip(retrieved_summaries, retrieved_metadata):
                if 'place_name' in metadata and metadata['place_name'].lower() == place_name.lower():
                    filtered_summaries.append(summary)
                    filtered_metadata.append(metadata)
            
            if not filtered_summaries:  # If no exact match, use all results
                filtered_summaries = retrieved_summaries
                filtered_metadata = retrieved_metadata
        else:
            filtered_summaries = retrieved_summaries
            filtered_metadata = retrieved_metadata
        
        # Format the summaries with place names for the LLM
        summaries_text = "\n".join(
            f"{metadata.get('place_name', 'Unknown place')}: {summary}"
            for summary, metadata in zip(filtered_summaries, filtered_metadata)
        )
        
        if not summaries_text:
            summaries_text = f"No relevant summary found for {place_name}." if place_name else "No relevant summaries found."
    except Exception as e:
        print(f"Error retrieving summaries: {str(e)}")
        summaries_text = f"No relevant summary found for {place_name}." if place_name else "No relevant summaries found."

    # Step 3: Generate the final answer using the retrieved summaries
    try:
        if not place_name:
            place_name = "the place"
        answer = llm.invoke(answer_prompt.format(
            query=query,
            place_name=place_name,
            summaries=summaries_text
        ))
        return str(answer)
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return f"I found information about {place_name}, but couldn't generate a complete answer. Please try asking in a different way."