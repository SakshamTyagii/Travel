import os
import chromadb
from chromadb.utils import embedding_functions
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from google.cloud import storage
import tempfile
import shutil

# Check if running on GCP or locally
is_gcp = os.environ.get('K_SERVICE') is not None  # Cloud Run sets K_SERVICE environment variable

# Configuration for Google Generative AI model
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "your-gemini-api-key")  # Set your API key as env var

# Initialize LLM - Using Google Gemini model instead of Ollama for cloud deployment
llm = GoogleGenerativeAI(model="gemini-pro")

# Handle ChromaDB path based on environment
def get_chroma_path():
    if is_gcp:
        # On GCP, store in the /tmp directory
        chroma_path = "/tmp/chroma_db"
        # Download ChromaDB from Cloud Storage if needed
        if not os.path.exists(chroma_path):
            try:
                download_chroma_from_gcs()
            except Exception as e:
                print(f"Error downloading ChromaDB: {e}")
                # Create empty directory if download fails
                if not os.path.exists(chroma_path):
                    os.makedirs(chroma_path)
        return chroma_path
    else:
        # Local development
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "data", "chroma_db")

# Function to download ChromaDB from Google Cloud Storage
def download_chroma_from_gcs():
    """Download ChromaDB from GCS bucket to temp directory"""
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "riitii-app-chroma-db")
    destination_dir = "/tmp/chroma_db"
    temp_dir = tempfile.mkdtemp()
    
    try:
        # First, try to download the zipped version if available
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        zip_blob = bucket.blob("chroma_db/chroma_db_backup.zip")
        
        if zip_blob.exists():
            print("Found zipped ChromaDB backup, downloading...")
            zip_path = os.path.join(temp_dir, "chroma_db_backup.zip")
            zip_blob.download_to_filename(zip_path)
            
            import zipfile
            # Create the destination directory
            os.makedirs(destination_dir, exist_ok=True)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("/tmp")
            
            print("Successfully extracted ChromaDB from zip backup")
            return
        
        print("No zip backup found. Trying to download individual files...")
        source_blob_name = "chroma_db"
        
        # List all blobs with the prefix
        blobs = bucket.list_blobs(prefix=source_blob_name)
        
        # Download each blob
        downloaded = False
        for blob in blobs:
            # Get relative path
            relative_path = blob.name
            # Create destination path
            destination_path = os.path.join(temp_dir, relative_path)
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            # Download file
            blob.download_to_filename(destination_path)
            downloaded = True
        
        if not downloaded:
            print("No ChromaDB files found in GCS bucket.")
            return
            
        # Move the downloaded files to the destination
        if os.path.exists(destination_dir):
            shutil.rmtree(destination_dir)
        source_dir = os.path.join(temp_dir, source_blob_name)
        if os.path.exists(source_dir):
            shutil.move(source_dir, destination_dir)
            print(f"Successfully downloaded ChromaDB to {destination_dir}")
        else:
            print(f"Error: Expected directory not found at {source_dir}")
    except Exception as e:
        print(f"Error downloading ChromaDB from GCS: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

# Step 3: Load ChromaDB vector database
chroma_path = get_chroma_path()
print(f"Using ChromaDB path: {chroma_path}")

client = chromadb.PersistentClient(path=chroma_path)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

try:
    collection = client.get_collection(
        name="google_maps_summaries",
        embedding_function=embedding_function
    )
    print("Successfully connected to ChromaDB collection")
except Exception as e:
    print(f"Error loading ChromaDB collection: {str(e)}")
    print("Trying to create a new collection...")
    # You may need to populate this collection from a backup
    collection = client.create_collection(
        name="google_maps_summaries",
        embedding_function=embedding_function
    )
    print("Created a new ChromaDB collection")

# Define the RAG Pipeline prompts
intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="Analyze the following user query and describe its intent in one sentence, focusing on what specific information the user is seeking (e.g., best time to visit, suitability for families, activities available). Also, identify the specific place name mentioned in the query, if any:\n\nQuery: {query}\n\nIntent:\nPlace Name:"
)

answer_prompt = PromptTemplate(
    input_variables=["query", "place_name", "summaries"],
    template="The user query is about {place_name}. Based on the following review summary for {place_name}, answer the user's query in 1-2 sentences, focusing on specific details like best visiting times, busy hours, suitability for specific groups, or unique features if mentioned. If the information is not available, provide a general response based on typical patterns (e.g., beaches are often less busy early morning):\n\nQuery: {query}\n\nSummary for {place_name}:\n{summaries}\n\nAnswer:"
)

def extract_place_name(query):
    """Extract the place name from the query for filtering summaries"""
    # More comprehensive list of places - future enhancement would be to pull from database
    known_places = ["Calangute Beach", "Palolem Beach", "Chapora River", "Shree Mangesh Temple",
                    "Colva Beach", "Tropical Spice Plantation", "Goa"]
    query_lower = query.lower()
    for place in known_places:
        if place.lower() in query_lower:
            return place
    return None

def rag_pipeline(query):
    """Implement the RAG pipeline: Understand intent, retrieve summaries, generate answer"""
    # Step 1: Understand query intent and extract place name
    try:
        intent_response = llm.invoke(intent_prompt.format(query=query))
        # Parse intent and place name from the response
        intent_lines = str(intent_response).split("\n")
        intent = intent_lines[0] if intent_lines else "General inquiry about the place."
        place_name = None
        for line in intent_lines:
            if line.startswith("Place Name:"):
                place_name = line.replace("Place Name:", "").strip()
                break
        
        # Fallback: Extract place name directly from query if LLM fails
        if not place_name:
            place_name = extract_place_name(query)
    except Exception as e:
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
                if metadata['place_name'].lower() == place_name.lower():
                    filtered_summaries.append(summary)
                    filtered_metadata.append(metadata)
        else:
            filtered_summaries = retrieved_summaries
            filtered_metadata = retrieved_metadata
        
        # Format the summaries with place names for the LLM
        summaries_text = "\n".join(
            f"{metadata['place_name']}: {summary}"
            for summary, metadata in zip(filtered_summaries, filtered_metadata)
        )
        
        if not summaries_text:
            summaries_text = f"No relevant summary found for {place_name}." if place_name else "No relevant summaries found."
    except Exception as e:
        summaries_text = f"No relevant summary found for {place_name}." if place_name else "No relevant summaries found."

    # Step 3: Generate the final answer using the retrieved summaries
    try:
        # Use the place name in the prompt, or default to a generic response if not specified
        if not place_name:
            place_name = "the place"
        answer = llm.invoke(answer_prompt.format(
            query=query,
            place_name=place_name,
            summaries=summaries_text
        ))
        return str(answer)
    except Exception as e:
        return f"Sorry, I couldn't generate an answer due to an error: {str(e)}"

# For testing purposes
if __name__ == "__main__":
    print("\nWelcome to the Google Maps Review Assistant!")
    print("You can ask questions about the places based on their review summaries.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Your question: ").strip()
        if not query:
            print("Please enter a valid question.")
            continue
        if query.lower() == "exit":
            break
        
        print(f"\nProcessing query: {query}")
        answer = rag_pipeline(query)
        print(f"Answer: {answer}\n")

    print("Thank you for using the Google Maps Review Assistant!")
    
# Export the necessary functions and objects for the web API
__all__ = ['rag_pipeline', 'extract_place_name']
