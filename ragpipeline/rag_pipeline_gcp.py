import os
import tempfile
from chromadb.utils import embedding_functions
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google.cloud import storage
import chromadb
import json
import google.generativeai as genai
import threading

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "riitii-app-chroma-db")

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY environment variable not set")
    api_key = "DEFAULT_KEY_FOR_DEVELOPMENT_ONLY"  # This is just a fallback for development

# Add more detailed logging
print(f"Configuring Gemini API (API key {'found' if api_key else 'NOT FOUND'})")
genai.configure(api_key=api_key)

# Initialize temp directory for ChromaDB
CHROMA_DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db")
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Complete the download_chromadb_from_gcs function:
def download_chromadb_from_gcs():
    """Download ChromaDB files from GCS bucket"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        # List all the files in the chroma_db folder
        blobs = list(bucket.list_blobs(prefix="chroma_db/"))
        print(f"Found {len(blobs)} files to download from GCS")
        
        if len(blobs) == 0:
            print(f"WARNING: No files found in GCS bucket {GCS_BUCKET_NAME}/chroma_db/")
            return False
            
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
        
        print(f"Successfully downloaded ChromaDB from GCS bucket {GCS_BUCKET_NAME}")
        return True
    except Exception as e:
        print(f"Error downloading ChromaDB from GCS: {str(e)}")
        return False

def initialize_chroma_db():
    """Initialize ChromaDB from CSV if it doesn't exist"""
    import pandas as pd
    
    csv_path = "ragpipeline/review_summaries.csv"
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Check if collection exists
    try:
        collection = client.get_collection(
            name="google_maps_summaries",
            embedding_function=embedding_function
        )
        print("ChromaDB collection already exists")
        return collection
    except Exception:
        print("ChromaDB collection does not exist, creating from CSV...")
        
    # Create collection
    collection = client.create_collection(
        name="google_maps_summaries",
        embedding_function=embedding_function
    )
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} reviews from CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # Add documents to collection
    ids = [str(i) for i in range(len(df))]
    documents = df['summary'].tolist()
    metadatas = [{"place_name": place_name} for place_name in df['place_name'].tolist()]
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Added {len(documents)} documents to ChromaDB")
    return collection

# Complete the load_place_names_from_csv function:
def load_place_names_from_csv():
    """Automatically extract place names from the CSV file"""
    try:
        import pandas as pd
        csv_path = "ragpipeline/review_summaries.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            place_names = df['place_name'].unique().tolist()
            print(f"Loaded {len(place_names)} place names from CSV")
            return place_names
        else:
            print(f"CSV file not found at {csv_path}")
            return []
    except Exception as e:
        print(f"Error loading place names from CSV: {str(e)}")
        return []

# Add this line to initialize the known places at the top-level of your script
KNOWN_PLACES = load_place_names_from_csv()

# Add after loading the place names
print(f"Loaded place names: {', '.join(KNOWN_PLACES[:5])}{'...' if len(KNOWN_PLACES) > 5 else ''}")

# Try to download ChromaDB from GCS first
download_success = download_chromadb_from_gcs()
if not download_success:
    print("Failed to download ChromaDB from GCS. Trying to initialize from CSV...")

# Initialize ChromaDB - now with multiple fallbacks
try:
    # Try to initialize from CSV if needed
    collection = initialize_chroma_db()
    if collection:
        print("Successfully initialized ChromaDB collection")
    else:
        print("Failed to initialize ChromaDB collection from CSV")
        collection = None
except Exception as e:
    print(f"Error initializing ChromaDB: {str(e)}")
    collection = None

# Validate collection content
if collection is not None:
    try:
        # Check if collection has documents
        check_result = collection.query(query_texts=["beach"], n_results=1)
        if check_result and len(check_result['documents']) > 0 and len(check_result['documents'][0]) > 0:
            print(f"ChromaDB collection contains documents. Example: {check_result['documents'][0][0][:50]}...")
        else:
            print("WARNING: ChromaDB collection appears to be empty")
    except Exception as e:
        print(f"Error checking collection content: {e}")

# Right after initializing the collection, add this check
if collection:
    try:
        count = collection.count()
        print(f"ChromaDB collection contains {count} documents")
        
        # List the first few places in the collection
        sample_query = collection.query(query_texts=["beach"], n_results=5)
        if sample_query and sample_query['metadatas'][0]:
            places = [metadata['place_name'] for metadata in sample_query['metadatas'][0]]
            print(f"Sample place names in ChromaDB: {', '.join(places)}")
        
        if count == 0:
            print("WARNING: ChromaDB collection is empty! Running initialization...")
            collection = initialize_chroma_db()
    except Exception as e:
        print(f"Error checking ChromaDB content: {str(e)}")

# Update the find_working_gemini_model function:
def find_working_gemini_model():
    """Find a working Gemini model by trying different options"""
    models_to_try = ["gemini-2.0-flash"]  # Only try the 2.0 flash model
    
    for model_name in models_to_try:
        try:
            print(f"Trying to initialize Gemini model: {model_name}")
            llm = GoogleGenerativeAI(model=model_name, google_api_key=os.environ.get("GEMINI_API_KEY"))
            # Test with a simple prompt
            _ = llm.invoke("Hello")
            print(f"Successfully initialized {model_name}")
            return llm
        except Exception as e:
            print(f"Error initializing {model_name}: {str(e)}")
    
    print("ERROR: Could not initialize Gemini model")
    return None

def initialize_llm_in_background():
    """Initialize the LLM in a background thread to avoid blocking startup"""
    global llm, model_initialization_complete
    
    try:
        # Make sure the API key is set
        if not os.environ.get("GEMINI_API_KEY"):
            print("ERROR: GEMINI_API_KEY environment variable not set")
            print("Available environment variables:", list(os.environ.keys()))
            
        # Try to initialize the LLM
        llm = find_working_gemini_model()
        
        if llm is None:
            print("ERROR: Failed to initialize any Gemini model")
            return
        
        print("LLM initialization successful")
        model_initialization_complete = True
    except Exception as e:
        print(f"Error during LLM initialization: {str(e)}")
        traceback.print_exc()

# Start initialization in background thread
threading.Thread(target=initialize_llm_in_background, daemon=True).start()

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

# Define prompts - KEEPING THE EXACT SAME PROMPTS AS IN rag_pipeline.py
intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="Analyze the following user query and describe its intent in one sentence, focusing on what specific information the user is seeking (e.g., best time to visit, suitability for families, activities available). Also, identify the specific place name mentioned in the query, if any:\n\nQuery: {query}\n\nIntent:\nPlace Name:"
)

answer_prompt = PromptTemplate(
    input_variables=["query", "place_name", "summaries"],
    template="The user query is about {place_name}. Based on the following review summary for {place_name}, answer the user's query in 1-2 sentences, focusing on specific details like best visiting times, busy hours, suitability for specific groups, or unique features if mentioned. If the information is not available, provide a general response based on typical patterns (e.g., beaches are often less busy early morning):\n\nQuery: {query}\n\nSummary for {place_name}:\n{summaries}\n\nAnswer:"
)

# Then replace your extract_place_name function with this:
def extract_place_name(query):
    """Extract the place name from the query for filtering summaries"""
    query_lower = query.lower()
    
    # Try exact matches first
    for place in KNOWN_PLACES:
        if place.lower() in query_lower:
            return place
    
    # If no exact match, try partial matches
    for place in KNOWN_PLACES:
        # Split place name into words and check for partial matches
        words = place.lower().split()
        if any(word in query_lower for word in words if len(word) > 3):  # Only match words longer than 3 chars
            return place
            
    return None

# Update the rag_pipeline function to have more robust error handling:
def rag_pipeline(query):
    """Implement the RAG pipeline with initialization check"""
    print(f"RAG Pipeline: Processing query: {query}")
    
    if not model_initialization_complete:
        print("RAG Pipeline: Model not initialized yet")
        return "The model is still initializing. Please try again in a few moments."
    
    if not collection:
        print("RAG Pipeline: ERROR - ChromaDB collection not available")
        return "Sorry, I couldn't connect to the database. This means I'm falling back to general knowledge rather than using specific information about places. Please try again later for more specific answers."
    
    # Fast path for simple queries
    simple_queries = {
        "hi": "Hello! I'm your travel assistant. How can I help you today?",
        "hello": "Hello! I'm your travel assistant. How can I help you today?",
        "hey": "Hi there! I'm here to answer your questions about attractions.",
        "thanks": "You're welcome! Feel free to ask if you have more questions.",
        "thank you": "You're welcome! I'm happy to help with your travel questions."
    }
    
    # Check for simple queries first for instant response
    query_lower = query.lower().strip()
    if query_lower in simple_queries:
        return simple_queries[query_lower]
    
    # Step 1: Understand query intent and extract place name
    try:
        intent_response = llm.invoke(intent_prompt.format(query=query))
        # Parse intent and place name from the response
        intent_lines = intent_response.split("\n")
        intent = intent_lines[0] if intent_lines else "General inquiry about the place."
        place_name = None
        for line in intent_lines:
            if line.startswith("Place Name:"):
                place_name = line.replace("Place Name:", "").strip()
                break
        
        print(f"RAG Pipeline: LLM extracted intent: {intent}")
        print(f"RAG Pipeline: LLM extracted place name: {place_name}")
        
        # Fallback: Extract place name directly from query if LLM fails
        if not place_name:
            place_name = extract_place_name(query)
            print(f"RAG Pipeline: Fallback place name extraction: {place_name}")
    except Exception as e:
        print(f"RAG Pipeline: Error during intent extraction: {str(e)}")
        intent = "General inquiry about the place."
        place_name = extract_place_name(query)
        print(f"RAG Pipeline: Fallback place name: {place_name}")

    # Step 2: Retrieve relevant summaries from ChromaDB
    try:
        print(f"RAG Pipeline: Querying ChromaDB with: {query}")
        query_result = collection.query(query_texts=[query], n_results=3)
        
        if not query_result or not query_result['documents'] or len(query_result['documents'][0]) == 0:
            print("RAG Pipeline: ERROR - No results returned from ChromaDB query")
            # Fall back to general Gemini response with a notice
            return f"I don't have specific information about {place_name if place_name else 'this topic'} in my database. Here's a general response: " + llm.invoke(f"Answer this question briefly: {query}")
        
        retrieved_summaries = query_result['documents'][0]
        retrieved_metadata = query_result['metadatas'][0]
        
        print(f"RAG Pipeline: Retrieved {len(retrieved_summaries)} summaries from ChromaDB")
        for i, (summary, metadata) in enumerate(zip(retrieved_summaries, retrieved_metadata)):
            print(f"RAG Pipeline: Summary {i+1} - Place: {metadata['place_name']}")
        
        # Filter summaries to only include the specific place, if a place name was identified
        if place_name:
            print(f"RAG Pipeline: Filtering summaries for place: {place_name}")
            filtered_summaries = []
            filtered_metadata = []
            for summary, metadata in zip(retrieved_summaries, retrieved_metadata):
                if metadata['place_name'].lower() == place_name.lower():
                    filtered_summaries.append(summary)
                    filtered_metadata.append(metadata)
            
            print(f"RAG Pipeline: After filtering, found {len(filtered_summaries)} relevant summaries")
        else:
            filtered_summaries = retrieved_summaries
            filtered_metadata = retrieved_metadata
        
        # Format the summaries with place names for the LLM
        summaries_text = "\n".join(
            f"{metadata['place_name']}: {summary}"
            for summary, metadata in zip(filtered_summaries, filtered_metadata)
        )
        
        if not summaries_text:
            print("RAG Pipeline: No relevant summaries found after filtering")
            summaries_text = f"No relevant summary found for {place_name}." if place_name else "No relevant summaries found."
    except Exception as e:
        print(f"RAG Pipeline: ERROR during ChromaDB retrieval: {str(e)}")
        summaries_text = f"Error retrieving information from database: {str(e)}"
        # Fall back to general Gemini response with a notice
        return f"I encountered an error while searching my database. Here's a general response instead: " + llm.invoke(f"Answer this question briefly: {query}")

    # Step 3: Generate the final answer using the retrieved summaries
    try:
        # Use the place name in the prompt, or default to a generic response if not specified
        if not place_name:
            place_name = "the place"
            
        print(f"RAG Pipeline: Generating answer about {place_name} with {len(summaries_text)} characters of context")
        
        answer = llm.invoke(answer_prompt.format(
            query=query,
            place_name=place_name,
            summaries=summaries_text
        ))
        
        print("RAG Pipeline: Successfully generated answer")
        return answer
    except Exception as e:
        print(f"RAG Pipeline: Error generating answer: {str(e)}")
        return f"Sorry, I encountered an error while generating your answer: {str(e)}"