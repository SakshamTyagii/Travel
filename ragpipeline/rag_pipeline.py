import requests
import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for consistency, though not used here)
load_dotenv()

# Step 1: Verify Ollama server is running
try:
    response = requests.get("http://localhost:11434")
    if response.status_code != 200:
        raise Exception("Ollama server is not running.")
except Exception as e:
    print(f"Failed to connect to Ollama server: {e}")
    print("Please start Ollama with 'ollama run llama2' and try again.")
    exit(1)

# Step 2: Initialize the Ollama LLM (LLaMA2)
llm = OllamaLLM(model="llama2", base_url="http://localhost:11434")

# Step 3: Load ChromaDB vector database
import os
chroma_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db")
client = chromadb.PersistentClient(path=chroma_path)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
try:
    collection = client.get_collection(
        name="google_maps_summaries",
        embedding_function=embedding_function
    )
except Exception as e:
    print(f"Error loading ChromaDB collection: {str(e)}")
    print("Ensure the ChromaDB database exists at ../data/chroma_db with the 'google_maps_summaries' collection.")
    exit(1)

# Step 4: Define the RAG Pipeline
# Prompt to understand query intent and extract the place name
intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="Analyze the following user query and describe its intent in one sentence, focusing on what specific information the user is seeking (e.g., best time to visit, suitability for families, activities available). Also, identify the specific place name mentioned in the query, if any:\n\nQuery: {query}\n\nIntent:\nPlace Name:"
)

# Prompt to generate the final answer
answer_prompt = PromptTemplate(
    input_variables=["query", "place_name", "summaries"],
    template="The user query is about {place_name}. Based on the following review summary for {place_name}, answer the user's query in 1-2 sentences, focusing on specific details like best visiting times, busy hours, suitability for specific groups, or unique features if mentioned. If the information is not available, provide a general response based on typical patterns (e.g., beaches are often less busy early morning):\n\nQuery: {query}\n\nSummary for {place_name}:\n{summaries}\n\nAnswer:"
)

def extract_place_name(query):
    """Extract the place name from the query for filtering summaries"""
    # Simple heuristic: Look for known place names in the query
    known_places = ["Calangute Beach", "Palolem Beach", "Chapora River"]  # Add more as needed
    query_lower = query.lower()
    for place in known_places:
        if place.lower() in query_lower:
            return place
    return None

def rag_pipeline(query):
    """Implement the RAG pipeline: Understand intent, retrieve summaries, generate answer"""
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
    
    # For non-simple queries, proceed with your existing pipeline...
    try:
        # Step 4.1: Understand query intent and extract place name
        intent_response = llm.invoke(intent_prompt.format(query=query))
        # Parse intent and place name from the response
        intent_lines = intent_response.split("\n")
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

    # Step 4.2: Retrieve relevant summaries from ChromaDB
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

    # Step 4.3: Generate the final answer using the retrieved summaries
    try:
        # Use the place name in the prompt, or default to a generic response if not specified
        if not place_name:
            place_name = "the place"
        answer = llm.invoke(answer_prompt.format(
            query=query,
            place_name=place_name,
            summaries=summaries_text
        ))
        return answer
    except Exception as e:
        return f"Sorry, I couldn't generate an answer due to an error: {str(e)}"

# Modify the code to only run when directly executed
if __name__ == "__main__":
    print("Welcome to the Google Maps Review Assistant!")
    print("You can ask questions about the places based on their review summaries.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("Your question: ")
        if user_input.lower() == 'exit':
            break
        answer = rag_pipeline(user_input)
        print(f"Answer: {answer}\n")