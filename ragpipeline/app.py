import streamlit as st
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
    st.error(f"Failed to connect to Ollama server: {e}")
    st.error("Please start Ollama with 'ollama run llama2' and refresh the app.")
    st.stop()

# Step 2: Initialize the Ollama LLM (LLaMA2)
llm = OllamaLLM(model="llama2", base_url="http://localhost:11434")

# Step 3: Load ChromaDB vector database
client = chromadb.PersistentClient(path="../data/chroma_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
try:
    collection = client.get_collection(
        name="google_maps_summaries",
        embedding_function=embedding_function
    )
except Exception as e:
    st.error(f"Error loading ChromaDB collection: {str(e)}")
    st.error("Ensure the ChromaDB database exists at ../data/chroma_db with the 'google_maps_summaries' collection.")
    st.stop()

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
    known_places = ["Calangute Beach", "Palolem Beach", "Chapora River"]  # Add more as needed
    query_lower = query.lower()
    for place in known_places:
        if place.lower() in query_lower:
            return place
    return None

def rag_pipeline(query):
    """Implement the RAG pipeline: Understand intent, retrieve summaries, generate answer"""
    # Step 4.1: Understand query intent and extract place name
    try:
        intent_response = llm.invoke(intent_prompt.format(query=query))
        intent_lines = intent_response.split("\n")
        intent = intent_lines[0] if intent_lines else "General inquiry about the place."
        place_name = None
        for line in intent_lines:
            if line.startswith("Place Name:"):
                place_name = line.replace("Place Name:", "").strip()
                break
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
        if not place_name:
            place_name = "the place"
        answer = llm.invoke(answer_prompt.format(
            query=query,
            place_name=place_name,
            summaries=summaries_text
        ))
        return answer
    except Exception as e:
        return "Sorry, I couldn't generate an answer due to an error."

# Step 5: Streamlit App Interface
st.title("Hey There! I am your travel assistant.")
# st.write("Ask questions about places based on their review summaries.")

# User input
query = st.text_input("Your question:", placeholder="e.g., What's the best time to visit Calangute Beach?")

# Submit button
if st.button("Submit"):
    if not query:
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Processing your query..."):
            answer = rag_pipeline(query)
            st.write(f"**Answer:** {answer}")