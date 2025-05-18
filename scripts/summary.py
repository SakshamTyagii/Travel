import pandas as pd
import requests
import chromadb
from chromadb.utils import embedding_functions
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Ensure data directory exists
os.makedirs("../data", exist_ok=True)

# Step 1: Load the reviews
try:
    reviews_df = pd.read_csv("../data/google_maps_reviews.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Group reviews by place_name and limit review text length
def chunk_reviews(reviews, max_chars=8000):
    """Split reviews into smaller chunks to avoid memory issues"""
    reviews = reviews[:max_chars] if len(reviews) > max_chars else reviews
    return reviews

grouped_reviews = reviews_df.groupby('place_name')['review_text'].apply(lambda x: chunk_reviews(' '.join(x))).reset_index()

# Step 2: Define Gemini API interaction
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_gemini_api(prompt):
    """Call the Gemini API to generate a detailed summary with retry logic"""
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 300,
        }
    }
    
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            raise ValueError("No valid response from Gemini API")
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        raise  # Re-raise the exception for retry handling

# Step 3: Create the detailed summary prompt
summary_prompt = """Generate a detailed summary of the following reviews in 5-7 sentences, focusing on the overall sentiment, common themes, and specific details that would help visitors plan their trip. Include mentions of busy times, visiting hours, wait times, service speed, cleanliness, accessibility, or unique features if present. Highlight any notable positive or negative aspects mentioned frequently, and note the suitability for specific groups (e.g., families, solo travelers, adventure seekers). If reviews are too short or contradictory, state the lack of clear consensus and provide a general impression based on available information. Ensure the summary is concise yet informative for storage in a vector database.

Reviews:
{reviews}

Summary:"""

# Step 4: Generate summaries for each place
summaries = []
for _, row in grouped_reviews.iterrows():
    place_name = row['place_name']
    print(f"Summarizing reviews for {place_name}...")
    try:
        # Prepare the prompt with the reviews
        prompt = summary_prompt.format(reviews=row['review_text'])
        # Call Gemini API to generate the summary
        summary = call_gemini_api(prompt)
        summaries.append({
            'place_name': place_name,
            'summary': summary
        })
        time.sleep(1)  # Add 1 second delay between requests
    except Exception as e:
        print(f"Error summarizing reviews for {place_name}: {str(e)}")
        summaries.append({
            'place_name': place_name,
            'summary': f"Error generating summary: {str(e)}"
        })

# Save summaries to CSV
summaries_df = pd.DataFrame(summaries)
summaries_df.to_csv("../data/review_summaries.csv", index=False)
print(f"✅ Generated detailed summaries for {len(summaries)} places and saved to ../data/review_summaries.csv")

# Step 5: Store summaries in ChromaDB
# Initialize ChromaDB client (persistent storage)
client = chromadb.PersistentClient(path="../data/chroma_db")

# Use HuggingFace embeddings (all-MiniLM-L6-v2) for ChromaDB
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get a collection in ChromaDB
collection_name = "google_maps_summaries"
try:
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
except Exception as e:
    print(f"Error creating ChromaDB collection: {str(e)}")
    exit(1)

# Prepare data for ChromaDB
documents = []
metadatas = []
ids = []
for idx, row in summaries_df.iterrows():
    documents.append(row['summary'])
    metadatas.append({"place_name": row['place_name']})
    ids.append(f"summary_{idx}")

# Add summaries to ChromaDB
try:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ Stored {len(documents)} summaries in ChromaDB collection '{collection_name}'")
except Exception as e:
    print(f"Error storing summaries in ChromaDB: {str(e)}")

# Verify storage by querying the collection
try:
    count = collection.count()
    print(f"Total documents in ChromaDB: {count}")
    # Example query to test retrieval
    query_result = collection.query(query_texts=["beach"], n_results=2)
    print("\nExample query result for 'beach':")
    for doc, metadata in zip(query_result['documents'][0], query_result['metadatas'][0]):
        print(f"Place: {metadata['place_name']}, Summary: {doc[:100]}...")
except Exception as e:
    print(f"Error querying ChromaDB: {str(e)}")

print("Done!")