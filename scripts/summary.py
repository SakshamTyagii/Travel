import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import requests

# Ensure data directory exists
os.makedirs("../data", exist_ok=True)

# Step 1: Verify Ollama server is running
try:
    response = requests.get("http://localhost:11434")
    if response.status_code != 200:
        raise Exception("Ollama server is not running.")
except Exception as e:
    print(f"Failed to connect to Ollama server: {e}")
    exit(1)

# Step 2: Load the reviews
try:
    reviews_df = pd.read_csv("../data/google_maps_reviews.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Group reviews by place_name but limit review text length
def chunk_reviews(reviews, max_chars=8000):
    """Split reviews into smaller chunks to avoid memory issues"""
    reviews = reviews[:max_chars] if len(reviews) > max_chars else reviews
    return reviews

grouped_reviews = reviews_df.groupby('place_name')['review_text'].apply(lambda x: chunk_reviews(' '.join(x))).reset_index()

# Step 3: Initialize the Ollama LLM
llm = OllamaLLM(model="llama2", base_url="http://localhost:11434")

# Step 4: Create the summary prompt and chain with error handling
summary_prompt_template = PromptTemplate(
    input_variables=["reviews"],
    template="Summarize the following reviews in 2-3 sentences, focusing on the overall sentiment, common themes, and key points such as mentions of busy times, visiting hours, wait times, or service speed. If reviews are too short or contradictory, note the lack of clear consensus:\n\n{reviews}"
)

summary_chain = summary_prompt_template | llm

# Step 5: Process reviews with error handling
summaries = []
for _, row in grouped_reviews.iterrows():
    try:
        summary = summary_chain.invoke({"reviews": row['review_text']})
        summaries.append({
            'place_name': row['place_name'],
            'summary': summary
        })
    except Exception as e:
        print(f"Error summarizing reviews for {row['place_name']}: {str(e)}")
        # Add a placeholder for failed summaries
        summaries.append({
            'place_name': row['place_name'],
            'summary': f"Error generating summary: {str(e)}"
        })

# Convert summaries to DataFrame and save
summaries_df = pd.DataFrame(summaries)
summaries_df.to_csv("../data/review_summaries.csv", index=False)
print(f"✅ Generated summaries for {len(summaries)} places")

# Step 5: Create a FAISS-based database from summaries
combined_summaries = "\n\n".join(
    f"{row['place_name']}:\n{row['summary']}" for _, row in summaries_df.iterrows()
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_text(combined_summaries)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(docs, embeddings)

# Save FAISS index to disk
vector_store.save_local("../data/faiss_index")
print("FAISS index saved to ../data/faiss_index")

# Step 6: Create a conversational interface for user queries
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Based on the following review summaries, answer the user's question concisely in 1-2 sentences, focusing on location-specific details like best visiting times, busy hours, opening hours, wait times, or service speed if mentioned. For list-based questions (e.g., 'which places'), list all relevant places with a brief explanation. If no specific information is found, state so and provide general advice based on typical patterns (e.g., cafes are often less busy early morning):\n\n{chat_history}\n\nQuestion: {question}\n\nAnswer:"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt_template}
)

# Step 7: Interactive query loop with debugging
print("\nYou can now ask questions about the reviews and location details! Type 'exit' to quit.")
while True:
    query = input("Your question: ").strip()
    if not query:
        print("Please enter a valid question.")
        continue
    if query.lower() == "exit":
        break
    try:
        # Debug: Log retrieved documents
        retrieved_docs = vector_store.as_retriever().invoke(query)
        print(f"[DEBUG] Retrieved {len(retrieved_docs)} documents for query '{query}':")
        for i, doc in enumerate(retrieved_docs):
            print(f"Doc {i+1}: {doc.page_content[:100]}...")

        response = qa_chain.invoke({"question": query})
        print(f"Answer: {response['answer']}\n")
    except Exception as e:
        print(f"Error answering query: {type(e).__name__} - {str(e)}\n")

print("Exiting program.")