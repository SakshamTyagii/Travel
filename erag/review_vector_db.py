import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import json
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from parent directory
load_dotenv(Path("../.env"))

class ReviewVectorDatabase:
    def __init__(self, db_path="review_vectors"):
        """Initialize the vector database system"""
        print("🚀 Initializing Review Vector Database...")
        
        # Initialize ChromaDB with absolute path
        self.db_path = Path(db_path).absolute()
        self.db_path.mkdir(exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            print(f"✅ ChromaDB initialized at: {self.db_path}")
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            raise
        
        # Initialize embedding function
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            print("✅ Embedding function loaded")
        except Exception as e:
            print(f"❌ Embedding function failed: {e}")
            raise
        
        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment variables")
            print("Please check your .env file in the parent directory")
            raise ValueError("Missing GEMINI_API_KEY")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            print("✅ Gemini AI configured")
        except Exception as e:
            print(f"❌ Gemini configuration failed: {e}")
            raise
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name="reviews",
                embedding_function=self.embedding_function
            )
            print("✅ Created new review collection")
        except Exception as e:
            try:
                self.collection = self.client.get_collection(
                    name="reviews",
                    embedding_function=self.embedding_function
                )
                print(f"✅ Loaded existing collection with {self.collection.count()} reviews")
            except Exception as e2:
                print(f"❌ Collection error: {e2}")
                raise

        # Define the 26 containers
        self.containers = [
            "Crowd Levels & Timing Suggestions", "Weather & Best Season to Visit", 
            "Accessibility & Mobility", "Cleanliness & Maintenance", "Safety & Security",
            "Staff, Guides & Management", "Tickets, Entry & Pricing", "Signal Connectivity",
            "Suitability for specific audiences", "Historical or Cultural Significance",
            "Architecture & Aesthetics", "Spiritual or Religious Vibe",
            "Natural Beauty & Photogenic Spots", "Wildlife & Biodiversity",
            "Adventure & Activities", "Peace & Relaxation", "Shops & Stalls",
            "Food & Local Cuisine", "Events & Entertainment", "Navigation & Signage",
            "Amenities & Facilities", "Transport Connectivity", "Value for Time & Money",
            "Local Tips & Insider Advice", "Comparisons & Alternatives", "Emotional Tone / Vibe"
        ]

    def load_reviews_to_vector_db(self, csv_path="../data/google_maps_reviews.csv"):
        """Load all reviews into vector database - ONE TIME SETUP"""
        print(f"📥 Loading reviews from {csv_path}...")
        
        # Try different CSV paths
        csv_paths_to_try = [
            csv_path,
            "../data/google_maps_reviews.csv",
            "../../data/google_maps_reviews.csv",
            Path(__file__).parent.parent / "data" / "google_maps_reviews.csv"
        ]
        
        df = None
        actual_path = None
        
        for path in csv_paths_to_try:
            try:
                df = pd.read_csv(path)
                actual_path = path
                print(f"✅ Found CSV at: {path}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
                continue
        
        if df is None:
            print("❌ Could not find google_maps_reviews.csv in any of these locations:")
            for path in csv_paths_to_try:
                print(f"   - {path}")
            return False
        
        print(f"📊 Found {len(df)} total records")
        
        # Check required columns
        required_columns = ['review_text', 'place_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Prepare data for vector storage
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Check if review_text is valid
            review_text = row['review_text']
            if pd.isna(review_text) or not str(review_text).strip():
                continue
                
            place_name = row['place_name']
            if pd.isna(place_name):
                continue
            
            documents.append(str(review_text))
            metadatas.append({
                "place_name": str(place_name),
                "rating": float(row.get('rating', 0)) if pd.notna(row.get('rating')) else 0.0,
                "reviewer_name": str(row.get('reviewer_name', '')) if pd.notna(row.get('reviewer_name')) else "",
                "date": str(row.get('date', '')) if pd.notna(row.get('date')) else "",
                "review_id": str(row.get('review_id', f'review_{idx}'))
            })
            ids.append(f"review_{idx}")
        
        if not documents:
            print("❌ No valid reviews found in the CSV")
            return False
        
        print(f"📊 Processing {len(documents)} valid reviews...")
        
        # Clear existing collection if it has data
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                print(f"🗑️ Clearing existing {existing_count} reviews...")
                # Delete the collection and recreate
                self.client.delete_collection("reviews")
                self.collection = self.client.create_collection(
                    name="reviews",
                    embedding_function=self.embedding_function
                )
        except Exception as e:
            print(f"Warning: Could not clear existing data: {e}")
        
        # Add to vector database in batches
        batch_size = 50  # Smaller batches for stability
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            try:
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
                print(f"✅ Added batch {i//batch_size + 1}/{total_batches} ({len(batch_docs)} reviews)")
            except Exception as e:
                print(f"❌ Error adding batch {i//batch_size + 1}: {e}")
                return False
        
        final_count = self.collection.count()
        print(f"🎉 Successfully loaded {final_count} reviews into vector database!")
        return True

    def get_locations_list(self):
        """Get all unique locations in the database"""
        try:
            # Check if collection exists and has data
            count = self.collection.count()
            if count == 0:
                print("⚠️ No reviews found in database. Please load reviews first.")
                return []
            
            # Get all documents to extract unique places
            all_data = self.collection.get()
            places = set()
            
            for metadata in all_data['metadatas']:
                places.add(metadata['place_name'])
            
            return sorted(list(places))
        except Exception as e:
            print(f"❌ Error getting locations: {e}")
            return []

    def query_location_reviews(self, location_name):
        """Get all reviews for a specific location"""
        try:
            # Get all reviews for this location
            results = self.collection.get(
                where={"place_name": location_name}
            )
            return results
        except Exception as e:
            print(f"❌ Error querying location: {e}")
            return {"documents": [], "metadatas": []}

    def generate_enhanced_summary(self, location_name):
        """Generate enhanced summary using RAG"""
        print(f"🎯 Generating enhanced summary for: {location_name}")
        
        # Get all reviews for the location
        results = self.query_location_reviews(location_name)
        
        if not results['documents']:
            return {"error": f"No reviews found for {location_name}"}
        
        reviews_text = "\n\n---REVIEW---\n\n".join(results['documents'])
        total_reviews = len(results['documents'])
        
        print(f"📊 Analyzing {total_reviews} reviews...")
        
        # Enhanced prompt with 26 containers
        prompt = f"""
Analyze ALL {total_reviews} reviews for {location_name} and create a comprehensive enhanced summary.

CRITICAL CONSOLIDATION RULES:
1. Combine similar points into single comprehensive statements
2. If multiple reviews mention "clean", create: "Consistently described as clean and well-maintained by most reviewers"
3. Preserve ALL specific details (times, prices, activities, etc.)
4. Handle conflicts by stating different perspectives in the same point
5. Group related information logically
6. Eliminate redundancy - each bullet point should be unique and valuable
7. If many reviews say similar things, mention the frequency: "Most reviewers", "Several visitors noted", etc.

Categorize information into these containers (only include containers that have relevant information):
{', '.join(self.containers)}

ALL REVIEWS FOR {location_name}:
{reviews_text}

Output format - ONLY return valid JSON:
{{
    "location_name": "{location_name}",
    "total_reviews_analyzed": {total_reviews},
    "summary": {{
        "container_name": ["consolidated comprehensive statement", "unique detail"],
        "another_container": ["consolidated content"]
    }}
}}

EXAMPLES of good consolidation:
- Instead of multiple "clean" entries: "Consistently rated as very clean and well-maintained by 85% of reviewers"
- Instead of multiple timing entries: "Best visited during early morning (6-9 AM) or late afternoon (4-6 PM) to avoid crowds and heat"
- Handle conflicts: "While most found it peaceful, some visitors noted crowds during weekends and festivals"
- Preserve specifics: "Entry fee ₹25 for adults, ₹10 for children; parking ₹50 for cars"

Return ONLY the JSON response.
"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            # Try to parse JSON
            result = json.loads(response_text)
            print(f"✅ Generated enhanced summary for {location_name}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response text: {response_text[:500]}...")
            return {"error": f"JSON parsing failed: {e}"}
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return {"error": str(e)}

    def get_database_stats(self):
        """Get statistics about the vector database"""
        try:
            total_reviews = self.collection.count()
            locations = self.get_locations_list()
            
            stats = {
                "total_reviews": total_reviews,
                "total_locations": len(locations),
                "locations": locations[:10] if locations else [],
                "database_path": str(self.db_path)
            }
            
            return stats
        except Exception as e:
            return {"error": str(e)}

    def query_specific_aspect(self, location_name, aspect_query):
        """Query specific aspects about a location"""
        print(f"🔍 Querying {location_name} for: {aspect_query}")
        
        try:
            # Get reviews for the location
            results = self.query_location_reviews(location_name)
            
            if not results['documents']:
                return f"No reviews found for {location_name}"
            
            # Use first 20 reviews to avoid token limits
            location_reviews = "\n\n".join(results['documents'][:20])
            
            prompt = f"""
Based on these reviews for {location_name}, answer the specific question: "{aspect_query}"

Reviews:
{location_reviews}

Provide a comprehensive answer that:
1. Consolidates similar points from multiple reviews
2. Includes all specific details (times, prices, etc.)
3. Handles conflicting opinions by mentioning both sides
4. Gives actionable insights

Answer the question directly and concisely:
"""
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Error: {e}"

    def generate_all_summaries(self, output_dir="rag_summaries"):
        """Generate enhanced summaries for all locations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        locations = self.get_locations_list()
        if not locations:
            print("❌ No locations found. Please load reviews first.")
            return
        
        print(f"🏭 Generating summaries for {len(locations)} locations...")
        
        successful = 0
        failed = 0
        
        for i, location in enumerate(locations, 1):
            print(f"\n[{i}/{len(locations)}] Processing: {location}")
            
            summary = self.generate_enhanced_summary(location)
            
            if "error" not in summary:
                # Save summary
                safe_name = "".join(c for c in location if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filepath = output_path / f"{safe_name}_enhanced.json"
                
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    
                    print(f"✅ Saved: {filepath.name}")
                    successful += 1
                except Exception as e:
                    print(f"❌ Error saving {location}: {e}")
                    failed += 1
            else:
                print(f"❌ Error processing {location}: {summary.get('error', 'Unknown error')}")
                failed += 1
        
        print(f"\n🎉 Summary Generation Complete!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Summaries saved in: {output_path}")