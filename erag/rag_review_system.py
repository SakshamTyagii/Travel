#!/usr/bin/env python3

import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from review_vector_db import ReviewVectorDatabase
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the erag directory and all files are present")
    sys.exit(1)

def main():
    """Main entry point for the RAG Review System"""
    print("🚀 RAG-Based Review Analysis System")
    print("="*50)
    
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1]
    
    try:
        if command == "--setup":
            setup_system()
        elif command == "--load":
            load_reviews()
        elif command == "--generate-all":
            generate_all_summaries()
        elif command == "--generate-json":
            generate_json_for_gcp()
        elif command == "--location":
            if len(sys.argv) > 2:
                get_location_summary(sys.argv[2])
            else:
                print("❌ Please provide location name")
                print("Example: python rag_review_system.py --location 'Aguada Fort'")
        elif command == "--interactive":
            start_interactive_mode()
        elif command == "--stats":
            show_database_stats()
        elif command == "--list":
            list_all_locations()
        elif command == "--test":
            test_system()
        elif command == "--deploy-prep":
            prepare_for_deployment()
        else:
            print(f"❌ Unknown command: {command}")
            show_usage()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def show_usage():
    """Show usage information"""
    print("\n📖 Usage:")
    print("="*50)
    print("🔧 Setup & Data Loading:")
    print("  python rag_review_system.py --setup         # Complete setup")
    print("  python rag_review_system.py --load          # Load reviews to vector DB")
    print("  python rag_review_system.py --test          # Test system")
    print()
    print("📊 Generate Summaries:")
    print("  python rag_review_system.py --generate-all  # Generate all summaries")
    print("  python rag_review_system.py --generate-json # Generate JSON for GCP")
    print("  python rag_review_system.py --location 'Aguada Fort'  # Get specific location")
    print()
    print("🚀 GCP Deployment:")
    print("  python rag_review_system.py --deploy-prep   # Prepare all files for GCP")
    print()
    print("📋 Information:")
    print("  python rag_review_system.py --list          # List all locations")
    print("  python rag_review_system.py --stats         # Show database stats")
    print("  python rag_review_system.py --interactive   # Interactive mode")
    print()
    print("🎯 Complete workflow for GCP deployment:")
    print("  1. python rag_review_system.py --setup      # Setup vector database")
    print("  2. python rag_review_system.py --deploy-prep # Prepare for deployment")
    print("  3. Deploy to GCP Cloud Run")

def test_system():
    """Test system setup"""
    print("🧪 Testing system setup...")
    print("="*50)
    
    try:
        print("1. Testing imports...")
        from review_vector_db import ReviewVectorDatabase
        print("✅ Imports successful")
        
        print("\n2. Testing database initialization...")
        db = ReviewVectorDatabase()
        print("✅ Database initialization successful")
        
        print("\n3. Testing stats...")
        stats = db.get_database_stats()
        if "error" in stats:
            print(f"⚠️ Stats error: {stats['error']}")
        else:
            print(f"✅ Database stats: {stats['total_reviews']} reviews, {stats['total_locations']} locations")
        
        print("\n🎉 System test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def setup_system():
    """Complete system setup"""
    print("🔧 Setting up RAG Review System...")
    print("="*50)
    
    try:
        print("Step 1: Initializing vector database...")
        db = ReviewVectorDatabase()
        
        print("\nStep 2: Loading reviews into vector database...")
        success = db.load_reviews_to_vector_db()
        
        if success:
            print("\n✅ Setup completed successfully!")
            print("🎯 You can now use:")
            print("   python rag_review_system.py --list")
            print("   python rag_review_system.py --deploy-prep")
        else:
            print("\n❌ Setup failed. Please check:")
            print("1. CSV file exists at ../data/google_maps_reviews.csv")
            print("2. .env file exists with GEMINI_API_KEY")
            print("3. All dependencies are installed")
    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback
        traceback.print_exc()

def load_reviews():
    """Load reviews into vector database"""
    print("📥 Loading reviews into vector database...")
    try:
        db = ReviewVectorDatabase()
        success = db.load_reviews_to_vector_db()
        if success:
            print("✅ Reviews loaded successfully!")
        else:
            print("❌ Failed to load reviews")
    except Exception as e:
        print(f"❌ Error: {e}")

def generate_all_summaries():
    """Generate enhanced summaries for all locations"""
    print("🏭 Generating enhanced summaries for all locations...")
    try:
        db = ReviewVectorDatabase()
        db.generate_all_summaries()
    except Exception as e:
        print(f"❌ Error: {e}")

def generate_json_for_gcp():
    """Generate JSON file compatible with GCP RAG system"""
    print("🏭 Generating JSON file for GCP integration...")
    print("="*50)
    
    try:
        db = ReviewVectorDatabase()
        locations = db.get_locations_list()
        
        if not locations:
            print("❌ No locations found. Please load reviews first.")
            return None
        
        print(f"📍 Found {len(locations)} locations to process")
        
        # Create enhanced summaries in GCP-compatible format
        enhanced_summaries = {}
        successful = 0
        failed = 0
        
        for i, location in enumerate(locations, 1):
            print(f"🎯 Processing {i}/{len(locations)}: {location}")
            
            try:
                summary = db.generate_enhanced_summary(location)
                if "error" not in summary:
                    # Convert to GCP format
                    enhanced_summaries[location] = {
                        "place_name": location,
                        "summary": convert_to_simple_format(summary),
                        "enhanced_data": summary,
                        "last_updated": datetime.now().isoformat(),
                        "total_reviews": summary.get("total_reviews_analyzed", 0)
                    }
                    successful += 1
                    print(f"✅ Processed {location}")
                else:
                    print(f"❌ Failed: {location} - {summary['error']}")
                    failed += 1
            except Exception as e:
                print(f"❌ Error processing {location}: {e}")
                failed += 1
        
        if enhanced_summaries:
            # Save JSON file for GCP
            output_file = Path("../ragpipeline/data/enhanced_place_summaries.json")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_summaries, f, indent=2, ensure_ascii=False)
            
            print(f"\n🎉 Successfully generated {successful} enhanced summaries!")
            print(f"❌ Failed: {failed}")
            print(f"📁 File saved: {output_file}")
            print(f"🚀 Ready for GCP deployment!")
            
            return str(output_file)
        else:
            print("❌ No summaries generated successfully")
            return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def convert_to_simple_format(enhanced_summary):
    """Convert enhanced summary to simple text format for GCP compatibility"""
    summary_parts = []
    
    summary_data = enhanced_summary.get("summary", {})
    for container, points in summary_data.items():
        if points and isinstance(points, list):
            # Clean container name and combine points
            clean_container = container.replace("&", "and")
            combined_points = " ".join(points)
            summary_parts.append(f"{clean_container}: {combined_points}")
    
    return " | ".join(summary_parts) if summary_parts else "No summary available"

def prepare_for_deployment():
    """Prepare all files needed for GCP deployment"""
    print("🚀 Preparing enhanced RAG system for GCP deployment...")
    print("="*50)
    
    try:
        # Step 1: Generate JSON summaries
        print("Step 1: Generating enhanced JSON summaries...")
        json_file = generate_json_for_gcp()
        
        if not json_file:
            print("❌ Failed to generate JSON file")
            return
        
        # Step 2: Create new enhanced RAG API
        print("\nStep 2: Creating enhanced RAG API...")
        create_enhanced_rag_api()
        
        # Step 3: Update requirements
        print("\nStep 3: Updating requirements...")
        update_requirements_for_gcp()
        
        # Step 4: Create Dockerfile
        print("\nStep 4: Creating Dockerfile...")
        create_dockerfile()
        
        print("\n🎉 Deployment preparation complete!")
        print("📁 Files created in ../ragpipeline/:")
        print("   - enhanced_place_summaries.json")
        print("   - enhanced_rag_api.py")
        print("   - requirements.txt (updated)")
        print("   - Dockerfile")
        print("\n🚀 Ready to deploy to GCP!")
        
    except Exception as e:
        print(f"❌ Error preparing deployment: {e}")

def create_enhanced_rag_api():
    """Create enhanced RAG API for GCP deployment"""
    api_code = '''
import os
import json
import logging
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Travel Assistant RAG API", version="2.0.0")

class QueryRequest(BaseModel):
    query: str

class LocationRequest(BaseModel):
    location: str
    question: str = None

class EnhancedRAG:
    def __init__(self, json_path: str, gemini_api_key: str):
        self.json_path = json_path
        self.gemini_api_key = gemini_api_key
        
        # Configure Gemini API
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Load enhanced summaries
        self.summaries = self._load_summaries()
        logger.info(f"Loaded {len(self.summaries)} enhanced location summaries")
    
    def _load_summaries(self) -> Dict[str, Any]:
        """Load enhanced summaries from JSON file"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading summaries: {e}")
            return {}
    
    def find_relevant_location(self, query: str) -> str:
        """Find the most relevant location based on query"""
        query_lower = query.lower()
        
        # Direct match
        for location in self.summaries.keys():
            if location.lower() in query_lower:
                return location
        
        # Partial match
        for location in self.summaries.keys():
            location_words = location.lower().split()
            if any(word in query_lower for word in location_words if len(word) > 3):
                return location
        
        return None
    
    def get_enhanced_summary(self, location: str) -> Dict[str, Any]:
        """Get enhanced summary for a location"""
        if location in self.summaries:
            return self.summaries[location]
        
        # Try fuzzy match
        for loc_name in self.summaries.keys():
            if location.lower() in loc_name.lower() or loc_name.lower() in location.lower():
                return self.summaries[loc_name]
        
        return None
    
    def query(self, user_query: str) -> str:
        """Process user query and return enhanced response"""
        try:
            # Find relevant location
            location = self.find_relevant_location(user_query)
            
            if not location:
                # General query without specific location
                return self._handle_general_query(user_query)
            
            # Get location data
            location_data = self.get_enhanced_summary(location)
            if not location_data:
                return f"Sorry, I don't have information about {location}."
            
            # Create context-aware response
            context = f"""
Location: {location}
Enhanced Summary: {location_data.get('summary', '')}
Total Reviews Analyzed: {location_data.get('total_reviews', 0)}
Enhanced Data Available: {bool(location_data.get('enhanced_data'))}
"""
            
            prompt = f"""
Based on the following detailed information about {location}, answer the user's question comprehensively.

Context:
{context}

User Question: {user_query}

Provide a helpful, detailed response that:
1. Directly answers the question
2. Uses the enhanced summary data
3. Mentions specific details when available
4. Is conversational and helpful
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I'm sorry, I encountered an error processing your request. Please try again."
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general queries without specific location"""
        locations_list = "\\n".join([f"- {loc}" for loc in self.summaries.keys()])
        
        return f"""I can help you with information about these locations in Delhi and Goa:

{locations_list}

Please specify which location you'd like to know about, or ask a question like:
- "Tell me about Aguada Fort"
- "What's the best time to visit Calangute Beach?"
- "Is Red Fort suitable for families?"
"""

# Initialize the RAG system
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found")
    
    json_path = os.path.join(os.path.dirname(__file__), "data", "enhanced_place_summaries.json")
    rag_system = EnhancedRAG(json_path=json_path, gemini_api_key=gemini_api_key)
    
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_system = None

@app.get("/")
async def root():
    return {
        "message": "Enhanced Travel Assistant RAG API",
        "version": "2.0.0",
        "status": "active" if rag_system else "error",
        "locations_available": len(rag_system.summaries) if rag_system else 0
    }

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Enhanced query endpoint with improved RAG"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        response = rag_system.query(request.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")

@app.post("/location-summary")
async def location_summary(request: LocationRequest):
    """Get enhanced summary for specific location"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        summary = rag_system.get_enhanced_summary(request.location)
        if summary:
            return {"location": request.location, "summary": summary}
        else:
            return {"error": f"Location '{request.location}' not found"}
    except Exception as e:
        logger.error(f"Location summary error: {e}")
        raise HTTPException(status_code=500, detail="Error getting location summary")

@app.get("/locations")
async def list_locations():
    """List all available locations"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    return {
        "locations": list(rag_system.summaries.keys()),
        "total": len(rag_system.summaries)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "locations_count": len(rag_system.summaries) if rag_system else 0
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
    
    # Save the API file
    api_file = Path("../ragpipeline/enhanced_rag_api.py")
    with open(api_file, 'w', encoding='utf-8') as f:
        f.write(api_code)
    
    print(f"✅ Created enhanced RAG API: {api_file}")

def update_requirements_for_gcp():
    """Update requirements.txt for GCP deployment"""
    requirements = '''
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
google-generativeai==0.3.2
python-dotenv==1.0.0
'''
    
    req_file = Path("../ragpipeline/requirements.txt")
    with open(req_file, 'w') as f:
        f.write(requirements.strip())
    
    print(f"✅ Updated requirements: {req_file}")

def create_dockerfile():
    """Create Dockerfile for GCP deployment"""
    dockerfile_content = '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "enhanced_rag_api.py"]
'''
    
    dockerfile_path = Path("../ragpipeline/Dockerfile")
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content.strip())
    
    print(f"✅ Created Dockerfile: {dockerfile_path}")

def test_system():
    """Test system setup"""
    print("🧪 Testing system setup...")
    print("="*50)
    
    try:
        print("1. Testing imports...")
        from review_vector_db import ReviewVectorDatabase
        print("✅ Imports successful")
        
        print("\\n2. Testing database initialization...")
        db = ReviewVectorDatabase()
        print("✅ Database initialization successful")
        
        print("\\n3. Testing stats...")
        stats = db.get_database_stats()
        if "error" in stats:
            print(f"⚠️ Stats error: {stats['error']}")
        else:
            print(f"✅ Database stats: {stats['total_reviews']} reviews, {stats['total_locations']} locations")
        
        print("\\n🎉 System test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def setup_system():
    """Complete system setup"""
    print("🔧 Setting up RAG Review System...")
    print("="*50)
    
    try:
        print("Step 1: Initializing vector database...")
        db = ReviewVectorDatabase()
        
        print("\\nStep 2: Loading reviews into vector database...")
        success = db.load_reviews_to_vector_db()
        
        if success:
            print("\\n✅ Setup completed successfully!")
            print("🎯 You can now use:")
            print("   python rag_review_system.py --list")
            print("   python rag_review_system.py --deploy-prep")
        else:
            print("\\n❌ Setup failed. Please check:")
            print("1. CSV file exists at ../data/google_maps_reviews.csv")
            print("2. .env file exists with GEMINI_API_KEY")
            print("3. All dependencies are installed")
    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback
        traceback.print_exc()

def load_reviews():
    """Load reviews into vector database"""
    print("📥 Loading reviews into vector database...")
    try:
        db = ReviewVectorDatabase()
        success = db.load_reviews_to_vector_db()
        if success:
            print("✅ Reviews loaded successfully!")
        else:
            print("❌ Failed to load reviews")
    except Exception as e:
        print(f"❌ Error: {e}")

def generate_all_summaries():
    """Generate enhanced summaries for all locations"""
    print("🏭 Generating enhanced summaries for all locations...")
    try:
        db = ReviewVectorDatabase()
        db.generate_all_summaries()
    except Exception as e:
        print(f"❌ Error: {e}")

def get_location_summary(location_name):
    """Get enhanced summary for a specific location"""
    print(f"🎯 Getting enhanced summary for: {location_name}")
    
    try:
        db = ReviewVectorDatabase()
        summary = db.generate_enhanced_summary(location_name)
        
        if "error" in summary:
            print(f"❌ Error: {summary['error']}")
        else:
            print("\\n✅ Enhanced Summary:")
            print("="*60)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ Error: {e}")

def start_interactive_mode():
    """Start interactive query mode"""
    try:
        from interactive_query import InteractiveReviewQuery
        query_system = InteractiveReviewQuery()
        query_system.start_interactive_session()
    except ImportError:
        print("❌ Interactive mode not available. Make sure interactive_query.py exists.")
    except Exception as e:
        print(f"❌ Error: {e}")

def show_database_stats():
    """Show database statistics"""
    print("📈 Database Statistics:")
    print("="*50)
    
    try:
        db = ReviewVectorDatabase()
        stats = db.get_database_stats()
        
        if "error" in stats:
            print(f"❌ Error: {stats['error']}")
        else:
            print(f"📊 Total Reviews: {stats['total_reviews']}")
            print(f"📍 Total Locations: {stats['total_locations']}")
            print(f"💾 Database Path: {stats['database_path']}")
            
            if stats['locations']:
                print(f"\\n📍 Sample Locations:")
                for loc in stats['locations']:
                    print(f"   • {loc}")
    except Exception as e:
        print(f"❌ Error: {e}")

def list_all_locations():
    """List all available locations"""
    print("📋 Available Locations:")
    print("="*50)
    
    try:
        db = ReviewVectorDatabase()
        locations = db.get_locations_list()
        
        if locations:
            for i, location in enumerate(locations, 1):
                print(f"{i:3d}. {location}")
        else:
            print("❌ No locations found. Please load reviews first with:")
            print("   python rag_review_system.py --setup")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()