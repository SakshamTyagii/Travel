import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI imports with proper error handling
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    logger.info("✅ Google Generative AI imported successfully")
except ImportError as e:
    GENAI_AVAILABLE = False
    logger.warning(f"⚠️ Google Generative AI not available: {e}")

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    VERTEX_AI_AVAILABLE = True
    logger.info("✅ Vertex AI imported successfully")
except ImportError as e:
    VERTEX_AI_AVAILABLE = False
    logger.warning(f"⚠️ Vertex AI not available: {e}")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Enhanced Travel Assistant RAG API", version="3.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-frontend-e4375.web.app",
        "https://rag-frontend-e4375.firebaseapp.com",
        "https://enhanced-travel-rag-422083294581.asia-south1.run.app",
        "http://localhost:5000",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:5000",
        "http://localhost:8080",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class LocationRequest(BaseModel):
    location: str
    question: str = None

class SimpleContextManager:
    """Simplified context management that works reliably"""
    
    def __init__(self, max_history: int = 10, session_timeout: int = 30):
        self.sessions = {}
        self.max_history = max_history
        self.session_timeout = session_timeout
    
    def get_or_create_session(self, session_id: str = None) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': [],
                'last_location': None,
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
        
        return session_id
    
    def add_interaction(self, session_id: str, query: str, response: str, location: str = None):
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session['history'].append({
            'query': query,
            'response': response,
            'location': location,
            'timestamp': datetime.now()
        })
        
        if location:
            session['last_location'] = location
        
        # Keep history manageable
        if len(session['history']) > self.max_history:
            session['history'] = session['history'][-self.max_history:]
        
        session['last_updated'] = datetime.now()
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        
        # Clean expired sessions
        if datetime.now() - session['last_updated'] > timedelta(minutes=self.session_timeout):
            del self.sessions[session_id]
            return {}
        
        return {
            'last_location': session.get('last_location'),
            'recent_history': session['history'][-3:],
            'session_age': (datetime.now() - session['created_at']).total_seconds() / 60
        }
    
    def cleanup_old_sessions(self):
        now = datetime.now()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if now - session['last_updated'] > timedelta(minutes=self.session_timeout)
        ]
        
        for sid in expired_sessions:
            del self.sessions[sid]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

class EnhancedRAG:
    def __init__(self, json_path: str, gemini_api_key: str = None, project_id: str = None, location: str = "us-central1"):
        self.json_path = json_path
        self.gemini_api_key = gemini_api_key
        self.project_id = project_id
        self.location = location
        
        # Load enhanced summaries FIRST
        self.summaries = self._load_summaries()
        logger.info(f"Loaded {len(self.summaries)} enhanced location summaries")
        
        # Initialize context manager
        self.context_manager = SimpleContextManager()
        
        # Initialize AI models
        self.use_vertex_ai = False
        self.model = None
        self.gemini_model = None
        
        # Try Vertex AI first (for database queries)
        if VERTEX_AI_AVAILABLE and project_id:
            self._init_vertex_ai()
        
        # Initialize Gemini API (for general queries)
        if GENAI_AVAILABLE and gemini_api_key:
            self._init_gemini_api()
        
        if not self.model and not self.gemini_model:
            logger.error("❌ No AI models initialized! Check your API keys and project settings.")
    
    def _init_vertex_ai(self):
        """Initialize Vertex AI for database-specific queries"""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel('gemini-2.0-flash')
            self.use_vertex_ai = True
            
            # Test the model
            test_response = self.model.generate_content("Hello")
            logger.info(f"✅ Vertex AI initialized successfully with project: {self.project_id}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Vertex AI initialization failed: {e}")
            self.model = None
            self.use_vertex_ai = False
            return False
    
    def _init_gemini_api(self):
        """Initialize Gemini API for general queries"""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("✅ Initialized Gemini API for general queries")
        except Exception as e:
            logger.error(f"❌ Gemini API initialization failed: {e}")
            self.gemini_model = None
    
    def _load_summaries(self) -> Dict[str, Any]:
        """Load enhanced summaries from JSON file with multiple fallback paths"""
        possible_paths = [
            self.json_path,
            os.path.join(os.path.dirname(__file__), "data", "enhanced_place_summaries.json"),
            os.path.join(os.getcwd(), "data", "enhanced_place_summaries.json"),
            "./data/enhanced_place_summaries.json",
            "data/enhanced_place_summaries.json",
            "/app/data/enhanced_place_summaries.json",
            "/app/ragpipeline/data/enhanced_place_summaries.json"
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    logger.info(f"📂 Loading summaries from: {path}")
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"✅ Successfully loaded {len(data)} summaries")
                    return data
            except Exception as e:
                logger.warning(f"⚠️ Failed to load from {path}: {e}")
                continue
        
        logger.error("❌ Could not load summaries from any path")
        return {}
    
    def find_relevant_location(self, query: str, context: Dict[str, Any] = None) -> str:
        """Find the most relevant location based on query with enhanced context awareness"""
        query_lower = query.lower().strip()
        
        # Direct match - highest priority
        for location in self.summaries.keys():
            if location.lower() in query_lower:
                return location
        
        # Partial match with longer words
        for location in self.summaries.keys():
            location_words = location.lower().split()
            if any(word in query_lower for word in location_words if len(word) > 3):
                return location
        
        # Enhanced fuzzy matching
        location_variations = {
            'red fort': 'Red Fort',
            'india gate': 'India Gate',
            'lotus temple': 'Lotus Temple',
            'qutub minar': 'Qutub Minar',
            'humayun tomb': "Humayun's Tomb",
            'colva beach': 'Colva Beach',
            'basilica': 'Basilica of Bom Jesus',
            'fort aguada': 'Aguada Fort',
            'akshardham': 'Akshardham Temple',
            'jama masjid': 'Jama Masjid'
        }
        
        for variation, actual_name in location_variations.items():
            if variation in query_lower and actual_name in self.summaries:
                return actual_name
        
        # Enhanced context-based inference
        if context and context.get('last_location'):
            return self._detect_contextual_followup(query_lower, context)
        
        return None

    def _detect_contextual_followup(self, query_lower: str, context: Dict[str, Any]) -> str:
        """Enhanced context detection using review categories"""
        last_location = context.get('last_location')
        if not last_location:
            return None
        
        # Get the location data to access review categories
        location_data = self.get_enhanced_summary(last_location)
        if not location_data:
            return None
        
        # Extract keywords from the review categories for this location
        category_keywords = self._extract_category_keywords(location_data)
        
        # Dynamic pattern matching based on actual review categories
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    logger.info(f"Category-based context match: '{keyword}' from '{category}' detected for {last_location}")
                    return last_location
        
        # Fallback to original contextual indicators
        contextual_indicators = [
            'it', 'this', 'that', 'there', 'here', 'the place', 'the location',
            'this place', 'that place', 'same place'
        ]
        
        question_starters = [
            'is', 'are', 'can', 'will', 'would', 'should', 'do', 'does',
            'how', 'what', 'when', 'where', 'why', 'which'
        ]
        
        # Method 1: Contextual indicators
        for indicator in contextual_indicators:
            if indicator in query_lower:
                logger.info(f"Contextual indicator: '{indicator}' detected for {last_location}")
                return last_location
        
        # Method 2: Short queries starting with question words
        query_words = query_lower.split()
        if (len(query_words) <= 8 and query_words[0] in question_starters):
            logger.info(f"Short question detected for {last_location}: '{query_lower}'")
            return last_location
        
        # Method 3: Very short queries (likely follow-ups)
        if len(query_words) <= 3:
            logger.info(f"Very short query detected for {last_location}: '{query_lower}'")
            return last_location
        
        return None

    def _extract_category_keywords(self, location_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract keywords from review categories for this specific location"""
        enhanced_data = location_data.get('enhanced_data', {})
        
        # Basic category keywords
        category_keyword_map = {
            'timing_and_schedule': [
                'timing', 'time', 'hours', 'open', 'close', 'schedule', 'when',
                'best time', 'visit time', 'opening', 'closing', 'duration'
            ],
            'cost_and_pricing': [
                'cost', 'price', 'fee', 'ticket', 'entry', 'charge', 'expensive',
                'cheap', 'budget', 'money', 'rupees', 'free', 'paid', 'worth'
            ],
            'transportation_and_access': [
                'reach', 'transport', 'metro', 'bus', 'taxi', 'auto', 'uber',
                'distance', 'directions', 'parking', 'accessibility'
            ],
            'crowd_and_peak_times': [
                'crowded', 'busy', 'rush', 'peak', 'weekend', 'weekday',
                'morning', 'evening', 'afternoon', 'queue', 'line'
            ],
            'family_and_kids': [
                'family', 'kids', 'children', 'child', 'baby', 'toddler',
                'stroller', 'child-friendly'
            ],
            'food_and_dining': [
                'food', 'restaurant', 'cafe', 'snacks', 'drink', 'water',
                'hungry', 'thirsty', 'meal', 'lunch', 'dinner'
            ],
            'facilities_and_amenities': [
                'toilet', 'washroom', 'restroom', 'bathroom', 'wifi',
                'facilities', 'amenities', 'shops', 'atm'
            ]
        }
        
        # Build dynamic keywords based on available data
        available_keywords = {}
        
        for category, keywords in category_keyword_map.items():
            if category in enhanced_data or any(keyword in str(enhanced_data).lower() for keyword in keywords[:3]):
                available_keywords[category] = keywords
        
        # Always include basic question patterns
        available_keywords['basic_questions'] = [
            'how', 'what', 'when', 'where', 'why', 'which', 'who',
            'is', 'are', 'can', 'will', 'should', 'would'
        ]
        
        return available_keywords
    
    def get_enhanced_summary(self, location: str) -> Dict[str, Any]:
        """Get enhanced summary for a location"""
        if location in self.summaries:
            return self.summaries[location]
        
        # Fuzzy matching
        for loc_name in self.summaries.keys():
            if location.lower() in loc_name.lower() or loc_name.lower() in location.lower():
                return self.summaries[loc_name]
        
        return None
    
    def generate_database_response(self, query: str, location: str, location_data: Dict, context: Dict) -> str:
        """Generate response using Vertex AI for database-specific queries"""
        
        if not self.model or not self.use_vertex_ai:
            logger.warning("Vertex AI not available for database query")
            return f"I have information about {location}, but I'm having trouble accessing the detailed analysis right now."
        
        # Build context string
        context_info = ""
        if context.get('recent_history'):
            recent_queries = [h['query'] for h in context['recent_history'][-2:]]
            if recent_queries:
                context_info = f"\nRecent conversation: {' | '.join(recent_queries)}"
        
        # Extract the summary and enhanced data
        summary = location_data.get('summary', '')
        total_reviews = location_data.get('total_reviews', 0)
        
        # Log that we're using Vertex AI
        logger.info(f"🤖 Using Vertex AI for database query about {location}")
        
        # Check if user wants detailed response
        detailed_keywords = ['detailed', 'detail', 'elaborate', 'explain more', 'tell me more', 'comprehensive', 'full information', 'complete guide']
        wants_detailed = any(keyword in query.lower() for keyword in detailed_keywords)
        
        if wants_detailed:
            prompt = f"""
You are a knowledgeable travel assistant with access to comprehensive location data from detailed reviews.

Location: {location}
Summary: {summary}
Reviews Analyzed: {total_reviews}
{context_info}

User Question: {query}

The user has requested detailed information. Provide a comprehensive, detailed response that:
1. Thoroughly answers the user's question using all relevant review data
2. Includes specific details, examples, and practical insights
3. Covers multiple aspects like timing, costs, facilities, tips, etc.
4. Is informative, engaging, and conversational
5. Provides actionable travel advice

Feel free to be comprehensive and detailed in your response.
"""
        else:
            prompt = f"""
You are a knowledgeable travel assistant with access to comprehensive location data from detailed reviews.

Location: {location}
Summary: {summary}
Reviews Analyzed: {total_reviews}
{context_info}

User Question: {query}

Provide a concise, helpful response in 100-150 words that:
1. Directly answers the user's question using the most relevant review data
2. Highlights the KEY points and important information in **bold**
3. Is informative but brief and to the point
4. Includes only the most essential practical tips
5. Uses a friendly, conversational tone
6. It should not miss out on any important details, but keep it concise
Keep it concise while covering the most important aspects. If the user wants more details, they can ask for a "detailed" explanation.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating Vertex AI response: {e}")
            return f"Based on {total_reviews} reviews, here's what I know about {location}:\n\n{summary}"

    def generate_general_response(self, query: str) -> str:
        """Generate response using Gemini API for general travel queries"""
        
        if not self.gemini_model:
            return "I'm specialized in providing information about specific tourist locations. Could you ask about a particular place like Red Fort, India Gate, or other attractions?"
        
        # Log that we're using Gemini API
        logger.info(f"🌐 Using Gemini API for general travel query")
        
        # Check if user wants detailed response
        detailed_keywords = ['detailed', 'detail', 'elaborate', 'explain more', 'tell me more', 'comprehensive', 'full guide', 'complete information']
        wants_detailed = any(keyword in query.lower() for keyword in detailed_keywords)
        
        if wants_detailed:
            prompt = f"""
You are a helpful travel assistant for travelers interested in India and general travel advice.

User Question: {query}

The user has requested detailed information. Provide a comprehensive, detailed response about general travel topics that:
- Covers multiple aspects of the topic thoroughly
- Includes practical examples and specific advice
- Provides actionable steps and recommendations
- Covers cultural context, safety, and practical considerations
- Is engaging and conversational

Feel free to be comprehensive and detailed in your response.
"""
        else:
            prompt = f"""
You are a helpful travel assistant for travelers interested in India and general travel advice.

User Question: {query}

Provide a concise, helpful response in 100-150 words that:
- Directly answers the question with the most important information
- Highlights KEY points in **bold**
- Focuses on essential practical advice
- Is brief but informative
- Uses a friendly, conversational tone

Keep it concise while covering the most crucial points. If the user wants more details, they can ask for a "detailed" explanation.
"""
    
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating Gemini API response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try asking about specific tourist locations or try again in a moment."
    
    def _is_general_travel_query(self, query: str) -> bool:
        """Determine if query is a general travel question outside our database"""
        query_lower = query.lower().strip()
        
        # First, check if any location from our database is mentioned
        for location in self.summaries.keys():
            if location.lower() in query_lower:
                return False  # It's database-specific
        
        # Check for location variations
        location_variations = ['red fort', 'india gate', 'lotus temple', 'qutub minar', 
                              'colva beach', 'basilica', 'fort aguada', 'akshardham']
        for variation in location_variations:
            if variation in query_lower:
                return False  # It's database-specific
        
        # General travel patterns
        general_patterns = [
            'travel tips for india', 'how to travel in india', 'travel advice',
            'visa for india', 'currency in india', 'language spoken in india',
            'festivals in india', 'religions in india', 
            'what to pack for india', 'travel budget for india',
            'travel insurance', 'flight booking', 'hotel booking',
            'train travel in india', 'bus travel in india',
            'safety tips for india', 'local customs in india', 
            'etiquette in india', 'communication in india',
            'planning a trip to india', 'india travel itinerary',
            'best time to visit india', 'weather patterns in india',
            'indian food culture', 'shopping tips for india'
        ]
        
        # General question patterns
        general_question_patterns = [
            'how to plan', 'what should i pack', 'how much money',
            'is it safe to', 'what language', 'what currency',
            'best way to travel', 'how to get visa', 'travel insurance',
            'cultural differences', 'local customs', 'food safety'
        ]
        
        # Check for general patterns
        for pattern in general_patterns:
            if pattern in query_lower:
                return True
        
        # Check for general question patterns
        for pattern in general_question_patterns:
            if pattern in query_lower:
                return True
        
        # If contains generic travel words but no specific location
        generic_travel_words = ['planning a trip', 'travel advice', 'travel tips', 
                               'how to travel', 'travel guide', 'backpacking']
        if any(word in query_lower for word in generic_travel_words):
            return True
        
        return False
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general queries without specific location"""
        if not self.summaries:
            return "I'm sorry, but I'm having trouble accessing the travel database right now. Please try again later."
        
        locations_list = "\n".join([f"- {loc}" for loc in list(self.summaries.keys())[:20]])
        
        return f"""I can help you with detailed information about these locations:

{locations_list}

...and many more places in Delhi, Goa, Kolkata, and Bangalore!

Please ask me about a specific location, like:
- "Tell me about Red Fort"
- "What's the best time to visit Colva Beach?"
- "Is India Gate good for families?"

What would you like to explore?"""
    
    def query(self, user_query: str, session_id: str = None) -> Dict[str, Any]:
        """Process user query with proper AI backend selection"""
        try:
            # Handle session
            session_id = self.context_manager.get_or_create_session(session_id)
            context = self.context_manager.get_context(session_id)
            
            # Find relevant location
            location = self.find_relevant_location(user_query, context)
            
            if location:
                # DATABASE QUERY - Use Vertex AI
                logger.info(f"🏛️ Database query detected for: {location}")
                
                location_data = self.get_enhanced_summary(location)
                if not location_data:
                    response = f"I found a reference to {location}, but I don't have detailed information about it in my database."
                    ai_backend = "Database Lookup"
                else:
                    response = self.generate_database_response(user_query, location, location_data, context)
                    ai_backend = "Vertex AI" if self.use_vertex_ai else "Database Only"
                
                self.context_manager.add_interaction(session_id, user_query, response, location)
                
                return {
                    "response": response,
                    "session_id": session_id,
                    "location": location,
                    "ai_backend": ai_backend,
                    "query_type": "database"
                }
            
            elif self._is_general_travel_query(user_query):
                # GENERAL QUERY - Use Gemini API
                logger.info("🌍 General travel query detected")
                
                response = self.generate_general_response(user_query)
                ai_backend = "Gemini API" if self.gemini_model else "General Guidance"
                
                self.context_manager.add_interaction(session_id, user_query, response)
                
                return {
                    "response": response,
                    "session_id": session_id,
                    "location": None,
                    "ai_backend": ai_backend,
                    "query_type": "general"
                }
            
            else:
                # GUIDANCE - Show available options
                response = self._handle_general_query(user_query)
                
                self.context_manager.add_interaction(session_id, user_query, response)
                
                return {
                    "response": response,
                    "session_id": session_id,
                    "location": None,
                    "ai_backend": "Guidance",
                    "query_type": "guidance"
                }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I'm sorry, I encountered an error processing your request. Please try again.",
                "session_id": session_id or self.context_manager.get_or_create_session()
            }

# Initialize the RAG system
try:
    # Get configuration
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    vertex_location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
    
    # Determine JSON path
    json_path = os.path.join(os.path.dirname(__file__), "data", "enhanced_place_summaries.json")
    
    # Initialize with both options
    rag_system = EnhancedRAG(
        json_path=json_path,
        gemini_api_key=gemini_api_key,
        project_id=project_id,
        location=vertex_location
    )
    
    if rag_system.summaries:
        logger.info(f"✅ RAG system initialized successfully")
        logger.info(f"📊 Data: {len(rag_system.summaries)} locations loaded")
        vertex_status = "✅" if rag_system.use_vertex_ai else "❌"
        gemini_status = "✅" if rag_system.gemini_model else "❌"
        logger.info(f"🤖 Vertex AI: {vertex_status} | Gemini API: {gemini_status}")
    else:
        logger.error("❌ RAG system initialized but no data loaded")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG system: {e}")
    rag_system = None

# API Endpoints
@app.get("/")
async def root():
    status = "active" if rag_system and rag_system.summaries else "error"
    locations_count = len(rag_system.summaries) if rag_system else 0
    vertex_status = "Available" if rag_system and rag_system.use_vertex_ai else "Unavailable"
    gemini_status = "Available" if rag_system and rag_system.gemini_model else "Unavailable"
    
    return {
        "message": "Enhanced Travel Assistant RAG API",
        "version": "3.0.0",
        "status": status,
        "locations_available": locations_count,
        "ai_backends": {
            "vertex_ai": vertex_status,
            "gemini_api": gemini_status
        },
        "features": ["Smart Fallback", "Dual AI Support", "Session Management", "Category-based Keywords"],
        "data_loaded": bool(rag_system and rag_system.summaries),
        "deployment": "Google Cloud Run",
        "region": "asia-south1"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check showing both AI models"""
    if not rag_system:
        return {"status": "error", "message": "RAG system not initialized"}
    
    # Cleanup old sessions
    rag_system.context_manager.cleanup_old_sessions()
    
    vertex_status = "available" if (rag_system.model and rag_system.use_vertex_ai) else "unavailable"
    gemini_status = "available" if rag_system.gemini_model else "unavailable"
    
    return {
        "status": "healthy" if rag_system.summaries else "degraded",
        "version": "3.0.0",
        "ai_backends": {
            "vertex_ai": vertex_status,
            "gemini_api": gemini_status
        },
        "primary_backend": "Vertex AI" if rag_system.use_vertex_ai else "Gemini API",
        "locations_available": len(rag_system.summaries),
        "active_sessions": len(rag_system.context_manager.sessions),
        "data_loaded": bool(rag_system.summaries)
    }

@app.get("/locations")
async def get_locations():
    return {
        "locations": list(rag_system.summaries.keys()),
        "total_count": len(rag_system.summaries)
    }

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Enhanced query endpoint with session support"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not rag_system.summaries:
        raise HTTPException(status_code=500, detail="Location data not loaded")
    
    try:
        result = rag_system.query(request.query, request.session_id)
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/location/{location_name}")
async def get_location_info(location_name: str):
    location_data = rag_system.get_enhanced_summary(location_name)
    if not location_data:
        raise HTTPException(status_code=404, detail="Location not found")
    
    return {
        "location": location_name,
        "data": location_data
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to see what's failing"""
    debug_info = {
        "environment_vars": {
            "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT"),
            "VERTEX_AI_LOCATION": os.getenv("VERTEX_AI_LOCATION"),
            "GEMINI_API_KEY": "***" if os.getenv("GEMINI_API_KEY") else None
        },
        "imports": {
            "GENAI_AVAILABLE": GENAI_AVAILABLE,
            "VERTEX_AI_AVAILABLE": VERTEX_AI_AVAILABLE
        },
        "rag_system": {
            "use_vertex_ai": rag_system.use_vertex_ai if rag_system else None,
            "model_exists": bool(rag_system.model) if rag_system else None,
            "gemini_model_exists": bool(rag_system.gemini_model) if rag_system else None
        }
    }
    
    # Test Vertex AI specifically
    if rag_system:
        try:
            vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("VERTEX_AI_LOCATION", "us-central1"))
            test_model = GenerativeModel('gemini-2.0-flash')
            test_response = test_model.generate_content("Hello")
            debug_info["vertex_ai_test"] = "SUCCESS"
        except Exception as e:
            debug_info["vertex_ai_test"] = f"FAILED: {str(e)}"
        
        # Test Gemini API
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            test_model = genai.GenerativeModel('gemini-2.0-flash')
            test_response = test_model.generate_content("Hello")
            debug_info["gemini_api_test"] = "SUCCESS"
        except Exception as e:
            debug_info["gemini_api_test"] = f"FAILED: {str(e)}"
    
    return debug_info

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)