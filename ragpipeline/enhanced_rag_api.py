import os
import json
import logging
import uuid
import aiohttp
import asyncio
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

app = FastAPI(title="Enhanced Travel Assistant RAG API with Google Places", version="4.0.0")

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

class BusinessInfoRequest(BaseModel):
    place_id: str

# Google Places API Integration
class GooglePlacesService:
    def __init__(self):
        self.api_key = 'AIzaSyCjejP3da8MIqgZHnETJHUX-DuBRrA_MmM'
        self.base_url = 'https://maps.googleapis.com/maps/api/place'
    
    async def get_business_info(self, place_id: str) -> Dict[str, Any]:
        """Get comprehensive business info and real-time data"""
        fields = [
            'name', 'formatted_address', 'formatted_phone_number',
            'international_phone_number', 'website', 'opening_hours',
            'current_opening_hours', 'price_level', 'rating',
            'user_ratings_total', 'photos', 'geometry', 'types',
            'business_status', 'url'
        ]
        
        url = f"{self.base_url}/details/json"
        params = {
            'place_id': place_id,
            'fields': ','.join(fields),
            'key': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if data.get('status') == 'OK':
                        return self._format_business_data(data['result'])
                    else:
                        logger.error(f"Google Places API error: {data.get('status')}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching business info: {e}")
            return None
    
    def _format_business_data(self, place: Dict) -> Dict[str, Any]:
        """Format the business data for your application"""
        return {
            # Basic Info
            'name': place.get('name'),
            'address': place.get('formatted_address'),
            'location': {
                'lat': place.get('geometry', {}).get('location', {}).get('lat'),
                'lng': place.get('geometry', {}).get('location', {}).get('lng')
            },
            
            # Contact Information
            'contact': {
                'phone': place.get('formatted_phone_number'),
                'international_phone': place.get('international_phone_number'),
                'website': place.get('website'),
                'google_maps_url': place.get('url')
            },
            
            # Business Status & Hours
            'real_time_data': {
                'business_status': place.get('business_status'),
                'is_open_now': place.get('opening_hours', {}).get('open_now'),
                'current_hours': place.get('current_opening_hours'),
                'weekly_hours': place.get('opening_hours', {}).get('weekday_text'),
                'special_hours': place.get('opening_hours', {}).get('special_days')
            },
            
            # Pricing & Ratings
            'business_metrics': {
                'price_level': place.get('price_level'),
                'average_rating': place.get('rating'),
                'total_ratings': place.get('user_ratings_total'),
                'categories': place.get('types', [])
            },
            
            # High-Quality Photos
            'photos': self._get_photo_urls(place.get('photos', [])),
            
            # Metadata
            'last_updated': datetime.now().isoformat(),
            'place_id': place.get('place_id')
        }
    
    def _get_photo_urls(self, photos: List, max_width: int = 800) -> List[Dict]:
        """Generate high-quality photo URLs"""
        if not photos:
            return []
        
        return [{
            'url': f"https://maps.googleapis.com/maps/api/place/photo?maxwidth={max_width}&photoreference={photo['photo_reference']}&key={self.api_key}",
            'width': photo.get('width'),
            'height': photo.get('height'),
            'attributions': photo.get('html_attributions', [])
        } for photo in photos[:10]]  # Limit to 10 photos
    
    async def search_delhi_places(self, query: str, place_type: str = None) -> List[Dict]:
        """Search for places in Delhi"""
        location = '28.6139,77.2090'  # Delhi center
        radius = 10000  # 10km
        
        url = f"{self.base_url}/nearbysearch/json"
        params = {
            'location': location,
            'radius': radius,
            'key': self.api_key
        }
        
        if query:
            params['keyword'] = query
        if place_type:
            params['type'] = place_type
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if data.get('status') == 'OK':
                        return [{
                            'place_id': place['place_id'],
                            'name': place['name'],
                            'rating': place.get('rating'),
                            'total_ratings': place.get('user_ratings_total'),
                            'price_level': place.get('price_level'),
                            'is_open_now': place.get('opening_hours', {}).get('open_now'),
                            'location': place['geometry']['location'],
                            'types': place.get('types', [])
                        } for place in data.get('results', [])]
                    else:
                        logger.error(f"Places search error: {data.get('status')}")
                        return []
        except Exception as e:
            logger.error(f"Error searching places: {e}")
            return []

    async def find_place_by_name(self, place_name: str, location: str = "Delhi, India") -> Dict[str, Any]:
        """Find place ID by name using Text Search"""
        url = f"{self.base_url}/textsearch/json"
        params = {
            'query': f"{place_name} {location}",
            'key': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if data.get('status') == 'OK' and data.get('results'):
                        first_result = data['results'][0]
                        return {
                            'place_id': first_result['place_id'],
                            'name': first_result['name'],
                            'formatted_address': first_result['formatted_address'],
                            'rating': first_result.get('rating')
                        }
                    else:
                        return None
        except Exception as e:
            logger.error(f"Error finding place: {e}")
            return None

class DelhiBusinessManager:
    def __init__(self, rag_system):
        self.places_service = GooglePlacesService()
        self.rag_system = rag_system
        
        # This will be populated automatically from your database
        self.location_place_ids = {}
        self._place_ids_initialized = False
        self._initialization_lock = asyncio.Lock()
    
    async def _initialize_place_ids(self):
        """Automatically extract Place ID for all locations in your database"""
        if not self.rag_system or not self.rag_system.summaries:
            logger.warning("No RAG system or summaries available for Place ID initialization")
            return
        
        logger.info("🔄 Starting automatic Place ID extraction for all database locations...")
        
        # Get all location names from your database
        all_locations = list(self.rag_system.summaries.keys())
        logger.info(f"📍 Found {len(all_locations)} locations in database")
        
        successful_extractions = 0
        failed_extractions = 0
        
        for location_name in all_locations:
            try:
                # Try to find Place ID for this location
                place_data = await self.places_service.find_place_by_name(location_name)
                
                if place_data and place_data.get('place_id'):
                    # Store both the original name and common variations
                    place_id = place_data['place_id']
                    self.location_place_ids[location_name.lower()] = place_id
                    
                    # Add common variations
                    variations = self._generate_location_variations(location_name)
                    for variation in variations:
                        self.location_place_ids[variation.lower()] = place_id
                    
                    successful_extractions += 1
                    logger.info(f"✅ Found Place ID for: {location_name}")
                else:
                    failed_extractions += 1
                    logger.warning(f"❌ Could not find Place ID for: {location_name}")
                
                # Rate limiting to avoid hitting API limits
                await asyncio.sleep(0.1)  # 100ms delay between requests
                
            except Exception as e:
                failed_extractions += 1
                logger.error(f"❌ Error finding Place ID for {location_name}: {e}")
                continue
        
        logger.info(f"🎯 Place ID extraction complete: {successful_extractions} successful, {failed_extractions} failed")
        logger.info(f"📊 Total Place IDs mapped: {len(self.location_place_ids)}")
    
    def _generate_location_variations(self, location_name: str):
        """Generate common variations of location names"""
        variations = set()
        name_lower = location_name.lower()
        
        variations.add(name_lower)
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['the ', 'fort ', 'temple ', 'beach ', 'church ', 'palace ']
        suffixes_to_remove = [' fort', ' temple', ' beach', ' church', ' palace', ' museum']
        
        for prefix in prefixes_to_remove:
            if name_lower.startswith(prefix):
                variations.add(name_lower[len(prefix):])
        
        for suffix in suffixes_to_remove:
            if name_lower.endswith(suffix):
                variations.add(name_lower[:-len(suffix)])
        
        # Add variations with/without spaces and special characters
        variations.add(name_lower.replace(' ', ''))
        variations.add(name_lower.replace('-', ' '))
        variations.add(name_lower.replace("'", ''))
        
        return list(variations)
    
    def get_place_id_for_location(self, location_name: str) -> str:
        """Get Place ID for a location, with fallback search"""
        if not location_name:
            return None
        
        # Direct lookup
        place_id = self.location_place_ids.get(location_name.lower())
        if place_id:
            return place_id        
        # Fuzzy search through all stored variations
        for stored_name, stored_id in self.location_place_ids.items():
            if location_name.lower() in stored_name or stored_name in location_name.lower():
                return stored_id
        
        return None
    
    async def ensure_place_ids_initialized(self):
        """Ensure Place IDs are initialized before using them"""
        if self._place_ids_initialized:
            return
        
        async with self._initialization_lock:
            if self._place_ids_initialized:
                return
            await self._initialize_place_ids()
            self._place_ids_initialized = True
    
    async def get_complete_location_profile(self, location_name: str, place_id: str = None) -> Dict[str, Any]:
        """Get combined Google Places data + RAG reviews data"""
        try:
            # Ensure Place IDs are initialized
            await self.ensure_place_ids_initialized()
            
            # Get place_id automatically if not provided
            if not place_id:
                place_id = self.get_place_id_for_location(location_name)
                
                # If still no Place ID, try a live search
                if not place_id:
                    logger.info(f"🔍 No cached Place ID for {location_name}, searching live...")
                    place_data = await self.places_service.find_place_by_name(location_name)
                    if place_data and place_data.get('place_id'):
                        place_id = place_data['place_id']
                        # Cache it for future use
                        self.location_place_ids[location_name.lower()] = place_id
            
            # Get real-time business data from Google Places
            business_info = None
            if place_id:
                logger.info(f"🏢 Getting business info for {location_name} (Place ID: {place_id[:20]}...)")
                business_info = await self.places_service.get_business_info(place_id)
            else:
                logger.warning(f"⚠️ No Place ID found for {location_name}")
            
            # Get detailed reviews and analysis from RAG system
            rag_data = self.rag_system.get_enhanced_summary(location_name) if self.rag_system else None
            
            # Combine both datasets
            combined_profile = {
                'location_name': location_name,
                'place_id': place_id,
                'google_places_data': business_info,
                'review_analysis': rag_data,
                'data_sources': {
                    'real_time_info': 'Google Places API' if business_info else 'Not Available',
                    'reviews_analysis': 'Local RAG Database' if rag_data else 'Not Available'
                },
                'last_updated': datetime.now().isoformat()
            }
            
            return combined_profile
            
        except Exception as e:
            logger.error(f"Error creating combined profile for {location_name}: {e}")
            return {
                'location_name': location_name,
                'error': str(e),
                'google_places_data': None,
                'review_analysis': None
            }
    
    def _get_price_level_text(self, level: int) -> str:
        """Convert price level to text"""
        if level is None:
            return 'Unknown'
        levels = {
            0: 'Free',
            1: 'Inexpensive',
            2: 'Moderate',
            3: 'Expensive',
            4: 'Very Expensive'
        }
        return levels.get(level, 'Unknown')

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
        
        # Initialize Business Manager (Google Places integration)
        self.business_manager = DelhiBusinessManager(self)
        
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
    
    async def generate_enhanced_response(self, query: str, location: str, location_data: Dict, context: Dict) -> str:
        """Generate enhanced response combining RAG data + Google Places data"""
        
        # Always try to get business info for comprehensive responses
        combined_profile = await self.business_manager.get_complete_location_profile(location)
        
        # Use appropriate AI model
        if not self.model and not self.gemini_model:
            base_response = f"Based on {location_data.get('total_reviews', 0)} reviews, here's what I know about {location}:\n\n{location_data.get('summary', '')}"
            if combined_profile and combined_profile.get('google_places_data'):
                base_response += f"\n\n🏢 **Real-time Info**: Currently {'Open' if combined_profile['google_places_data'].get('real_time_data', {}).get('is_open_now') else 'Closed'}"
            return base_response
        
        # Build context string
        context_info = ""
        if context.get('recent_history'):
            recent_queries = [h['query'] for h in context['recent_history'][-2:]]
            if recent_queries:
                context_info = f"\nRecent conversation: {' | '.join(recent_queries)}"
        
        # Extract the summary and enhanced data
        summary = location_data.get('summary', '')
        total_reviews = location_data.get('total_reviews', 0)
        
        # Build enhanced prompt with both datasets
        business_data_text = ""
        photos_info = ""
        if combined_profile and combined_profile.get('google_places_data'):
            gp_data = combined_profile['google_places_data']
            
            # Format business info
            business_data_text = f"""
🏢 **REAL-TIME BUSINESS INFORMATION:**
• **Address**: {gp_data.get('address', 'Not available')}
• **Phone**: {gp_data.get('contact', {}).get('phone', 'Not available')}
• **Website**: {gp_data.get('contact', {}).get('website', 'Not available')}
• **Currently**: {'🟢 Open' if gp_data.get('real_time_data', {}).get('is_open_now') else '🔴 Closed'}
• **Google Rating**: {gp_data.get('business_metrics', {}).get('average_rating', 'N/A')}⭐ ({gp_data.get('business_metrics', {}).get('total_ratings', 'N/A')} reviews)
• **Price Level**: {self.business_manager._get_price_level_text(gp_data.get('business_metrics', {}).get('price_level'))}

📍 **Location**: {gp_data.get('location', {}).get('lat', 'N/A')}, {gp_data.get('location', {}).get('lng', 'N/A')}
"""
            
            # Add opening hours if available
            if gp_data.get('real_time_data', {}).get('weekly_hours'):
                hours = gp_data['real_time_data']['weekly_hours']
                business_data_text += f"\n🕐 **Opening Hours**:\n"
                for day_hours in hours[:3]:  # Show first 3 days
                    business_data_text += f"• {day_hours}\n"
                if len(hours) > 3:
                    business_data_text += f"• ... (and {len(hours)-3} more days)\n"
            
            # Add photos info
            photos = gp_data.get('photos', [])
            if photos:
                photos_info = f"\n📸 **HIGH-QUALITY IMAGES AVAILABLE**: {len(photos)} professional photos of {location}"
                # Add first few photo URLs for the AI to reference
                # if len(photos) >= 3:
                #     photos_info += f"\n🖼️ **Sample Images**:\n"
                #     for i, photo in enumerate(photos[:3]):
                #         photos_info += f"• Image {i+1}: {photo['url'][:50]}...\n"
                # Don't add any URLs to photos_info - just mention count
                # The frontend will fetch images separately via /location-complete endpoint
        
        # Check if user wants detailed response
        detailed_keywords = ['detailed', 'detail', 'elaborate', 'explain more', 'tell me more', 'comprehensive', 'full information', 'complete guide']
        wants_detailed = any(keyword in query.lower() for keyword in detailed_keywords)
        
        # Check if user is asking for images specifically
        image_keywords = ['photos', 'pictures', 'images', 'show me', 'look like', 'appearance', 'visual']
        wants_images = any(keyword in query.lower() for keyword in image_keywords)
        
        if wants_detailed:
            prompt = f"""
You are a knowledgeable travel assistant with access to comprehensive location data from detailed reviews AND real-time business information.

Location: {location}
Summary from Reviews: {summary}
Reviews Analyzed: {total_reviews}
{business_data_text}
{photos_info}
{context_info}

User Question: {query}

The user has requested detailed information. Provide a comprehensive, detailed response that:
1. Thoroughly answers the user's question using all available data (both review analysis AND real-time business info)
2. Includes specific details, examples, and practical insights
3. Covers multiple aspects like timing, costs, facilities, contact info, current status, etc.
4. Is informative, engaging, and conversational
5. Provides actionable travel advice
6. When relevant, mention real-time information like current operating status, contact details, etc.
7. If photos are available, mention that high-quality images are available but DO NOT include the actual URLs in your response

IMPORTANT: Do not include any image URLs in your response. Just mention that images are available.

Feel free to be comprehensive and detailed in your response.
"""
        else:
            prompt = f"""
You are a knowledgeable travel assistant with access to comprehensive location data from detailed reviews AND real-time business information.

Location: {location}
Summary from Reviews: {summary}
Reviews Analyzed: {total_reviews}
{business_data_text}
{photos_info}
{context_info}

User Question: {query}

Provide a helpful response in 150-200 words that:
1. Directly answers the user's question using the most relevant data (both reviews AND real-time business info)
2. Highlights the KEY points and important information in **bold**
3. Is informative but conversational
4. Includes practical tips and current information
5. Uses a friendly, engaging tone
6. When relevant, includes real-time information like current hours, contact details, etc.
7. If photos are available and relevant to the query, mention that high-quality images are available but DO NOT include the URLs

IMPORTANT: Do not include any image URLs in your response. Just mention that images are available.

Always include both review insights and current business information when available.
"""

        try:
            if self.model and self.use_vertex_ai:
                response = self.model.generate_content(prompt)
                generated_text = response.text
            elif self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                generated_text = response.text
            else:
                generated_text = f"Based on {total_reviews} reviews, here's what I know about {location}:\n\n{summary}"
                if business_data_text:
                    generated_text += f"\n\n{business_data_text}"
            
            # Add photos mention at the end if available and relevant (but NO URLs)
            if combined_profile and combined_profile.get('google_places_data', {}).get('photos') and (wants_images or 'photo' in query.lower()):
                photos = combined_profile['google_places_data']['photos']
                # Just mention that images are available - frontend will handle display
                generated_text += f"\n\n📸 **High-Quality Images Available**: {len(photos)} professional photos of {location}"
                # NO URLs added to text - frontend will fetch them separately
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating enhanced response: {e}")
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
    
    async def query(self, user_query: str, session_id: str = None) -> Dict[str, Any]:
        """Process user query with enhanced Google Places integration"""
        try:
            # Handle session
            session_id = self.context_manager.get_or_create_session(session_id)
            context = self.context_manager.get_context(session_id)
            
            # Find relevant location
            location = self.find_relevant_location(user_query, context)
            
            if location:
                # ENHANCED DATABASE QUERY - Use both RAG + Google Places
                logger.info(f"🏛️ Enhanced database query detected for: {location}")
                
                location_data = self.get_enhanced_summary(location)
                if not location_data:
                    response = f"I found a reference to {location}, but I don't have detailed information about it in my database."
                    ai_backend = "Database Lookup"
                else:
                    response = await self.generate_enhanced_response(user_query, location, location_data, context)
                    ai_backend = "Enhanced RAG + Google Places" if self.business_manager else "RAG Only"
                
                self.context_manager.add_interaction(session_id, user_query, response, location)
                
                return {
                    "response": response,
                    "session_id": session_id,
                    "location": location,
                    "ai_backend": ai_backend,
                    "query_type": "enhanced_database"
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

# END of EnhancedRAG class - all methods are now properly inside the class

# Initialize the enhanced RAG system
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
        logger.info(f"✅ Enhanced RAG system initialized successfully")
        logger.info(f"📊 Data: {len(rag_system.summaries)} locations loaded")
        vertex_status = "✅" if rag_system.use_vertex_ai else "❌"
        gemini_status = "✅" if rag_system.gemini_model else "❌"
        logger.info(f"🤖 Vertex AI: {vertex_status} | Gemini API: {gemini_status}")
        logger.info(f"🏢 Google Places API: ✅ Integrated")
    else:
        logger.error("❌ RAG system initialized but no data loaded")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG system: {e}")
    rag_system = None

# Place ID Management Endpoints
@app.get("/place-id-status")
async def get_place_id_status():
    """Check the status of automatic Place ID mapping"""
    if not rag_system or not rag_system.business_manager:
        raise HTTPException(status_code=500, detail="Business manager not initialized")
    
    total_locations = len(rag_system.summaries) if rag_system.summaries else 0
    mapped_locations = len(rag_system.business_manager.location_place_ids)
    
    # Get sample mappings
    sample_mappings = dict(list(rag_system.business_manager.location_place_ids.items())[:10])
    
    return {
        "total_database_locations": total_locations,
        "mapped_place_ids": mapped_locations,
        "mapping_percentage": round((mapped_locations / total_locations * 100), 2) if total_locations > 0 else 0,
        "sample_mappings": sample_mappings,
        "status": "Complete" if mapped_locations > 0 else "In Progress"
    }

@app.post("/refresh-place-ids")
async def refresh_place_ids():
    """Manually trigger Place ID refresh for all locations"""
    if not rag_system or not rag_system.business_manager:
        raise HTTPException(status_code=500, detail="Business manager not initialized")
    
    try:
        # Clear existing mappings
        rag_system.business_manager.location_place_ids.clear()
        rag_system.business_manager._place_ids_initialized = False
        
        # Reinitialize using the proper method
        await rag_system.business_manager.ensure_place_ids_initialized()
        
        return {
            "message": "Place ID refresh completed",
            "total_mapped": len(rag_system.business_manager.location_place_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing Place IDs: {str(e)}")

# Enhanced API Endpoints
@app.get("/")
async def root():
    status = "active" if rag_system and rag_system.summaries else "error"
    locations_count = len(rag_system.summaries) if rag_system else 0
    vertex_status = "Available" if rag_system and rag_system.use_vertex_ai else "Unavailable"
    gemini_status = "Available" if rag_system and rag_system.gemini_model else "Unavailable"
    
    return {
        "message": "Enhanced Travel Assistant RAG API with Google Places Integration",
        "version": "4.0.0",
        "status": status,
        "locations_available": locations_count,
        "ai_backends": {
            "vertex_ai": vertex_status,
            "gemini_api": gemini_status,
            "google_places_api": "Available"
        },
        "features": [
            "Smart Fallback", 
            "Dual AI Support", 
            "Session Management", 
            "Category-based Keywords",
            "Google Places Integration",
            "Real-time Business Data",
            "High-quality Photos",
            "Combined Data Sources"
        ],
        "data_loaded": bool(rag_system and rag_system.summaries),
        "deployment": "Google Cloud Run",
        "region": "asia-south1"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check showing all integrated systems"""
    if not rag_system:
        return {"status": "error", "message": "RAG system not initialized"}
    
    # Cleanup old sessions
    rag_system.context_manager.cleanup_old_sessions()
    
    vertex_status = "available" if (rag_system.model and rag_system.use_vertex_ai) else "unavailable"
    gemini_status = "available" if rag_system.gemini_model else "unavailable"
    places_status = "available" if rag_system.business_manager else "unavailable"
    
    return {
        "status": "healthy" if rag_system.summaries else "degraded",
        "version": "4.0.0",
        "ai_backends": {
            "vertex_ai": vertex_status,
            "gemini_api": gemini_status,
            "google_places_api": places_status
        },
        "primary_backend": "Enhanced RAG + Google Places",
        "locations_available": len(rag_system.summaries),
        "active_sessions": len(rag_system.context_manager.sessions),
        "data_loaded": bool(rag_system.summaries),
        "integrations": {
            "review_analysis": "Local RAG Database",
            "real_time_data": "Google Places API",
            "photos": "Google Places API",
            "business_info": "Google Places API"
        }
    }

@app.get("/locations")
async def get_locations():
    return {
        "locations": list(rag_system.summaries.keys()),
        "total_count": len(rag_system.summaries)
    }

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Enhanced query endpoint with Google Places integration"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not rag_system.summaries:
        raise HTTPException(status_code=500, detail="Location data not loaded")
    
    try:
        result = await rag_system.query(request.query, request.session_id)
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/location/{location_name}")
async def get_location_info(location_name: str):
    """Get comprehensive location info (RAG + Google Places)"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Get combined profile
        complete_profile = await rag_system.business_manager.get_complete_location_profile(location_name)
        
        if not complete_profile.get('review_analysis') and not complete_profile.get('google_places_data'):
            raise HTTPException(status_code=404, detail="Location not found in any data source")
        
        return {
            "location": location_name,
            "complete_profile": complete_profile
        }
    except Exception as e:
        logger.error(f"Error getting location info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving location data: {str(e)}")

@app.post("/business-info")
async def get_business_info(request: BusinessInfoRequest):
    """Get real-time business information from Google Places"""
    if not rag_system or not rag_system.business_manager:
        raise HTTPException(status_code=500, detail="Business manager not initialized")
    
    try:
        business_info = await rag_system.business_manager.places_service.get_business_info(request.place_id)
        
        if not business_info:
            raise HTTPException(status_code=404, detail="Business information not found")
        
        return {
            "place_id": request.place_id,
            "business_info": business_info
        }
    except Exception as e:
        logger.error(f"Error getting business info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving business data: {str(e)}")

@app.get("/search-places")
async def search_places(query: str, place_type: str = None):
    """Search for places using Google Places API"""
    if not rag_system or not rag_system.business_manager:
        raise HTTPException(status_code=500, detail="Business manager not initialized")
    
    try:
        results = await rag_system.business_manager.places_service.search_delhi_places(query, place_type)
        
        return {
            "query": query,
            "place_type": place_type,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching places: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching places: {str(e)}")

@app.get("/find-place")
async def find_place(place_name: str):
    """Find place ID by name - This helps you get real Place IDs"""
    if not rag_system or not rag_system.business_manager:
        raise HTTPException(status_code=500, detail="Business manager not initialized")
    
    try:
        result = await rag_system.business_manager.places_service.find_place_by_name(place_name)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Place '{place_name}' not found")
        
        return {
            "search_query": place_name,
            "result": result,
            "note": "Use the 'place_id' from this result in the /business-info endpoint"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/popular-places")
async def get_popular_places():
    """Get popular Delhi places with their Place IDs"""
    if not rag_system or not rag_system.business_manager:
        raise HTTPException(status_code=500, detail="Business manager not initialized")
    
    popular_places = []
    for name, place_id in rag_system.business_manager.delhi_attractions.items():
        popular_places.append({
            "name": name.replace('_', ' ').title(),
            "place_id": place_id
        })
    
    return {
        "popular_delhi_places": popular_places,
        "note": "Use any of these place_ids in the /business-info endpoint"
    }

@app.get("/debug")
async def debug_info():
    """Enhanced debug endpoint to see integration status"""
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
            "gemini_model_exists": bool(rag_system.gemini_model) if rag_system else None,
            "business_manager_exists": bool(rag_system.business_manager) if rag_system else None
        },
        "google_places_integration": {
            "api_key_configured": bool(rag_system.business_manager.places_service.api_key) if (rag_system and rag_system.business_manager) else False,
            "delhi_attractions_mapped": len(rag_system.business_manager.delhi_attractions) if (rag_system and rag_system.business_manager) else 0
        }
    }
    
    # Test Google Places API
    if rag_system and rag_system.business_manager:
        try:
            # Test search functionality
            test_results = await rag_system.business_manager.places_service.search_delhi_places("red fort")
            debug_info["google_places_test"] = f"SUCCESS - Found {len(test_results)} results"
        except Exception as e:
            debug_info["google_places_test"] = f"FAILED: {str(e)}"
    
    return debug_info

@app.get("/location-complete/{location_name}")
async def get_complete_location_with_images(location_name: str):
    """Get complete location info with images and all data combined"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Get combined profile
        complete_profile = await rag_system.business_manager.get_complete_location_profile(location_name)
        
        # Get RAG summary
        rag_summary = rag_system.get_enhanced_summary(location_name)
        
        # Combine everything
        result = {
            "location_name": location_name,
            "review_analysis": rag_summary,
            "real_time_business_data": complete_profile.get('google_places_data'),
            "combined_summary": None,
            "high_quality_images": [],
            "quick_facts": {}
        }
        
        # Add images if available
        # if complete_profile.get('google_places_data', {}).get('photos'):
        
        # Add images if available
        if complete_profile.get('google_places_data', {}).get('photos'):
            result["high_quality_images"] = complete_profile['google_places_data']['photos']
        
        # Add quick facts
        if complete_profile.get('google_places_data'):
            gp_data = complete_profile['google_places_data']
            result["quick_facts"] = {
                "currently_open": gp_data.get('real_time_data', {}).get('is_open_now'),
                "google_rating": gp_data.get('business_metrics', {}).get('average_rating'),
                "total_google_reviews": gp_data.get('business_metrics', {}).get('total_ratings'),
                "price_level": rag_system.business_manager._get_price_level_text(gp_data.get('business_metrics', {}).get('price_level')),
                "phone": gp_data.get('contact', {}).get('phone'),
                "website": gp_data.get('contact', {}).get('website'),
                "address": gp_data.get('address')
            }
        
        # Generate a combined summary using AI
        if rag_summary and complete_profile.get('google_places_data'):
            context = {"recent_history": []}
            combined_summary = await rag_system.generate_enhanced_response(
                f"Tell me about {location_name}", location_name, rag_summary, context
            )
            result["combined_summary"] = combined_summary
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting complete location info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving complete location data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)