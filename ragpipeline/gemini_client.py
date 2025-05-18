import google.generativeai as genai
import os
import logging
from typing import Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client with API key"""
        # Try to get API key from environment variable if not provided
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("No Gemini API key provided. Please set GEMINI_API_KEY environment variable.")
            self.model = None
            return
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Get default model
        try:
            # Use Gemini 2.0 Flash as specified
            model_name = "gemini-2.0-flash"
            
            try:
                self.model = genai.GenerativeModel(model_name=model_name)
                logger.info(f"Gemini client initialized successfully with model: {model_name}")
            except Exception as model_error:
                logger.error(f"Could not initialize Gemini 2.0 Flash model: {str(model_error)}")
                self.model = None
                
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            self.model = None
    
    def answer_query(self, place_data: Dict, user_query: str) -> str:
        """
        Generate an answer to the user query based on the place data
        Args:
            place_data: Dictionary with "name" and "summary" keys
            user_query: The user's question
        Returns:
            The AI-generated answer
        """
        if not self.model:
            return "Sorry, Gemini API is not configured properly. Please check your API key and available models."
        
        place_name = place_data.get("name", "Unknown")
        summary = place_data.get("summary", "")
        
        # Create a prompt that instructs Gemini how to answer
        prompt = f"""
        Based solely on the following information about {place_name}, please answer the user's query.
        Your response should be concise (under 100 words) and only use facts mentioned in the information provided.
        If the information doesn't contain the answer, say you don't have that specific information.

        INFORMATION ABOUT {place_name}:
        {summary}

        USER QUERY: {user_query}

        YOUR ANSWER:
        """
        
        try:
            logger.info(f"Sending query about {place_name} to Gemini")
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error getting response from Gemini: {str(e)}")
            return f"I have information about {place_name}, but I'm having trouble generating a specific response right now. Please try again in a moment."