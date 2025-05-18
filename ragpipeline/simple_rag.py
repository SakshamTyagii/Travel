import re
import logging
import csv
import json
from typing import List, Dict, Tuple, Optional
import os
from .gemini_client import GeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAG:
    def __init__(self, csv_path=None, json_path=None, gemini_api_key=None):
        """Initialize with path to summaries (either CSV or JSON)"""
        self.places_data = []
        self.place_names = []
        
        # Initialize Gemini client
        self.gemini_client = GeminiClient(api_key=gemini_api_key)
        
        # First try JSON if path provided
        if json_path and os.path.exists(json_path):
            self.load_from_json(json_path)
        # Otherwise try CSV
        elif csv_path and os.path.exists(csv_path):
            self.load_from_csv(csv_path)
        else:
            logger.error("No valid data source provided (neither CSV nor JSON path exists)")
            
        logger.info(f"Loaded {len(self.places_data)} place summaries")
    
    def load_from_json(self, json_path):
        """Load place data from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.places_data = json.load(f)
                self.place_names = [place["name"] for place in self.places_data]
            logger.info(f"Loaded {len(self.places_data)} places from JSON")
        except Exception as e:
            logger.error(f"Error loading JSON: {str(e)}")
            self.places_data = []
            self.place_names = []
            
    def load_from_csv(self, csv_path):
        """Load place data from CSV file"""
        try:
            self.places_data = []
            self.place_names = []
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.places_data.append({
                        "name": row["place_name"],
                        "summary": row["summary"]
                    })
                    self.place_names.append(row["place_name"])
                    
            logger.info(f"Loaded {len(self.places_data)} places from CSV")
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            
    def find_matching_place(self, query: str) -> Optional[Dict]:
        """Find the place that best matches the query"""
        query_lower = query.lower()
        
        # First try exact match
        for place in self.places_data:
            if place["name"].lower() in query_lower:
                return place
                
        # Try fuzzy match - find any place name word in query
        for place in self.places_data:
            name_words = place["name"].lower().split()
            for word in name_words:
                if len(word) > 3 and word in query_lower:  # Only match on words > 3 chars
                    return place
                    
        # No match found
        return None
        
    def generate_response(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Generate a response based on the query and matching place
        Returns: (response, matched_place_name)
        """
        matched_place = self.find_matching_place(query)
        
        if not matched_place:
            suggested_places = ", ".join(self.place_names[:5])
            return (f"I don't have specific information about that place in my database. "
                    f"I can answer questions about places like {suggested_places}, and more."), None
        
        place_name = matched_place["name"]
        logger.info(f"Found match: {place_name}")
        
        # Instead of extracting specific information ourselves,
        # send the entire place data and query to Gemini
        response = self.gemini_client.answer_query(matched_place, query)
        
        return response, place_name

    def _extract_time_info(self, summary: str) -> Optional[str]:
        """Extract visiting hours information from summary"""
        # Regex patterns for time info
        time_patterns = [
            r"visiting hours(?: are)? (\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm).+\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))",
            r"open(?:ing|s)?(?: hours| time)?(?: (?:from|is|are))? (\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm).+\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))",
            r"(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))\s*(?:to|-)\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))",
            r"hours?(?:\s*(?:are|is))?\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))\s*(?:to|-)\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))",
            r"closes at (\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))",
            r"open(?:s| daily)? (?:from )(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))"
        ]
        
        # Check patterns
        for pattern in time_patterns:
            match = re.search(pattern, summary, re.IGNORECASE)
            if match:
                # Find which sentence contains this match
                sentences = re.split(r'[.!?]', summary)
                for sentence in sentences:
                    if any(group in sentence for group in match.groups() if group):
                        return f"Regarding opening hours: {sentence.strip()}."
        
        # Keyword search if regex fails
        time_keywords = ["hour", "open", "close", "time", "am", "pm", "morning", "afternoon", "evening", "visit"]
        sentences = re.split(r'[.!?]', summary)
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in time_keywords):
                if any(char.isdigit() for char in sentence):
                    return f"Regarding opening hours: {sentence.strip()}."
        
        return None

    def _extract_price_info(self, summary: str) -> Optional[str]:
        """Extract price information from summary"""
        # Regex patterns for price info
        price_patterns = [
            r"(?:entry|admission|ticket) fee(?:s)? (?:is|are) (?:Rs\.?|₹|INR|\$)?\s?(\d+)",
            r"(?:costs?|priced at) (?:Rs\.?|₹|INR|\$)?\s?(\d+)",
            r"(?:Rs\.?|₹|INR|\$)\s?(\d+)(?:\s*(?:for|per)\s*(?:person|adult|entry|ticket))",
            r"fee of (?:Rs\.?|₹|INR|\$)?\s?(\d+)",
            r"entry is free"
        ]
        
        # Check patterns
        for pattern in price_patterns:
            match = re.search(pattern, summary, re.IGNORECASE)
            if match:
                # Find which sentence contains this match
                sentences = re.split(r'[.!?]', summary)
                for sentence in sentences:
                    if "free" in pattern and "free" in sentence.lower():
                        return f"Regarding pricing: {sentence.strip()}."
                    elif match.group(0) in sentence:
                        return f"Regarding pricing: {sentence.strip()}."
        
        # Keyword search if regex fails
        price_keywords = ["price", "cost", "fee", "ticket", "entry", "free", "Rs", "₹", "INR", "$", "expensive", "cheap"]
        sentences = re.split(r'[.!?]', summary)
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in price_keywords):
                if "free" in sentence.lower() or any(char.isdigit() for char in sentence):
                    return f"Regarding costs: {sentence.strip()}."
        
        return None

    def _extract_crowd_info(self, summary: str) -> Optional[str]:
        """Extract information about crowds and wait times"""
        # Keywords related to crowds
        crowd_keywords = ["crowd", "busy", "quiet", "wait", "line", "queue", "peak", "rush", "weekend", "weekday"]
        sentences = re.split(r'[.!?]', summary)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in crowd_keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            if len(relevant_sentences) > 2:
                return f"Regarding crowds: {' '.join(relevant_sentences[:2])}"
            return f"Regarding crowds: {' '.join(relevant_sentences)}"
        
        return None

    def _extract_food_info(self, summary: str) -> Optional[str]:
        """Extract information about food options"""
        # Keywords related to food
        food_keywords = ["food", "restaurant", "cafe", "eat", "dining", "snack", "meal", "cuisine", 
                         "drink", "breakfast", "lunch", "dinner", "shack", "vendor", "refreshment"]
        sentences = re.split(r'[.!?]', summary)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in food_keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            if len(relevant_sentences) > 2:
                return f"Regarding food options: {' '.join(relevant_sentences[:2])}"
            return f"Regarding food options: {' '.join(relevant_sentences)}"
        
        return None

    def _extract_activity_info(self, summary: str) -> Optional[str]:
        """Extract information about activities and things to do"""
        # Keywords related to activities
        activity_keywords = ["activity", "attraction", "see", "sport", "adventure", "photography", 
                             "swim", "tour", "explore", "enjoy", "water", "sport", "hike", "walk"]
        sentences = re.split(r'[.!?]', summary)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in activity_keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            if len(relevant_sentences) > 2:
                return f"Regarding activities: {' '.join(relevant_sentences[:2])}"
            return f"Regarding activities: {' '.join(relevant_sentences)}"
        
        return None
    
# Usage example
if __name__ == "__main__":
    rag = SimpleRAG("review_summaries.csv")
    response, place = rag.generate_response("What are the hours for Qutub Minar?")
    print(f"Place: {place}")
    print(f"Response: {response}")