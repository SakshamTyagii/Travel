import json
import os
import logging
from dotenv import load_dotenv
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def enhance_places_data():
    """Enhance places data with relationships and additional attributes"""
    # Define the path to places.json
    places_json_path = "search_function/data/places.json"
    
    try:
        # Check if file exists
        if not os.path.exists(places_json_path):
            logger.error(f"File not found: {places_json_path}")
            return False
            
        # Load the existing places data
        with open(places_json_path, "r", encoding="utf-8") as f:
            places_data = json.load(f)
            
        logger.info(f"Loaded {len(places_data)} places from {places_json_path}")
        
        # Enhance with relationships and additional attributes
        for place in places_data:
            # Add canonical form for name variants
            if "synonyms" in place:
                place["canonical_name"] = place.get("name", place.get("Location", ""))
                
            # Add related places (nearby, part-of relationships)
            place["related_places"] = []
            
            # For places in the same area, add as related
            area = place.get("area", place.get("City", ""))
            if area:
                for other_place in places_data:
                    other_area = other_place.get("area", other_place.get("City", ""))
                    other_name = other_place.get("name", other_place.get("Location", ""))
                    current_name = place.get("name", place.get("Location", ""))
                    
                    if other_name != current_name and other_area == area:
                        place["related_places"].append({
                            "name": other_name,
                            "relationship": "same_area"
                        })
            
            # If coordinates are not available, try to geocode
            if "coordinates" not in place and "Location" in place:
                from search_function.search import geocode
                
                location_name = place.get("Location", "")
                city = place.get("City", "")
                if location_name and city:
                    address = f"{location_name}, {city}, India"
                    logger.info(f"Geocoding: {address}")
                    
                    try:
                        coords = geocode(address)
                        if coords:
                            place["coordinates"] = {
                                "lat": coords["lat"],
                                "lng": coords["lng"]
                            }
                            logger.info(f"Added coordinates for {location_name}")
                    except Exception as e:
                        logger.error(f"Error geocoding {address}: {str(e)}")
        
        # Save the enhanced data
        enhanced_places_path = "search_function/data/enhanced_places.json"
        with open(enhanced_places_path, "w", encoding="utf-8") as f:
            json.dump(places_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved enhanced data to {enhanced_places_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error enhancing places data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    enhance_places_data()