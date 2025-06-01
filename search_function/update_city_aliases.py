import json
import os

def update_city_aliases():
    """Update enhanced_places.json with city name aliases"""
    # Path to enhanced places JSON
    json_path = "c:\\Users\\tyagi\\Desktop\\DelhiTravelAssistant\\search_function\\data\\enhanced_places.json"
    
    # Define mappings of city name aliases
    city_aliases = {
        "chennai": ["madras"],
        "mumbai": ["bombay"],
        "kolkata": ["calcutta"],
        "bengaluru": ["bangalore"],
        "kochi": ["cochin"],
        "thiruvananthapuram": ["trivandrum"],
        "varanasi": ["banaras", "benares", "kashi"],
        "pune": ["poona"],
        "shimla": ["simla"],
        "kodaikanal": ["kodai"],
        "puducherry": ["pondicherry"],
        # Add more as needed
    }
    
    # Create reverse mappings too (e.g., madras → chennai)
    reverse_aliases = {}
    for city, aliases in city_aliases.items():
        for alias in aliases:
            if alias not in reverse_aliases:
                reverse_aliases[alias] = []
            reverse_aliases[alias].append(city)
    
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        places = json.load(f)
    
    print(f"Loaded {len(places)} places from {json_path}")
    
    # Track updates
    updated_count = 0
    
    # Process each place
    for place in places:
        name = place.get("name", "").lower()
        area = place.get("area", "").lower()
        
        # Skip if no area defined
        if not area:
            continue
            
        # Get current synonyms
        synonyms = place.get("synonyms", [])
        if synonyms is None:
            synonyms = []
        
        # Convert to set for easy manipulation
        synonym_set = set(synonyms)
        original_size = len(synonym_set)
        
        # Check if this place's area has aliases
        area_lower = area.lower()
        
        # Check if area is a city with aliases
        if area_lower in city_aliases:
            # Add all aliases for this city
            for alias in city_aliases[area_lower]:
                # Add the alias itself
                synonym_set.add(alias)
                
                # Add "name alias" format
                synonym_set.add(f"{name.lower()} {alias}")
                
                # Add "alias name" format
                synonym_set.add(f"{alias} {name.lower()}")
                
                # Add aliases with substrings
                for syn in list(synonyms):
                    if area_lower in syn.lower():
                        new_syn = syn.lower().replace(area_lower, alias)
                        synonym_set.add(new_syn)
        
        # Check if area is an alias that has canonical cities
        if area_lower in reverse_aliases:
            for city in reverse_aliases[area_lower]:
                # Add the canonical city name
                synonym_set.add(city)
                
                # Add "name city" format
                synonym_set.add(f"{name.lower()} {city}")
                
                # Add "city name" format
                synonym_set.add(f"{city} {name.lower()}")
        
        # Check if area appears in any existing synonym
        for syn in list(synonyms):
            syn_lower = syn.lower()
            
            # Check for all cities and their aliases
            for city, aliases in city_aliases.items():
                if city in syn_lower:
                    # Replace with each alias
                    for alias in aliases:
                        new_syn = syn_lower.replace(city, alias)
                        synonym_set.add(new_syn)
            
            # Check reverse aliases too
            for alias, cities in reverse_aliases.items():
                if alias in syn_lower:
                    # Replace with each canonical city
                    for city in cities:
                        new_syn = syn_lower.replace(alias, city)
                        synonym_set.add(new_syn)
        
        # Update if we added new synonyms
        if len(synonym_set) > original_size:
            place["synonyms"] = sorted(list(synonym_set))
            updated_count += 1
    
    # Write back to file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(places, f, indent=2, ensure_ascii=False)
    
    print(f"Updated {updated_count} places with city aliases")
    print(f"Changes saved to {json_path}")

if __name__ == "__main__":
    update_city_aliases()