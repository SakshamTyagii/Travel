import json
import os
import re
from fuzzywuzzy import fuzz
import jellyfish

# Import the functions you've already created in search.py
# If import doesn't work, we'll recreate them here
try:
    from search_function.search import generate_keyboard_variations, add_transliteration_variations
except ImportError:
    # Recreate the functions if import fails
    def get_keyboard_neighbors():
        """Return a dictionary of neighboring keys on a QWERTY keyboard"""
        return {
            'a': ['q', 'w', 's', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'e': ['w', 's', 'd', 'r'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'i': ['u', 'j', 'k', 'o'],
            'j': ['h', 'u', 'i', 'k', 'm', 'n'],
            'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p', ';'],
            'm': ['n', 'j', 'k', ','],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'k', 'l', 'p'],
            'p': ['o', 'l', '[', ';'],
            'q': ['1', '2', 'w', 'a'],
            'r': ['e', 'd', 'f', 't'],
            's': ['a', 'w', 'e', 'd', 'x', 'z'],
            't': ['r', 'f', 'g', 'y'],
            'u': ['y', 'h', 'j', 'i'],
            'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'a', 's', 'e'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'g', 'h', 'u'],
            'z': ['a', 's', 'x'],
            '0': ['9', '-', '='],
            '1': ['`', 'q', '2'],
            '2': ['1', 'q', 'w', '3'],
            '3': ['2', 'w', 'e', '4'],
            '4': ['3', 'e', 'r', '5'],
            '5': ['4', 'r', 't', '6'],
            '6': ['5', 't', 'y', '7'],
            '7': ['6', 'y', 'u', '8'],
            '8': ['7', 'u', 'i', '9'],
            '9': ['8', 'i', 'o', '0'],
            ' ': [',', '.', '/']
        }

    def generate_keyboard_variations(word):
        """Generate variations of a word with possible keyboard typos"""
        if not word or len(word) < 3:  # Too short for variations
            return set()
            
        neighbors = get_keyboard_neighbors()
        variations = set()
        
        # Single character typos
        for i in range(len(word)):
            # Character replacement (pressing a neighboring key)
            char = word[i]
            if char in neighbors:
                for neighbor in neighbors[char]:
                    variations.add(word[:i] + neighbor + word[i+1:])
            
            # Character omission (not pressing a key)
            if len(word) > 2:  # Ensure we don't create too-short strings
                variations.add(word[:i] + word[i+1:])
                
        # Only add a few variations to avoid explosion
        return list(variations)[:5]
    
    def add_transliteration_variations(word):
        """Add common transliterations for Indian words"""
        if not word or len(word) < 3:  # Too short for transliterations
            return set()
            
        variations = set()
        
        # Common transliteration patterns
        transliterations = {
            'a': ['aa', 'ah'],
            'aa': ['a', 'ah'],
            'ee': ['i', 'ea'],
            'i': ['ee', 'ea'],
            'u': ['oo'],
            'oo': ['u'],
            'v': ['w'],
            'w': ['v'],
            'sh': ['s'],
            's': ['sh'],
            'k': ['c', 'q'],
            'q': ['k'],
            'c': ['k', 's'],
            'z': ['j'],
            'j': ['z'],
            'ph': ['f'],
            'f': ['ph'],
            'ksh': ['x'],
            'th': ['t'],
            'dh': ['d'],
            'fort': ['qila', 'kila'],
            'temple': ['mandir'],
            'road': ['marg', 'path']
        }
        
        # Generate variations based on transliteration patterns
        for pattern, replacements in transliterations.items():
            if pattern in word:
                for replacement in replacements:
                    variations.add(word.replace(pattern, replacement))
        
        return list(variations)[:5]  # Limit to avoid explosion

# Function to automatically generate additional synonyms for a place
def generate_enhanced_synonyms(place):
    """Generate enhanced synonyms for a place using multiple methods"""
    name = place.get("name", "")
    category = place.get("category", "")
    area = place.get("area", "")
    
    # Start with existing synonyms or empty list
    existing_synonyms = place.get("synonyms", [])
    if existing_synonyms is None:
        existing_synonyms = []
    
    # Convert existing synonyms to a set for easier deduplication
    synonym_set = set(existing_synonyms)
    
    # Add common variations based on category
    if "fort" in name.lower():
        synonym_set.add(name.lower().replace("fort", "qila"))
        synonym_set.add(name.lower().replace("fort", "kila"))
    
    if "temple" in name.lower():
        synonym_set.add(name.lower().replace("temple", "mandir"))
    
    if "masjid" in name.lower():
        synonym_set.add(name.lower().replace("masjid", "mosque"))
    
    # Special cases for common monuments
    special_cases = {
        "red fort": ["lal kila", "lal qila"],
        "india gate": ["india geit", "indiaghat"],
        "qutub minar": ["kutub minar", "qutab minar", "kutab minar", "qutb minar"],
        "taj mahal": ["taj mehal", "tajmahal", "taj"],
        "jama masjid": ["jama mosque", "jumma masjid", "friday mosque"],
        "lotus temple": ["lotus mandir", "kamal mandir", "bahai temple"],
        "akshardham temple": ["akshardham mandir", "swaminarayan akshardham"],
        "gateway of india": ["india gate mumbai"],
        "golden temple": ["harmandir sahib", "darbar sahib"],
    }
    
    # Add special cases
    for case, variations in special_cases.items():
        if case in name.lower():
            for var in variations:
                synonym_set.add(var)
    
    # === NEW AIRPORT HANDLING ===
    # Special handling for airports and terminals
    airport_keywords = ["airport", "terminal", "international", "domestic", "airways", "t1", "t2", "t3"]
    is_airport = any(keyword in name.lower() for keyword in airport_keywords) or \
                 "airport" in category.lower() or \
                 "terminal" in category.lower() or \
                 any(f"t{i}" == name.lower() for i in range(1, 5)) or \
                 re.search(r'\bt\d+\b', name.lower()) is not None
    
    if is_airport:
        # Major airport mappings
        airport_codes = {
            "delhi": "DEL",
            "mumbai": "BOM",
            "bangalore": "BLR",
            "bengaluru": "BLR",
            "chennai": "MAA",
            "kolkata": "CCU",
            "hyderabad": "HYD",
            "goa": "GOI",
            "pune": "PNQ",
            "ahmedabad": "AMD",
            "kochi": "COK",
            "thiruvananthapuram": "TRV",
            "lucknow": "LKO",
        }
        
        airport_full_names = {
            "delhi": "Indira Gandhi International",
            "mumbai": "Chhatrapati Shivaji Maharaj International",
            "bangalore": "Kempegowda International",
            "bengaluru": "Kempegowda International",
            "chennai": "Chennai International",
            "kolkata": "Netaji Subhash Chandra Bose International",
            "hyderabad": "Rajiv Gandhi International",
            "goa": "Dabolim Airport",
            "pune": "Pune Airport",
            "ahmedabad": "Sardar Vallabhbhai Patel International",
            "kochi": "Cochin International",
            "thiruvananthapuram": "Trivandrum International",
        }
        
        airport_alt_names = {
            "indira gandhi": ["delhi airport", "igi", "delhi international"],
            "chhatrapati shivaji": ["mumbai airport", "csia", "bombay airport", "sahar airport", "santacruz"],
            "kempegowda": ["bengaluru airport", "bangalore airport"],
            "rajiv gandhi": ["hyderabad airport", "shamshabad airport"],
            "netaji subhash": ["kolkata airport", "calcutta airport", "dum dum airport"],
        }
        
        # Identify city
        city_found = None
        for city in airport_codes.keys():
            if city in name.lower() or city in area.lower():
                city_found = city
                break
        
        # Try to extract airport name for alternate matching
        airport_name_found = None
        for airport_name in airport_full_names.values():
            if airport_name.lower() in name.lower():
                airport_name_found = airport_name
                break
            
        # Try alternate names
        for alt_name, variations in airport_alt_names.items():
            if alt_name in name.lower():
                for var in variations:
                    synonym_set.add(var)
            
        # Extract terminal information - enhance to catch more formats
        terminal_match = re.search(r'terminal\s*(\d+)', name.lower())
        if not terminal_match:
            # Try to find format like T1, T2, etc.
            terminal_match = re.search(r't(\d+)', name.lower())
        
        terminal_num = terminal_match.group(1) if terminal_match else None
        
        # Create airport variations
        if city_found:
            # Add city-based variations
            synonym_set.add(f"{city_found} airport")
            synonym_set.add(f"{city_found} international")
            synonym_set.add(f"{city_found} international airport")
            
            # Add airport code
            if city_found in airport_codes:
                code = airport_codes[city_found]
                synonym_set.add(code)
                synonym_set.add(f"{code} airport")
                
                # Add terminal variations with code
                if terminal_num:
                    synonym_set.add(f"{code} t{terminal_num}")
                    synonym_set.add(f"{code} terminal {terminal_num}")
                
            # Add full name variations
            if city_found in airport_full_names:
                full_name = airport_full_names[city_found]
                synonym_set.add(full_name)
                synonym_set.add(f"{full_name} airport")
                
                # Split the full name for common short forms
                full_name_parts = full_name.split()
                if len(full_name_parts) >= 2:
                    synonym_set.add(full_name_parts[0])  # First word
                    synonym_set.add(' '.join(full_name_parts[:2]))  # First two words
                
                # Add terminal variations with full name
                if terminal_num:
                    synonym_set.add(f"{full_name} terminal {terminal_num}")
                    synonym_set.add(f"terminal {terminal_num} {full_name}")
                    
        # Generic terminal variations
        if "terminal" in name.lower() and terminal_num:
            synonym_set.add(f"t{terminal_num}")
            synonym_set.add(f"terminal {terminal_num}")
            
            # Try to extract what airport this terminal belongs to (if not explicit)
            if not city_found and area:
                synonym_set.add(f"{area} terminal {terminal_num}")
                synonym_set.add(f"{area} t{terminal_num}")
        
        # Handle Mumbai's Chhatrapati Shivaji specifically (most complex name)
        if "chhatrapati" in name.lower() or "shivaji" in name.lower() or city_found == "mumbai":
            # Add common variations people search for
            synonym_set.add("mumbai airport")
            synonym_set.add("csmia")
            synonym_set.add("csia")
            synonym_set.add("bom")
            synonym_set.add("bombay airport")
            synonym_set.add("sahar airport")
            
            # Add all variations with "shivaji" - this is what people often search for
            synonym_set.add("shivaji airport")
            synonym_set.add("shivaji maharaj airport")
            synonym_set.add("mumbai shivaji airport")
            synonym_set.add("mumbai shiv")
            synonym_set.add("shiv airport mumbai")
            
            # Add variations without full name
            synonym_set.add("chhatrapati shivaji airport")
            synonym_set.add("chhatrapati shivaji maharaj airport")
            synonym_set.add("shivaji maharaj airport")
            
            if terminal_num:
                synonym_set.add(f"mumbai t{terminal_num}")
                synonym_set.add(f"mumbai terminal {terminal_num}")
                synonym_set.add(f"bom t{terminal_num}")
                synonym_set.add(f"shivaji terminal {terminal_num}")
                synonym_set.add(f"shivaji t{terminal_num}")
                synonym_set.add(f"mumbai shivaji t{terminal_num}")
                synonym_set.add(f"shiv t{terminal_num}")
                
                # Add more specific combined forms for complex airport terminals
                synonym_set.add(f"airport terminal {terminal_num}")
                synonym_set.add(f"airport t{terminal_num}")
                
                # Handle parenthesized forms that users might type
                if city_found:
                    # Examples: "Terminal 2 (Mumbai)", "T2 (Chhatrapati Shivaji)"
                    synonym_set.add(f"terminal {terminal_num} ({city_found})")
                    synonym_set.add(f"t{terminal_num} ({city_found})")
                    
                    if city_found in airport_full_names:
                        full_name = airport_full_names[city_found]
                        # Examples: "Terminal 2 (Chhatrapati Shivaji)"
                        synonym_set.add(f"terminal {terminal_num} ({full_name})")
                        synonym_set.add(f"t{terminal_num} ({full_name})")
                        
                        # Handle the case that triggered the bug: "Airport Terminal 2 (Chhatrapati Shivaji)"
                        synonym_set.add(f"airport terminal {terminal_num} ({full_name})")
                        synonym_set.add(f"airport t{terminal_num} ({full_name})")
                
                # For Mumbai's complex case, add extra forms with both parts of the name
                if city_found == "mumbai":
                    synonym_set.add(f"chhatrapati shivaji terminal {terminal_num}")
                    synonym_set.add(f"terminal {terminal_num} chhatrapati shivaji")
                    synonym_set.add(f"maharaj terminal {terminal_num}")
                    # The exact form from the bug report
                    synonym_set.add(f"airport terminal {terminal_num} (chhatrapati shivaji maharaj international)")
    
        # === GENERIC ENHANCEMENTS FOR ALL COMPLEX ENTITIES ===
        # Apply similar patterns for all major airports and complex locations
        if city_found and city_found in airport_full_names:
            full_name = airport_full_names[city_found]
            full_name_parts = full_name.split()
            
            # For all cities, add similar patterns as we did for Mumbai
            if len(full_name_parts) >= 2:
                # First word of official name (like "Indira", "Chhatrapati", "Kempegowda")
                first_word = full_name_parts[0].lower()
                
                # Add variations with first word only (what people often search)
                synonym_set.add(f"{city_found} {first_word}")
                synonym_set.add(f"{first_word} airport")
                
                # Add first few letters as people often abbreviate
                if len(first_word) > 3:
                    abbrev = first_word[:3].lower()
                    synonym_set.add(f"{city_found} {abbrev}")
                    synonym_set.add(f"{abbrev} airport")
                
                # Add custom variations for each airport's common search patterns
                if city_found == "delhi":
                    synonym_set.add("igi")
                    synonym_set.add("delhi airport")
                    synonym_set.add("indira gandhi")
                    synonym_set.add("delhi international")
                elif city_found == "bangalore" or city_found == "bengaluru":
                    synonym_set.add("blr airport")
                    synonym_set.add("bengaluru international")
                    synonym_set.add("bangalore international")
                    synonym_set.add("kempe")
                elif city_found == "hyderabad":
                    synonym_set.add("hyd airport")
                    synonym_set.add("rajiv gandhi")
                    synonym_set.add("shamshabad")
                elif city_found == "chennai":
                    synonym_set.add("maa airport")
                    synonym_set.add("madras airport")
                elif city_found == "kolkata":
                    synonym_set.add("ccu airport")
                    synonym_set.add("calcutta airport")
                    synonym_set.add("netaji")
                    synonym_set.add("dum dum")
            
            # For terminals, add common search patterns for all airports
            if terminal_num:
                synonym_set.add(f"{city_found} terminal {terminal_num}")
                synonym_set.add(f"{city_found} t{terminal_num}")
                
                # With first word of airport name
                if len(full_name_parts) >= 1:
                    first_word = full_name_parts[0].lower()
                    synonym_set.add(f"{first_word} terminal {terminal_num}")
                    synonym_set.add(f"{first_word} t{terminal_num}")
                
                # With airport code
                if city_found in airport_codes:
                    code = airport_codes[city_found]
                    synonym_set.add(f"{code} terminal {terminal_num}")
                    synonym_set.add(f"{code} t{terminal_num}")
                
                # Common terminal search pattern variations
                synonym_set.add(f"terminal {terminal_num} {city_found}")
                synonym_set.add(f"t{terminal_num} {city_found}")
                
                # Handle parenthesized forms (generic version)
                synonym_set.add(f"terminal {terminal_num} ({city_found})")
                synonym_set.add(f"t{terminal_num} ({city_found})")
                synonym_set.add(f"terminal {terminal_num} ({full_name})")
                synonym_set.add(f"t{terminal_num} ({full_name})")
                synonym_set.add(f"{city_found} airport terminal {terminal_num}")
                
                # The exact pattern that caused the bug
                synonym_set.add(f"airport terminal {terminal_num} ({full_name})")
    
    # === GENERIC ENHANCEMENTS FOR ALL COMPLEX NAMES ===
    # Apply smart splitting for multi-part names (beyond just airports)
    complex_name_threshold = 3  # Names with this many words are considered complex
    words = name.split()
    
    if len(words) >= complex_name_threshold:
        # Add first word as many people search by it
        synonym_set.add(words[0].lower())
        
        # Add first two words
        if len(words) >= 2:
            synonym_set.add(f"{words[0]} {words[1]}".lower())
        
        # Add last word if it's a category word
        category_words = ["airport", "temple", "fort", "monument", "palace", "museum", "hospital", "station"]
        if words[-1].lower() in category_words:
            remaining = " ".join(words[:-1]).lower()
            synonym_set.add(remaining)
        
        # Add first and last word for long names
        if len(words) >= 4:
            synonym_set.add(f"{words[0]} {words[-1]}".lower())
    
    # Add name without spaces
    if " " in name:
        synonym_set.add(name.lower().replace(" ", ""))
    
    # Add abbreviations for multi-word names
    words = name.split()
    if len(words) > 1:
        abbr = ''.join([word[0].lower() for word in words if word and len(word) > 0])
        if len(abbr) > 1:
            synonym_set.add(abbr)
    
    # Add area + name combined
    if area:
        synonym_set.add(f"{area} {name}".lower())
    
    # Add keyboard variations
    name_lower = name.lower()
    words = name_lower.split()
    for word in words:
        if len(word) > 3:  # Only for longer words
            keyboard_vars = generate_keyboard_variations(word)
            for var in keyboard_vars:
                # Replace the word in the name
                synonym_set.add(name_lower.replace(word, var))
    
    # Add transliteration variations
    for word in words:
        if len(word) > 3:  # Only for longer words
            trans_vars = add_transliteration_variations(word)
            for var in trans_vars:
                # Replace the word in the name
                synonym_set.add(name_lower.replace(word, var))
    
    # Add phonetic variations based on Soundex
    soundex = jellyfish.soundex(name_lower)
    
    # For each word in the name, add soundex variant
    for word in words:
        if len(word) > 3:  # Only for longer words
            word_soundex = jellyfish.soundex(word)
            # We store the soundex code itself as a finding aid
            synonym_set.add(f"soundex:{word_soundex}")
    
    # Add location-aware combinations
    location_synonyms = add_location_aware_synonyms(place)
    for location_syn in location_synonyms:
        synonym_set.add(location_syn)
    
    # Add city name aliases
    add_city_name_aliases(place, synonym_set)
    
    # Filter out empty strings and the name itself
    result = [s for s in synonym_set if s and s.strip() and s.lower() != name.lower()]
    
    # Sort for consistency
    result.sort()
    
    # Increased limit for complex entities like airports and major landmarks
    if is_airport:
        return result[:40]  # Allow more synonyms for airports which have complex names
    elif "international" in name.lower():
        return result[:30]  # Allow more synonyms for other complex entities
    else:
        return result[:15]  # Standard limit for other places

def process_json_file(file_path):
    print(f"Processing {file_path}...")
    
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        places = json.load(f)
    
    print(f"Found {len(places)} places in the file.")
    
    # Process each place
    for i, place in enumerate(places):
        print(f"Processing place {i+1}/{len(places)}: {place.get('name', 'Unknown')}")
        
        # Generate enhanced synonyms
        enhanced_synonyms = generate_enhanced_synonyms(place)
        
        # Update the place
        place["synonyms"] = enhanced_synonyms
    
    # Write the updated JSON back to file
    output_path = file_path.replace('.json', '_enhanced.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(places, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Enhanced file saved to {output_path}")
    return output_path

def update_existing_enhanced_json(input_file):
    """Update the existing enhanced_places.json file with improved synonyms"""
    print(f"Updating synonyms in {input_file}...")
    
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        places = json.load(f)
    
    print(f"Found {len(places)} places in the file.")
    
    # Count of updated entries
    updated_count = 0
    
    # Process each place
    for i, place in enumerate(places):
        if i % 20 == 0:  # Status update every 20 items
            print(f"Processing place {i+1}/{len(places)}...")
            
        # Generate enhanced synonyms
        new_synonyms = generate_enhanced_synonyms(place)
        
        # Compare with existing synonyms
        existing_synonyms = set(place.get("synonyms", []))
        new_synonyms_set = set(new_synonyms)
        
        if new_synonyms_set != existing_synonyms:
            # Update the place with improved synonyms
            place["synonyms"] = list(new_synonyms_set)
            updated_count += 1
    
    # Write back to the same file
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(places, f, indent=2, ensure_ascii=False)
    
    print(f"Updated synonyms for {updated_count} places")
    print(f"Changes saved to {input_file}")
    return input_file

# Add this function after the generate_enhanced_synonyms function definition

def add_location_aware_synonyms(place):
    """Generate location-aware combinations for improved search"""
    name = place.get("name", "").lower()
    area = place.get("area", "").lower()
    category = place.get("category", "").lower()
    
    result = set()
    
    if not name or not area:
        return result
    
    # Skip if area is already in the name
    if area.lower() in name.lower():
        return result
    
    # Basic combinations
    result.add(f"{name} {area}")  # "Qutub Minar Delhi"
    result.add(f"{area} {name}")  # "Delhi Qutub Minar"
    
    # Add standalone area as it helps in many search contexts
    result.add(area)  # "Delhi"
    
    # Handle multi-word landmarks
    words = name.split()
    if len(words) >= 2:
        # Add first word + area
        result.add(f"{words[0]} {area}")  # "Qutub Delhi"
        
        # Add landmark words + area (based on common landmark terms)
        landmark_terms = ["fort", "temple", "mandir", "masjid", "mosque", "beach", 
                         "market", "minar", "palace", "gate", "lake", "airport", 
                         "station", "tomb", "museum", "garden", "park"]
        
        for word in words:
            if word.lower() in landmark_terms:
                result.add(f"{word} {area}")  # "Minar Delhi"
                result.add(f"{area} {word}")  # "Delhi Minar"
    
    # Extract main category words
    category_words = []
    if category:
        category_words = [word for word in category.lower().split() if len(word) > 3]
    
    # Add category + area combinations
    for cat_word in category_words:
        result.add(f"{cat_word} {area}")  # "Beach Goa"
        result.add(f"{area} {cat_word}")  # "Goa Beach"
    
    # Handle common attraction prefixes/suffixes
    if "temple" in name.lower():
        result.add(f"{area} temple")  # "Delhi Temple"
    if "fort" in name.lower():
        result.add(f"{area} fort")    # "Delhi Fort"
    if "beach" in name.lower():
        result.add(f"{area} beach")   # "Goa Beach" 
    if "museum" in name.lower():
        result.add(f"{area} museum")  # "Delhi Museum"
        
    return result

# Add this to the bottom of your file's main section
if __name__ == "__main__":

    # Path to existing enhanced_places.json
    enhanced_json_path = "c:\\Users\\tyagi\\Desktop\\DelhiTravelAssistant\\search_function\\data\\enhanced_places.json"
    
    # Update the existing file
    update_existing_enhanced_json(enhanced_json_path)
    
    print("Success! Enhanced synonyms have been updated with location-aware combinations.")

# Add this function after the add_location_aware_synonyms function

def add_city_name_aliases(place, synonym_set):
    """Add alternate city names as synonyms (Chennai/Madras, Mumbai/Bombay, etc.)"""
    
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
    
    # Merge both mappings for easier lookup
    all_aliases = {**city_aliases, **reverse_aliases}
    
    # Get name, area, and all current synonyms from the place
    name = place.get("name", "").lower()
    area = place.get("area", "").lower()
    existing_synonyms = [s.lower() for s in place.get("synonyms", [])]
    
    # Check if any city name or its alias appears in the place data
    for city, aliases in all_aliases.items():
        # Check if city appears in name, area, or any existing synonym
        city_in_data = (city in name.lower() or 
                       city in area.lower() or 
                       any(city in syn.lower() for syn in existing_synonyms))
        
        if city_in_data:
            # Add all aliases for this city
            for alias in aliases:
                # Replace city with alias in various patterns
                if area.lower() == city:
                    synonym_set.add(alias)  # Add plain alias as location
                
                # Add "attraction alias" for "attraction city"
                for syn in list(synonym_set):
                    if city in syn and ' ' in syn:
                        new_syn = syn.replace(city, alias)
                        synonym_set.add(new_syn)
                
                # Add "alias attraction" patterns
                if area.lower() == city:
                    synonym_set.add(f"{alias} {name}")
                    synonym_set.add(f"{name} {alias}")
                    
                    # Handle multi-word name parts
                    words = name.split()
                    if len(words) >= 2:
                        # Add first/main word + alias
                        synonym_set.add(f"{words[0]} {alias}")
                        synonym_set.add(f"{alias} {words[0]}")
                        
                        # Handle landmark keywords
                        landmark_terms = ["fort", "temple", "beach", "museum", "park"]
                        for word in words:
                            if word.lower() in landmark_terms:
                                synonym_set.add(f"{word} {alias}")
                                synonym_set.add(f"{alias} {word}")