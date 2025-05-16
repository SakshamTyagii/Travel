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
    
    # Filter out empty strings and the name itself
    result = [s for s in synonym_set if s and s.strip() and s.lower() != name.lower()]
    
    # Sort for consistency
    result.sort()
    
    # Limit to avoid too many synonyms (adjust as needed)
    return result[:15]  # Limit to 15 synonyms per place

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

if __name__ == "__main__":
    # Look for places.json in different possible locations
    possible_paths = [
        "data/places.json",  # Relative to current directory
        "./data/places.json",  # Explicit relative path
        os.path.join(os.path.dirname(__file__), "data", "places.json"),  # Next to script
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "search_function", "data", "places.json"),  # Parent dir
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "places.json"),  # Parent dir alt
        "c:\\Users\\tyagi\\Desktop\\DelhiTravelAssistant\\search_function\\data\\places.json",  # Absolute path
    ]
    
    places_json_path = None
    for path in possible_paths:
        if os.path.exists(path):
            places_json_path = path
            print(f"Found places.json at: {path}")
            break
    
    if not places_json_path:
        print("Error: Could not find places.json in any expected location.")
        print("Please enter the full path to places.json:")
        user_path = input("> ")
        if os.path.exists(user_path):
            places_json_path = user_path
        else:
            print(f"Error: File not found at {user_path}")
            exit(1)
    
    # Process the file
    output_file = process_json_file(places_json_path)
    
    print(f"Success! Enhanced synonyms have been added to {output_file}")
    print("You can now replace your original places.json with this file or review it first.")
    
    # Show a sample of the changes
    with open(output_file, 'r', encoding='utf-8') as f:
        places = json.load(f)
        
    print("\nSample of enhanced synonyms:")
    for i, place in enumerate(places[:3]):  # Show first 3 places
        print(f"\n{place.get('name', 'Unknown')}:")
        for synonym in place.get('synonyms', []):
            print(f"  - {synonym}")
        if i >= 2:
            break