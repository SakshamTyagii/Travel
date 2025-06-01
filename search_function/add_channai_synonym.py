import json
import os

def add_channai_synonym():
    """Add 'channai' as a synonym to any place that has 'chennai' in its synonyms"""
    
    # Path to enhanced places JSON
    json_path = "c:\\Users\\tyagi\\Desktop\\DelhiTravelAssistant\\search_function\\data\\enhanced_places.json"
    
    print(f"Processing {json_path}...")
    
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        places = json.load(f)
    
    print(f"Loaded {len(places)} places.")
    
    # Track modifications
    updated_count = 0
    
    # Process each place
    for place in places:
        # Skip empty entries (I noticed many in your file)
        if not place or "synonyms" not in place or not place["synonyms"]:
            continue
            
        # Get current synonyms
        synonyms = place.get("synonyms", [])
        
        # Check if any synonym contains "chennai"
        has_chennai = any("chennai" in syn.lower() for syn in synonyms)
        
        if has_chennai:
            # Create new synonyms with "channai" variant
            new_chennai_synonyms = []
            
            # Check each existing synonym
            for syn in synonyms:
                if "chennai" in syn.lower():
                    # Create new synonym with "channai" replacing "chennai"
                    new_syn = syn.lower().replace("chennai", "channai")
                    new_chennai_synonyms.append(new_syn)
            
            # Add "channai" as a standalone synonym if "chennai" exists
            if "chennai" in synonyms:
                new_chennai_synonyms.append("channai")
                
            # Add all new synonyms (avoid duplicates)
            for new_syn in new_chennai_synonyms:
                if new_syn not in synonyms:
                    synonyms.append(new_syn)
            
            # Update place with new synonyms
            place["synonyms"] = synonyms
            updated_count += 1
    
    # Write back to file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(places, f, indent=2, ensure_ascii=False)
    
    print(f"Updated {updated_count} places with 'channai' as a synonym.")
    print(f"Changes saved to {json_path}")

if __name__ == "__main__":
    add_channai_synonym()