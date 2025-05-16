import csv
import json
import os

# Create output directory if it doesn't exist
os.makedirs("search_function/data", exist_ok=True)

# Convert CSV to JSON
places = []
try:
    with open("search_function/data/appsheet.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            place = {
                "name": row.get("Location", ""),
                "category": row.get("Layer", ""),
                "area": row.get("City", ""),
                "description": f"A notable {row.get('Layer', 'location')} in {row.get('City', '')}.",
                "synonyms": row.get("synonyms", "").split(",") if row.get("synonyms") else []
            }
            places.append(place)
    
    # Write to JSON file
    with open("search_function/data/places.json", "w", encoding="utf-8") as jsonfile:
        json.dump(places, jsonfile, indent=2)
    
    print(f"Successfully converted {len(places)} places to JSON")
except Exception as e:
    print(f"Error: {str(e)}")