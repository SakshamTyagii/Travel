import csv
import json
import os

def convert_csv_to_json(csv_path, json_path):
    """Convert CSV file to JSON format"""
    places = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # Skip header
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    place_name = row[0].strip('"()"')
                    summary = row[1]
                    places.append({
                        "name": place_name,
                        "summary": summary
                    })
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        # Write to JSON file
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(places, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted {len(places)} places to JSON at {json_path}")
        return True
    except Exception as e:
        print(f"Error converting CSV to JSON: {str(e)}")
        return False

if __name__ == "__main__":
    # When run directly, use relative paths from the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    csv_path = os.path.join(project_dir, "ragpipeline", "review_summaries.csv")
    json_path = os.path.join(project_dir, "ragpipeline", "data", "place_summaries.json")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    convert_csv_to_json(csv_path, json_path)