import json
import csv
import pandas as pd

def convert_json_to_csv(json_file_path, csv_file_path):
    """Convert the enhanced place summaries JSON to CSV format"""
    
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Prepare data for CSV
    csv_data = []
    
    for location_name, location_data in data.items():
        row = {
            'location_name': location_name,
            'place_name': location_data.get('place_name', ''),
            'summary': location_data.get('summary', ''),
            'total_reviews': location_data.get('total_reviews', 0),
            'last_updated': location_data.get('last_updated', ''),
            'total_reviews_analyzed': location_data.get('enhanced_data', {}).get('total_reviews_analyzed', 0)
        }
        
        # Add enhanced summary categories
        enhanced_summary = location_data.get('enhanced_data', {}).get('summary', {})
        
        # Define all possible categories
        categories = [
            'Crowd Levels & Timing Suggestions',
            'Weather & Best Season to Visit',
            'Accessibility & Mobility',
            'Cleanliness & Maintenance',
            'Safety & Security',
            'Staff, Guides & Management',
            'Tickets, Entry & Pricing',
            'Signal Connectivity',
            'Suitability for specific audiences',
            'Historical or Cultural Significance',
            'Architecture & Aesthetics',
            'Spiritual or Religious Vibe',
            'Natural Beauty & Photogenic Spots',
            'Wildlife & Biodiversity',
            'Adventure & Activities',
            'Peace & Relaxation',
            'Shops & Stalls',
            'Food & Local Cuisine',
            'Events & Entertainment',
            'Navigation & Signage',
            'Amenities & Facilities',
            'Transport Connectivity',
            'Value for Time & Money',
            'Local Tips & Insider Advice',
            'Comparisons & Alternatives',
            'Emotional Tone / Vibe'
        ]
        
        # Add each category as a column
        for category in categories:
            # Join all items in the category with " | "
            category_items = enhanced_summary.get(category, [])
            row[category.replace(' & ', '_').replace(' / ', '_').replace(' ', '_').lower()] = ' | '.join(category_items)
        
        csv_data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    
    print(f"✅ CSV file created successfully: {csv_file_path}")
    print(f"📊 Total locations: {len(csv_data)}")
    print(f"📋 Total columns: {len(df.columns)}")
    
    return df

# Convert your JSON to CSV
df = convert_json_to_csv('enhanced_place_summaries.json', 'enhanced_place_summaries.csv')

# Display first few rows to verify
print("\n🔍 First 3 rows preview:")
print(df[['location_name', 'total_reviews', 'crowd_levels_timing_suggestions']].head(3))