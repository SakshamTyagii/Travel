import pandas as pd
from review_processor import ReviewSummarizer
import argparse

def process_specific_location(location_name):
    """Process reviews for a specific location"""
    summarizer = ReviewSummarizer()
    reviews_df = pd.read_csv("../data/google_maps_reviews.csv")
    summarizer.process_location_reviews(location_name, reviews_df)

def list_available_locations():
    """List all available locations in the CSV"""
    try:
        reviews_df = pd.read_csv("../data/google_maps_reviews.csv")
        locations = reviews_df['place_name'].dropna().unique()
        print(f"Available locations ({len(locations)}):")
        for i, location in enumerate(sorted(locations), 1):
            review_count = len(reviews_df[reviews_df['place_name'] == location]['review_text'].dropna())
            print(f"{i}. {location} ({review_count} reviews)")
    except Exception as e:
        print(f"Error loading locations: {e}")

def show_progress():
    """Show processing progress for all locations"""
    try:
        summarizer = ReviewSummarizer()
        reviews_df = pd.read_csv("../data/google_maps_reviews.csv")
        locations = reviews_df['place_name'].dropna().unique()
        
        print(f"Processing Progress Report:")
        print("-" * 60)
        
        for location in sorted(locations):
            existing_summary, processed_count = summarizer.load_existing_summary(location)
            total_reviews = len(reviews_df[reviews_df['place_name'] == location]['review_text'].dropna())
            
            if processed_count == 0:
                status = "❌ Not Started"
            elif processed_count < total_reviews:
                status = f"🔄 In Progress"
            else:
                status = "✅ Completed"
                
            print(f"{status} {location}: {processed_count}/{total_reviews} reviews")
            
    except Exception as e:
        print(f"Error showing progress: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Review Summarizer')
    parser.add_argument('--location', type=str, help='Process specific location')
    parser.add_argument('--list', action='store_true', help='List all available locations')
    parser.add_argument('--all', action='store_true', help='Process all locations')
    parser.add_argument('--resume', action='store_true', help='Resume processing from where it left off')
    parser.add_argument('--progress', action='store_true', help='Show processing progress')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_locations()
    elif args.progress:
        show_progress()
    elif args.location:
        process_specific_location(args.location)
    elif args.resume:
        summarizer = ReviewSummarizer()
        summarizer.resume_processing()
    elif args.all:
        summarizer = ReviewSummarizer()
        summarizer.process_all_locations()
    else:
        print("Usage:")
        print("  python batch_processor.py --list                    # List locations")
        print("  python batch_processor.py --progress               # Show progress")
        print("  python batch_processor.py --location 'Place Name'   # Process specific location")
        print("  python batch_processor.py --resume                 # Resume from where left off")
        print("  python batch_processor.py --all                     # Process all locations")