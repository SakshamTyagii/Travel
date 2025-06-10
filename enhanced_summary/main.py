from review_processor import ReviewSummarizer
import sys

def main():
    print("Enhanced Review Summarization System")
    print("=" * 50)
    
    summarizer = ReviewSummarizer()
    
    # Check if user wants to resume
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        print("🔄 Resuming processing from where it left off...")
        summarizer.resume_processing("../data/google_maps_reviews.csv")
    else:
        print("🚀 Starting fresh processing...")
        # Process all locations from CSV
        summarizer.process_all_locations("../data/google_maps_reviews.csv")
    
    print("\n" + "=" * 50)
    print("Enhanced summarization completed!")
    print(f"Check the '{summarizer.output_dir}' folder for location summaries")

if __name__ == "__main__":
    main()