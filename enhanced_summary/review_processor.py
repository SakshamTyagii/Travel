import pandas as pd
import json
import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()

class ReviewSummarizer:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Define the 26 containers
        self.containers = [
            "Crowd Levels & Timing Suggestions",
            "Weather & Best Season to Visit", 
            "Accessibility & Mobility",
            "Cleanliness & Maintenance",
            "Safety & Security",
            "Staff, Guides & Management",
            "Tickets, Entry & Pricing",
            "Signal Connectivity",
            "Suitability for specific audiences",
            "Historical or Cultural Significance",
            "Architecture & Aesthetics",
            "Spiritual or Religious Vibe",
            "Natural Beauty & Photogenic Spots",
            "Wildlife & Biodiversity",
            "Adventure & Activities",
            "Peace & Relaxation",
            "Shops & Stalls",
            "Food & Local Cuisine",
            "Events & Entertainment",
            "Navigation & Signage",
            "Amenities & Facilities",
            "Transport Connectivity",
            "Value for Time & Money",
            "Local Tips & Insider Advice",
            "Comparisons & Alternatives",
            "Emotional Tone / Vibe"
        ]
        
        # Create output directory
        self.output_dir = Path("enhanced_summary/location_summaries")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_initial_extraction_prompt(self, review_text):
        """Step 1: Initial review categorization prompt"""
        return f"""
I need you to act as a review summarizer with EXTREME ATTENTION TO DETAIL. You must extract EVERY piece of information from the review and categorize it meticulously. Do not miss ANY detail, no matter how small.

Break down the review into its constituent parts and categorize each relevant snippet under the most appropriate container(s). Extract ALL specific details including:
- Exact times, prices, dates mentioned
- Specific activities, sports, facilities
- Detailed descriptions of experiences
- All recommendations and suggestions
- Every positive and negative aspect
- Specific locations, areas, or features mentioned

Please output the categorized content in the following JSON format:

{{
    "container_name": ["extracted review part 1", "extracted review part 2", ...],
    "another_container": ["extracted content"],
    ...
}}

Here are the 26 containers to use for categorization:

{', '.join(self.containers)}

Review to analyze (extract EVERY detail):
"{review_text}"

Be extremely thorough - if a review mentions water sports after 4 PM, extract that EXACT detail. If it mentions specific activities like paragliding, speed boat, banana ride, camel ride, horse ride - extract ALL of them. Do not summarize or skip any information.

Please provide only the JSON output with ALL extracted and categorized content.
"""

    def get_merge_prompt(self, existing_summary, new_review):
        """Step 2-4: Merge new review with existing summary"""
        return f"""
I want you to append this new extracted version with the old review extracted version. Follow these rules EXACTLY:

MERGING RULES:
1. Extract EVERY detail from the new review first using the 26 containers
2. If the new review contains points similar to existing summary, merge them by combining details (avoid redundancy but keep all specific information)
3. If the new review conflicts with existing points, retain both facts (e.g., "some reviewers found it crowded while others say it's not crowded")
4. Maintain ALL specific details like times, prices, activity names, etc.
5. If new review adds specific details to existing general statements, enhance them
6. Keep information in bullet points within each container

CRITICAL: Do not lose ANY specific information during merging. If existing summary says "water sports available" and new review says "water sports available after 4 PM including paragliding, speed boat, banana ride", the result should include ALL these specific details.

Existing Summary:
{json.dumps(existing_summary, indent=2)}

New Review to process (extract EVERY detail):
"{new_review}"

First, extract and categorize the new review using these containers: {', '.join(self.containers)}

Then merge it with the existing summary following the rules above, ensuring NO information is lost.

Provide the final merged summary in JSON format with comprehensive bullet points for each container.
"""

    def process_single_review(self, review_text, existing_summary=None):
        """Process a single review and return categorized content"""
        try:
            if existing_summary is None:
                # Step 1: Initial extraction
                prompt = self.get_initial_extraction_prompt(review_text)
            else:
                # Steps 2-4: Merge with existing
                prompt = self.get_merge_prompt(existing_summary, review_text)
            
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
                
            return json.loads(response_text)
            
        except Exception as e:
            print(f"Error processing review: {e}")
            return existing_summary if existing_summary else {}

    def load_existing_summary(self, location_name):
        """Load existing summary if it exists"""
        safe_name = "".join(c for c in location_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        final_file = self.output_dir / f"{safe_name}_final.json"
        
        if final_file.exists():
            try:
                with open(final_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('summary', {}), data.get('processed_reviews', 0)
            except Exception as e:
                print(f"Error loading existing summary: {e}")
                return {}, 0
        return {}, 0

    def process_location_reviews(self, location_name, reviews_df):
        """Process all reviews for a specific location - UPDATE SAME FILE"""
        print(f"Processing reviews for: {location_name}")
        
        # Filter reviews for this location
        location_reviews = reviews_df[reviews_df['place_name'] == location_name]['review_text'].dropna()
        
        if location_reviews.empty:
            print(f"No reviews found for {location_name}")
            return
        
        # Load existing summary if it exists
        summary, processed_count = self.load_existing_summary(location_name)
        
        if processed_count > 0:
            print(f"Found existing summary with {processed_count} processed reviews. Continuing from where we left off...")
        
        # Process remaining reviews
        remaining_reviews = location_reviews.iloc[processed_count:]
        
        for i, review in enumerate(remaining_reviews):
            if pd.isna(review) or not review.strip():
                continue
                
            current_review_num = processed_count + i + 1
            print(f"Processing review {current_review_num}/{len(location_reviews)}")
            print(f"Review preview: {review[:100]}...")
            
            # Process review one by one (prompt chaining)
            summary = self.process_single_review(review, summary if summary else None)
            processed_count = current_review_num
            
            # Add delay to respect API limits
            time.sleep(2)
            
            # Save progress every 5 reviews (update same file)
            if processed_count % 5 == 0:
                self.save_summary(location_name, summary, processed_count, final=True)
                print(f"Progress saved: {processed_count} reviews processed")
        
        # Save final summary (update same file)
        self.save_summary(location_name, summary, processed_count, final=True)
        print(f"✅ Completed processing {processed_count} reviews for {location_name}")

    def save_summary(self, location_name, summary, count, final=False):
        """Save summary to JSON file - ALWAYS update the same final file"""
        safe_name = "".join(c for c in location_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}_final.json"  # Always use the same final file
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "location_name": location_name,
                "processed_reviews": count,
                "last_updated": pd.Timestamp.now().isoformat(),
                "summary": summary
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Summary updated: {filepath}")

    def process_all_locations(self, csv_file_path="../data/google_maps_reviews.csv"):
        """Process reviews for all locations in the CSV"""
        try:
            # Load reviews data
            reviews_df = pd.read_csv(csv_file_path)
            print(f"Loaded {len(reviews_df)} reviews from {csv_file_path}")
            
            # Get unique locations
            unique_locations = reviews_df['place_name'].dropna().unique()
            print(f"Found {len(unique_locations)} unique locations")
            
            for i, location in enumerate(unique_locations, 1):
                print(f"\n{'='*60}")
                print(f"Processing Location {i}/{len(unique_locations)}: {location}")
                print(f"{'='*60}")
                self.process_location_reviews(location, reviews_df)
                
                # Add delay between locations
                time.sleep(3)
                
        except FileNotFoundError:
            print(f"CSV file not found: {csv_file_path}")
        except Exception as e:
            print(f"Error processing locations: {e}")

    def resume_processing(self, csv_file_path="../data/google_maps_reviews.csv"):
        """Resume processing from where it left off"""
        try:
            reviews_df = pd.read_csv(csv_file_path)
            unique_locations = reviews_df['place_name'].dropna().unique()
            
            for location in unique_locations:
                existing_summary, processed_count = self.load_existing_summary(location)
                total_reviews = len(reviews_df[reviews_df['place_name'] == location]['review_text'].dropna())
                
                if processed_count < total_reviews:
                    print(f"\n📍 Resuming {location}: {processed_count}/{total_reviews} reviews processed")
                    self.process_location_reviews(location, reviews_df)
                else:
                    print(f"✅ {location}: Already completed ({processed_count}/{total_reviews})")
                    
        except Exception as e:
            print(f"Error during resume: {e}")