import sys
from pathlib import Path
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from review_vector_db import ReviewVectorDatabase
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class InteractiveReviewQuery:
    def __init__(self):
        """Initialize the interactive query system"""
        print("🎯 Starting Interactive Review Query System...")
        try:
            self.db = ReviewVectorDatabase()
            print("✅ System initialized successfully!")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
        
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("🎯 INTERACTIVE REVIEW QUERY SYSTEM")
        print("="*60)
        print("1. 📊 Get enhanced summary for a location")
        print("2. 🔍 Ask specific question about a location")
        print("3. 📋 List all available locations")
        print("4. 📈 Show database statistics")
        print("5. 🏭 Generate enhanced summary for all locations")
        print("6. ❌ Exit")
        print("="*60)
    
    def get_enhanced_summary(self):
        """Get enhanced summary for a location"""
        try:
            locations = self.db.get_locations_list()
            
            if not locations:
                print("❌ No locations found. Please load reviews first.")
                return
            
            print(f"\n📍 Available locations ({len(locations)}):")
            for i, loc in enumerate(locations[:20], 1):  # Show first 20
                print(f"{i:2d}. {loc}")
            
            if len(locations) > 20:
                print(f"... and {len(locations) - 20} more locations")
            
            location = input("\n🎯 Enter location name (or part of it): ").strip()
            
            if not location:
                print("❌ No location entered")
                return
            
            # Find matching location
            matches = [loc for loc in locations if location.lower() in loc.lower()]
            
            if not matches:
                print(f"❌ No locations found matching '{location}'")
                print("💡 Try using part of the name or check the list above")
                return
            elif len(matches) > 1:
                print(f"\n🔍 Multiple matches found:")
                for i, match in enumerate(matches[:10], 1):  # Show max 10
                    print(f"{i}. {match}")
                
                try:
                    choice = int(input(f"Choose location (1-{min(len(matches), 10)}): ")) - 1
                    if 0 <= choice < len(matches):
                        location = matches[choice]
                    else:
                        print("❌ Invalid choice")
                        return
                except ValueError:
                    print("❌ Invalid input")
                    return
            else:
                location = matches[0]
            
            print(f"\n🎯 Generating enhanced summary for: {location}")
            print("⏳ This may take a few seconds...")
            
            summary = self.db.generate_enhanced_summary(location)
            
            if "error" in summary:
                print(f"❌ Error: {summary['error']}")
            else:
                print(f"\n✅ Enhanced Summary for {location}:")
                print("="*60)
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                
                # Ask if user wants to save
                save = input(f"\n💾 Save this summary to file? (y/n): ").strip().lower()
                if save == 'y':
                    self.save_summary(location, summary)
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def ask_specific_question(self):
        """Ask a specific question about a location"""
        try:
            locations = self.db.get_locations_list()
            
            if not locations:
                print("❌ No locations found. Please load reviews first.")
                return
            
            location = input("🎯 Enter location name: ").strip()
            
            if not location:
                print("❌ No location entered")
                return
            
            # Find matching location
            matches = [loc for loc in locations if location.lower() in loc.lower()]
            
            if not matches:
                print(f"❌ No locations found matching '{location}'")
                return
            elif len(matches) > 1:
                print(f"Multiple matches: {', '.join(matches[:5])}")
                location = matches[0]
                print(f"Using: {location}")
            else:
                location = matches[0]
            
            print("\n🔍 Example questions:")
            print("- What are the water sports available?")
            print("- What are the opening hours?")
            print("- How crowded is it on weekends?")
            print("- What's the entry fee?")
            print("- Is it suitable for families?")
            
            question = input(f"\n❓ Your question about {location}: ").strip()
            
            if not question:
                print("❌ No question provided")
                return
            
            print(f"\n🔍 Searching for information about: {question}")
            print("⏳ Analyzing reviews...")
            
            answer = self.db.query_specific_aspect(location, question)
            
            print(f"\n✅ Answer:")
            print("="*60)
            print(answer)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def list_locations(self):
        """List all available locations"""
        try:
            locations = self.db.get_locations_list()
            
            if not locations:
                print("❌ No locations found. Please load reviews first with:")
                print("   python rag_review_system.py --setup")
                return
            
            print(f"\n📋 Available Locations ({len(locations)}):")
            print("="*60)
            
            for i, location in enumerate(locations, 1):
                print(f"{i:3d}. {location}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def show_database_stats(self):
        """Show database statistics"""
        try:
            print("\n📈 Database Statistics:")
            print("="*60)
            
            stats = self.db.get_database_stats()
            
            if "error" in stats:
                print(f"❌ Error: {stats['error']}")
            else:
                print(f"📊 Total Reviews: {stats['total_reviews']}")
                print(f"📍 Total Locations: {stats['total_locations']}")
                print(f"💾 Database Path: {stats['database_path']}")
                
                if stats['locations']:
                    print(f"\n📍 Sample Locations:")
                    for loc in stats['locations']:
                        print(f"   • {loc}")
                        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def generate_all_summaries(self):
        """Generate summaries for all locations"""
        try:
            confirm = input("⚠️  This will generate summaries for ALL locations. Continue? (y/n): ").strip().lower()
            
            if confirm == 'y':
                print("\n🏭 Starting batch summary generation...")
                self.db.generate_all_summaries()
            else:
                print("❌ Operation cancelled")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def save_summary(self, location_name, summary):
        """Save summary to file"""
        try:
            # Create rag_summaries directory if it doesn't exist
            from pathlib import Path
            output_dir = Path("rag_summaries")
            output_dir.mkdir(exist_ok=True)
            
            safe_name = "".join(c for c in location_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filepath = output_dir / f"{safe_name}_enhanced.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Summary saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving file: {e}")
    
    def start_interactive_session(self):
        """Start the main interactive session"""
        while True:
            try:
                self.display_menu()
                
                choice = input("\n🎯 Enter your choice (1-6): ").strip()
                
                if choice == "1":
                    self.get_enhanced_summary()
                elif choice == "2":
                    self.ask_specific_question()
                elif choice == "3":
                    self.list_locations()
                elif choice == "4":
                    self.show_database_stats()
                elif choice == "5":
                    self.generate_all_summaries()
                elif choice == "6":
                    print("\n👋 Thank you for using the Review Query System!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-6.")
                    
                input("\n⏸️  Press Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    query_system = InteractiveReviewQuery()
    query_system.start_interactive_session()