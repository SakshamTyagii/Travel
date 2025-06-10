import json
import pandas as pd
from pathlib import Path
import streamlit as st

class SummaryViewer:
    def __init__(self):
        self.summaries_dir = Path("enhanced_summary/location_summaries")
    
    def load_summary(self, location_file):
        """Load summary from JSON file"""
        with open(self.summaries_dir / location_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def display_summary(self, summary_data):
        """Display summary in a formatted way"""
        st.title(f"📍 {summary_data['location_name']}")
        st.info(f"Based on {summary_data['processed_reviews']} reviews")
        
        summary = summary_data['summary']
        
        for container, points in summary.items():
            if points:  # Only show containers with content
                st.subheader(f"🔸 {container}")
                if isinstance(points, list):
                    for point in points:
                        st.write(f"• {point}")
                else:
                    st.write(f"• {points}")
                st.write("")

def create_streamlit_app():
    """Create Streamlit app for viewing summaries"""
    st.set_page_config(page_title="Enhanced Review Summaries", page_icon="📊")
    
    viewer = SummaryViewer()
    
    # Get available summaries
    if not viewer.summaries_dir.exists():
        st.error("No summaries found. Please run the summarizer first.")
        return
    
    summary_files = list(viewer.summaries_dir.glob("*_final.json"))
    
    if not summary_files:
        st.warning("No final summaries found. Please complete processing first.")
        return
    
    # Sidebar for location selection
    st.sidebar.title("Select Location")
    
    location_options = {}
    for file in summary_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                location_options[data['location_name']] = file.name
        except:
            continue
    
    selected_location = st.sidebar.selectbox(
        "Choose a location:",
        options=list(location_options.keys())
    )
    
    if selected_location:
        summary_data = viewer.load_summary(location_options[selected_location])
        viewer.display_summary(summary_data)

if __name__ == "__main__":
    create_streamlit_app()