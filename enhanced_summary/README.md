# Enhanced Review Summarization System

This system processes travel reviews one by one using prompt chaining to create comprehensive, categorized summaries for each location.

## Features

- **Prompt Chaining**: Processes reviews individually and progressively builds summaries
- **26 Category System**: Categorizes review content into specific containers
- **Conflict Resolution**: Handles conflicting opinions by preserving both viewpoints
- **Progressive Summarization**: Merges similar points while avoiding redundancy
- **Automated Processing**: Can process entire CSV files automatically

## Setup

1. Install requirements:
```bash
pip install -r enhanced_summary/requirements.txt
```

2. Ensure your `.env` file contains:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

### Process All Locations
```bash
python enhanced_summary/main.py
```

### Process Specific Location
```bash
python enhanced_summary/batch_processor.py --location "Location Name"
```

### List Available Locations
```bash
python enhanced_summary/batch_processor.py --list
```

### View Summaries (Streamlit App)
```bash
streamlit run enhanced_summary/summary_viewer.py
```

## Output

- Summaries are saved in `enhanced_summary/location_summaries/`
- Each location gets a JSON file with categorized insights
- Partial summaries are saved every 10 reviews for backup

## Categories

The system categorizes reviews into 26 containers including:
- Crowd Levels & Timing Suggestions
- Weather & Best Season to Visit
- Accessibility & Mobility
- Food & Local Cuisine
- Natural Beauty & Photogenic Spots
- And 21 more specialized categories...

This creates a comprehensive database of location insights that can be used for enhanced travel recommendations.