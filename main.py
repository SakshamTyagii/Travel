import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import importlib.util

main_app = Flask(__name__)
CORS(main_app)  # Enable CORS for all routes

# Import the RAG pipeline from the GCP-compatible file
try:
    # Try to import the GCP version first
    from ragpipeline.rag_pipeline_gcp import rag_pipeline
    print("Using GCP-compatible RAG pipeline")
except ImportError:
    # Fall back to local version
    try:
        from ragpipeline.rag_pipeline import rag_pipeline
        print("Using local RAG pipeline")
    except ImportError:
        print("WARNING: Could not import RAG pipeline")
        def rag_pipeline(query):
            return "RAG pipeline not available"

# Import the search app
try:
    # Import search dynamically because it might have conflicting dependencies
    spec = importlib.util.spec_from_file_location("search", 
                                                os.path.join(os.path.dirname(__file__), 
                                                            "search_function", "search.py"))
    search_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(search_module)
    search_app = search_module.app
    main_app.register_blueprint(search_app, url_prefix='/search')
    print("Successfully registered search blueprint")
except Exception as e:
    print(f"Error importing search module: {str(e)}")

# Add a simple endpoint for the RAG pipeline
@main_app.route('/rag/query', methods=['POST'])
def rag_query():
    try:
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        answer = rag_pipeline(query)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_app.route('/rag/query', methods=['GET'])
def rag_query_get():
    return "RAG API is running! Use POST to query.", 200
    
# Serve the index.html file from search_function if available
@main_app.route('/')
def index():
    index_path = os.path.join(os.path.dirname(__file__), "search_function", "index.html")
    if os.path.exists(index_path):
        return send_from_directory(os.path.join(os.path.dirname(__file__), "search_function"), "index.html")
    else:
        return "Delhi Travel Assistant API is running!", 200

if __name__ == "__main__":
    # This will be used when running locally
    port = int(os.environ.get("PORT", 8080))
    main_app.run(host="0.0.0.0", port=port, debug=False)
