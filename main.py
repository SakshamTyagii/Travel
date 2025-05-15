import sys
import os
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import importlib.util

main_app = Flask(__name__)
CORS(main_app)  # Enable CORS for all routes

# In main.py, import the working local pipeline instead
try:
    # Explicitly use local version first 
    from ragpipeline.rag_pipeline import rag_pipeline
    print("Using local Ollama RAG pipeline")
except ImportError:
    print("WARNING: Could not import local RAG pipeline")
    try:
        from ragpipeline.rag_pipeline_gcp import rag_pipeline
        print("Using GCP-compatible RAG pipeline")
    except ImportError:
        print("WARNING: No RAG pipeline available")
        def rag_pipeline(query):
            return "RAG pipeline not available"

# Add a route to serve the RAG test HTML file
@main_app.route('/rag/test')
def rag_test():
    return send_from_directory(os.path.join(os.path.dirname(__file__), "ragpipeline"), "rag_test.html")

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
        print(f"RAG Query Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@main_app.route('/rag/query', methods=['GET'])
def rag_query_get():
    return "RAG API is running! Use POST to query.", 200

# In main.py
try:
    # Import search dynamically because it might have conflicting dependencies
    import importlib.util
    import importlib
    
    # Try a standard import first (often more reliable)
    try:
        from search_function.search import app as search_app
        main_app.register_blueprint(search_app, url_prefix='/search')
        print("Successfully registered search blueprint via direct import")
    except ImportError:
        # Fall back to spec-based import
        spec = importlib.util.spec_from_file_location("search", 
                                                    os.path.join(os.path.dirname(__file__), 
                                                                "search_function", "search.py"))
        search_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(search_module)
        search_app = search_module.app
        
        # Add debug info
        print(f"Search app type: {type(search_app)}")
        
        main_app.register_blueprint(search_app, url_prefix='/search')
        print("Successfully registered search blueprint via spec import")
except Exception as e:
    print(f"Error importing search module: {str(e)}")
    import traceback
    traceback.print_exc()
    
# Serve the index.html file from search_function if available
@main_app.route('/')
def index():
    # First check if we have a dedicated index.html in the root
    root_index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(root_index_path):
        return send_from_directory(os.path.dirname(__file__), "index.html")
        
    # Fall back to the search index if available
    search_index_path = os.path.join(os.path.dirname(__file__), "search_function", "index.html")
    if os.path.exists(search_index_path):
        return send_from_directory(os.path.join(os.path.dirname(__file__), "search_function"), "index.html")
    else:
        return "Delhi Travel Assistant API is running!", 200

# Add this to main.py to handle requests to /suggest
@main_app.route('/suggest')
def redirect_to_search_suggest():
    # Get all the original query parameters
    query_string = request.query_string.decode()
    
    # Redirect to the proper search endpoint
    return redirect(f'/search/suggest?{query_string}')

if __name__ == "__main__":
    # This will be used when running locally
    port = int(os.environ.get("PORT", 8080))
    main_app.run(host="0.0.0.0", port=port, debug=True)  # Set debug=True for better error messages
