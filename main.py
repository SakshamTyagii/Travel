import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from search_function.search import app as search_app
from ragpipeline.rag_pipeline import rag_pipeline

main_app = Flask(__name__)

# Register the search blueprint (Flask app)
main_app.register_blueprint(search_app, url_prefix='/search')

# Add a simple endpoint for the RAG pipeline (wrap as Flask route)
@main_app.route('/rag/query', methods=['POST'])
def rag_query():
    from flask import request, jsonify
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    answer = rag_pipeline(query)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    main_app.run(host="0.0.0.0", port=8080)
