import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_swagger_ui import get_swaggerui_blueprint
from rag_app.application import RagApplication

load_dotenv()

if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_TRACING_V2") == "true":
    from langsmith import Client
    client = Client()

def create_app(rag_app=None):
    """Create and configure the Flask application.
    
    Args:
        rag_app (RagApplication): An initialized RagApplication instance
        
    Returns:
        Flask: The configured Flask application
    """
    if rag_app is None:
        rag_app = RagApplication()
        rag_app.initialize()
        
    app = Flask(__name__)
    
    @app.route('/api/query', methods=['POST'])
    def handle_query():
        try:
            data = request.get_json()
            if not data or 'question' not in data:
                return jsonify({"error": "Missing 'question' field in request body"}), 400

            question = data['question']
            result = rag_app.query(question)
            
            return jsonify({
                "data": result
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/openapi.yaml')
    def serve_openapi_spec():
        return send_file('../openapi.yaml', mimetype='text/yaml')

    swagger_ui_blueprint = get_swaggerui_blueprint(
        '/api/docs',
        '/api/openapi.yaml',
        config={
            'app_name': "RAG Application API"
        }
    )
    app.register_blueprint(swagger_ui_blueprint, url_prefix='/api/docs')
    
    return app

if __name__ == '__main__':
    rag_app = RagApplication()
    rag_app.initialize()
    
    app = create_app(rag_app)
    port = int(os.getenv("FLASK_PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)