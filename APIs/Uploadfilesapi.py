"""
Document Upload API for Pharmacy Problem Analyzer

This Flask API handles document upload, embedding generation, and storage in MongoDB.
It processes pharmacy documents and creates vector embeddings for semantic search.
"""

from flask import Flask, request, jsonify
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
import pymongo
import os
import shutil
import logging
from pathlib import Path

from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Validate configuration
try:
    Config.validate_config()
except ValueError as e:
    logger.error(f"Configuration Error: {e}")
    raise

# Initialize MongoDB connection
mongodb_client = pymongo.MongoClient(Config.MONGODB_URI)
db = mongodb_client[Config.DB_NAME]
collection = db[Config.COLLECTION_NAME]

logger.info("MongoDB Atlas client initialized")

# Initialize embedding model
try:
    # Use a more standard model path or fallback to online model
    model_path = "./sentence-transformers"
    if not os.path.exists(model_path):
        model_path = Config.HUGGINGFACE_MODEL
    
    embed_model = HuggingFaceEmbedding(model_name=model_path)
    logger.info(f"Embedding model loaded: {model_path}")
    
    # Test embedding
    test_vector = embed_model.get_text_embedding("Vector Search with MongoDB")
    logger.info(f"Embedding dimension: {len(test_vector)}")
    
except Exception as e:
    logger.error(f"Failed to initialize embedding model: {e}")
    raise

# Create service context
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)

# Setup vector store
vector_store = MongoDBAtlasVectorSearch(
    mongodb_client=mongodb_client,
    db_name=Config.DB_NAME,
    collection_name=Config.COLLECTION_NAME,
    index_name=Config.INDEX_NAME
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

@app.route('/upload', methods=['POST'])
def upload_documents():
    """
    Upload and process documents for embedding generation.
    
    Returns:
        JSON response with success/error message
    """
    try:
        # Create temporary directory for uploaded files
        temp_dir = Path("./temp_files")
        temp_dir.mkdir(exist_ok=True)
        
        # Get uploaded files
        files = request.files.getlist("files")
        if not files or not files[0].filename:
            return jsonify({"error": "No files provided"}), 400
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file.filename:
                file_path = temp_dir / file.filename
                file.save(file_path)
                saved_files.append(file.filename)
                logger.info(f"Saved file: {file.filename}")
        
        if not saved_files:
            return jsonify({"error": "No valid files found"}), 400
        
        # Load and process documents
        docs = SimpleDirectoryReader(input_dir=str(temp_dir)).load_data()
        logger.info(f"Loaded {len(docs)} document chunks from uploaded files")
        
        if not docs:
            return jsonify({"error": "No documents could be processed"}), 400
        
        # Create vector index
        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            service_context=service_context
        )
        
        # Store documents with embeddings in MongoDB
        for doc in docs:
            try:
                embedding = embed_model.get_text_embedding(doc.text)
                document_data = {
                    "text": doc.text,
                    "embedding": embedding,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
                collection.insert_one(document_data)
            except Exception as e:
                logger.error(f"Failed to process document: {e}")
                continue
        
        # Cleanup temporary files
        shutil.rmtree(temp_dir)
        
        response_data = {
            "message": f"Successfully processed {len(docs)} documents from {len(saved_files)} files",
            "files_processed": saved_files,
            "documents_created": len(docs)
        }
        
        logger.info(f"Upload completed: {response_data}")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        # Cleanup on error
        if 'temp_dir' in locals() and temp_dir.exists():
            shutil.rmtree(temp_dir)
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Test MongoDB connection
        mongodb_client.admin.command('ismaster')
        return jsonify({"status": "healthy", "database": "connected"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle not found errors."""
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == "__main__":
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=False)