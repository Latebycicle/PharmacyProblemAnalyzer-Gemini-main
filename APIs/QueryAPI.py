"""
Query API for Pharmacy Problem Analyzer

This Flask API handles user queries using vector search and TinyLlama for response generation.
It retrieves relevant documents from MongoDB and generates contextual responses.
"""

from flask import Flask, request, jsonify
import torch
from transformers import pipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pymongo
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

# Initialize embedding model
try:
    model_path = "./sentence-transformers"
    if not os.path.exists(model_path):
        model_path = Config.HUGGINGFACE_MODEL
    
    embed_model = HuggingFaceEmbedding(model_name=model_path)
    logger.info(f"Embedding model loaded: {model_path}")
except Exception as e:
    logger.error(f"Failed to initialize embedding model: {e}")
    raise

# Initialize TinyLlama pipeline
try:
    pipe = pipeline(
        "text-generation",
        model=Config.TINYLLAMA_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    logger.info(f"TinyLlama model loaded: {Config.TINYLLAMA_MODEL}")
except Exception as e:
    logger.error(f"Failed to initialize TinyLlama model: {e}")
    raise

def generate_embedding(query):
    """Generate embedding for the given query."""
    try:
        return embed_model.get_text_embedding(query)
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise

def retrieve_context(query, limit=3, num_candidates=50):
    """
    Retrieve relevant context from MongoDB using vector search.
    
    Args:
        query (str): User query
        limit (int): Number of documents to retrieve
        num_candidates (int): Number of candidates for vector search
        
    Returns:
        str: Combined context from retrieved documents
    """
    try:
        client = pymongo.MongoClient(Config.MONGODB_URI)
        db = client[Config.DB_NAME]
        collection = db[Config.COLLECTION_NAME]
        
        query_embedding = generate_embedding(query)
        
        # Perform vector search
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": num_candidates,
                    "limit": limit,
                    "index": "RAGIndexing",
                }
            }
        ])
        
        # Combine retrieved documents
        context = ""
        doc_count = 0
        for document in results:
            if "text" in document:
                context += document["text"] + "\n\n"
                doc_count += 1
                logger.info(f"Retrieved document {doc_count}: {document['text'][:100]}...")
        
        if not context:
            logger.warning("No relevant documents found")
            context = "No relevant information found in the knowledge base."
        else:
            logger.info(f"Retrieved {doc_count} relevant documents")
        
        client.close()
        return context.strip()
        
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        return "Error retrieving context from knowledge base."

def prompt_tinyllama(prompt, system_prompt=""):
    """
    Generate response using TinyLlama model.
    
    Args:
        prompt (str): User prompt with context
        system_prompt (str): System instructions
        
    Returns:
        str: Generated response
    """
    try:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        
        formatted_prompt = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = pipe(
            formatted_prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        response = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
        return response
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "I apologize, but I encountered an error while generating a response."

@app.route('/query', methods=['POST'])
def query_model():
    """
    Handle user queries and return AI-generated responses.
    
    Returns:
        JSON response with the generated answer
    """
    try:
        # Validate request
        query_data = request.json
        if not query_data or 'query' not in query_data:
            return jsonify({"error": "Query is required"}), 400
        
        query = query_data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant context
        context = retrieve_context(query)
        
        # Generate response
        system_prompt = """You are an expert pharmacy assistant specializing in analyzing pharmacy problems and operational issues. 
        Provide detailed, accurate, and helpful responses based on the given context. 
        If the context doesn't contain relevant information, clearly state that and provide general guidance if appropriate."""
        
        prompt = f"""Context: {context}

Question: {query}

Please provide a comprehensive answer based on the context provided above."""
        
        logger.info(f"Generating response for query: {query[:50]}...")
        response = prompt_tinyllama(prompt, system_prompt)
        
        logger.info("Response generated successfully")
        
        return jsonify({
            "query": query,
            "response": response,
            "context_length": len(context)
        }), 200
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return jsonify({"error": f"Query processing failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Test MongoDB connection
        client = pymongo.MongoClient(Config.MONGODB_URI)
        client.admin.command('ismaster')
        client.close()
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "model": "loaded"
        }), 200
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