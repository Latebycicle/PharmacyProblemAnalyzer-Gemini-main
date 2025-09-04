"""
Configuration module for Pharmacy Problem Analyzer.
Handles environment variables and application settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class."""
    
    # Google API Configuration
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv('MONGODB_URI', '')
    DB_NAME = os.getenv('DB_NAME', 'langchain_demo')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'collection_of_text_blobs')
    INDEX_NAME = os.getenv('INDEX_NAME', 'Indexx')
    
    # Streamlit Configuration
    APP_TITLE = os.getenv('APP_TITLE', 'Pharmacy Problem Analyzer')
    PAGE_TITLE = os.getenv('PAGE_TITLE', 'Pharmacy Assistant')
    
    # Model Configuration
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'models/gemini-pro')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'models/embedding-001')
    HUGGINGFACE_MODEL = os.getenv('HUGGINGFACE_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    TINYLLAMA_MODEL = os.getenv('TINYLLAMA_MODEL', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    
    # Flask Configuration
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        required_vars = ['GOOGLE_API_KEY', 'MONGODB_URI']
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True