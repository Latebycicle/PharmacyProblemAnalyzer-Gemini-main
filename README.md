# AI-Powered Pharmacy Problem Analyzer

A comprehensive AI-powered system for analyzing pharmacy operational problems using advanced natural language processing, vector search, and retrieval-augmented generation (RAG) techniques.

## 🎯 Overview

This application helps pharmacy administrators and staff understand, analyze, and address common operational issues by leveraging AI to process and query pharmacy-related documents and reports. The system uses sophisticated embedding models and vector search to provide contextual, intelligent responses to pharmacy-related questions.

## 🏗️ Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Web Interface (ragtrial.py)                     │
│  - Interactive chat interface                              │
│  - Real-time query processing                              │
│  - Session management                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   API Layer                                 │
├─────────────────────────────────────────────────────────────┤
│  Upload API (APIs/Uploadfilesapi.py)                       │
│  - Document processing and embedding generation            │
│  - Vector storage in MongoDB                               │
│                                                             │
│  Query API (APIs/QueryAPI.py)                              │
│  - Vector search and context retrieval                     │
│  - Response generation using TinyLlama                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 AI/ML Layer                                 │
├─────────────────────────────────────────────────────────────┤
│  Embedding Models:                                          │
│  - Google Gemini (models/embedding-001)                    │
│  - HuggingFace Sentence Transformers                       │
│                                                             │
│  Language Models:                                           │
│  - Google Gemini Pro (primary LLM)                         │
│  - TinyLlama 1.1B (alternative/API LLM)                    │
│                                                             │
│  Processing Framework:                                      │
│  - LlamaIndex for document processing                      │
│  - Vector store integration                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Data Layer                                  │
├─────────────────────────────────────────────────────────────┤
│  MongoDB Atlas                                              │
│  - Vector embeddings storage                               │
│  - Document metadata                                        │
│  - Atlas Vector Search index                               │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend & Web Framework:**
- **Streamlit**: Interactive web interface with chat functionality
- **Flask**: RESTful API endpoints for document upload and querying

**AI/ML Technologies:**
- **Google Gemini**: Primary language model and embedding generation
- **TinyLlama**: Lightweight alternative LLM for API responses
- **LlamaIndex**: Document processing, indexing, and RAG implementation
- **HuggingFace Transformers**: Alternative embedding models and model management
- **Sentence Transformers**: Semantic text embeddings

**Database & Vector Search:**
- **MongoDB Atlas**: Primary database with vector search capabilities
- **Atlas Vector Search**: Semantic similarity search using cosine similarity
- **PyMongo**: MongoDB Python driver

**Development & Configuration:**
- **Python 3.12+**: Core programming language
- **python-dotenv**: Environment variable management
- **Pydantic**: Data validation and configuration management

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- MongoDB Atlas account
- Google Cloud account with Gemini API access

### Step 1: Clone Repository

```bash
git clone https://github.com/Latebycicle/PharmacyProblemAnalyzer-Gemini-main.git
cd PharmacyProblemAnalyzer-Gemini-main
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Environment Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Configure your environment variables in `.env`:
```env
# Google API Configuration
GOOGLE_API_KEY=your_google_gemini_api_key

# MongoDB Configuration
MONGODB_URI=your_mongodb_atlas_connection_string
DB_NAME=langchain_demo
COLLECTION_NAME=collection_of_text_blobs
INDEX_NAME=Indexx

# Application Configuration
APP_TITLE=Pharmacy Problem Analyzer
PAGE_TITLE=Pharmacy Assistant
```

### Step 4: MongoDB Atlas Setup

1. Create a MongoDB Atlas cluster
2. Create a database named `langchain_demo`
3. Create a collection named `collection_of_text_blobs`
4. Create an Atlas Vector Search index named `Indexx` with the following configuration:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    }
  ]
}
```

## 📖 Usage

### Running the Streamlit Web Interface

```bash
streamlit run ragtrial.py
```

The web interface provides:
- Interactive chat interface for querying pharmacy problems
- Real-time AI responses based on uploaded documents
- Session management for conversation history
- Sidebar with usage instructions and system information

### API Usage

#### 1. Document Upload API

Upload pharmacy documents for processing and embedding generation:

```bash
curl -X POST http://localhost:5000/upload \
  -F "files=@pharmacy_report1.txt" \
  -F "files=@pharmacy_issues.pdf"
```

**Response:**
```json
{
  "message": "Successfully processed 15 documents from 2 files",
  "files_processed": ["pharmacy_report1.txt", "pharmacy_issues.pdf"],
  "documents_created": 15
}
```

#### 2. Query API

Query the system for pharmacy-related information:

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are common medication adherence issues in pharmacies?"}'
```

**Response:**
```json
{
  "query": "What are common medication adherence issues in pharmacies?",
  "response": "Based on the pharmacy documents, common medication adherence issues include...",
  "context_length": 1250
}
```

#### 3. Health Check

Check system status:

```bash
curl http://localhost:5000/health
```

## 🔧 Technical Implementation Details

### Document Processing Pipeline

1. **Document Ingestion**: Users upload pharmacy-related documents (text, PDF, etc.)
2. **Text Extraction**: LlamaIndex SimpleDirectoryReader processes documents
3. **Chunking**: Documents are split into semantic chunks for optimal embedding
4. **Embedding Generation**: 
   - Primary: Google Gemini embedding model (768 dimensions)
   - Fallback: HuggingFace Sentence Transformers
5. **Vector Storage**: Embeddings stored in MongoDB Atlas with metadata
6. **Index Creation**: Atlas Vector Search index enables fast similarity search

### Query Processing Pipeline

1. **Query Embedding**: User query converted to vector representation
2. **Vector Search**: MongoDB Atlas performs cosine similarity search
3. **Context Retrieval**: Most relevant document chunks retrieved
4. **Response Generation**: 
   - Streamlit interface: Google Gemini Pro generates response
   - API interface: TinyLlama generates response with retrieved context
5. **Response Delivery**: Formatted response returned to user

### Configuration Management

The system uses a centralized configuration approach:

- **Environment Variables**: Sensitive data (API keys, connection strings)
- **Config Class**: Validates and manages all application settings
- **Default Values**: Sensible defaults for non-sensitive configurations
- **Validation**: Startup validation ensures required configuration is present

### Error Handling & Logging

- **Comprehensive Logging**: All components include structured logging
- **Graceful Degradation**: System handles missing models or connection issues
- **Error Recovery**: Automatic cleanup of temporary files and resources
- **Health Monitoring**: Built-in health check endpoints for system monitoring

### Security Features

- **No Hardcoded Credentials**: All sensitive data managed via environment variables
- **Input Validation**: API inputs validated and sanitized
- **Connection Management**: Proper database connection handling and cleanup
- **File Security**: Temporary file handling with automatic cleanup

## 📊 Performance Characteristics

### Embedding Models
- **Google Gemini**: High quality, 768-dimensional embeddings
- **Processing Speed**: ~100-500 documents per minute (varies by document size)
- **Memory Usage**: ~2-4GB RAM for model loading

### Vector Search
- **Search Latency**: <100ms for typical queries
- **Scalability**: Supports millions of document chunks
- **Accuracy**: High semantic similarity matching

### Response Generation
- **Gemini Pro**: High-quality responses, ~2-5 seconds
- **TinyLlama**: Faster responses (~1-2 seconds), lower resource usage

## 🔍 Monitoring & Maintenance

### Health Checks
- Database connectivity monitoring
- Model availability verification
- API endpoint status checking

### Logging
- Structured logging with configurable levels
- Request/response tracking
- Error tracking and debugging information

### Performance Monitoring
- Response time tracking
- Resource usage monitoring
- Query pattern analysis

## 🛠️ Development & Customization

### Adding New Models

1. Update `config.py` with new model configuration
2. Implement model initialization in relevant API files
3. Update requirements.txt with new dependencies

### Extending API Functionality

1. Add new endpoints to Flask applications
2. Implement proper error handling and logging
3. Update documentation and tests

### Custom Document Processing

1. Extend SimpleDirectoryReader for new file types
2. Implement custom chunking strategies
3. Add metadata extraction as needed

## 📋 File Structure

```
PharmacyProblemAnalyzer-Gemini-main/
├── APIs/
│   ├── Uploadfilesapi.py          # Document upload and processing API
│   └── QueryAPI.py                # Query processing and response API
├── sample_files/                  # Example pharmacy documents
│   ├── sample1.txt
│   ├── sample2.txt
│   └── sample3.txt
├── test_files/                    # Test documents
├── config.py                      # Configuration management
├── ragtrial.py                    # Streamlit web interface
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
└── README.md                      # This documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Ensure all tests pass and code follows style guidelines
5. Submit a pull request

## 📄 License

This project is designed for pharmacy operational analysis and research purposes.

## 🆘 Support & Troubleshooting

### Common Issues

**Installation Issues:**
- Ensure Python 3.8+ is installed
- Check CUDA availability for GPU acceleration (optional)
- Verify MongoDB Atlas connectivity

**Model Loading Issues:**
- Verify Google API key is valid and has Gemini access
- Check internet connectivity for model downloads
- Ensure sufficient disk space for model caching

**Database Issues:**
- Verify MongoDB Atlas connection string
- Check database and collection names
- Ensure Vector Search index is properly configured

### Getting Help

For technical issues or questions:
1. Check the troubleshooting section above
2. Review application logs for error details
3. Ensure all environment variables are properly set
4. Verify network connectivity to external services

---

*This documentation provides a comprehensive overview of the Pharmacy Problem Analyzer's technical implementation. For specific implementation questions or custom deployment scenarios, refer to the code comments and configuration options within each module.*
 