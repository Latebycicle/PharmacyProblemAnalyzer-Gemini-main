"""
Pharmacy Problem Analyzer - Streamlit Web Interface

This module provides a Streamlit-based web interface for querying pharmacy problems
using AI-powered vector search and retrieval-augmented generation (RAG).
"""

import streamlit as st
import os
from pymongo import MongoClient
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import ServiceContext
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex

from config import Config

# Initialize configuration
try:
    Config.validate_config()
except ValueError as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# Set Google API key
os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY

# Configure Streamlit app
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    layout="wide",
    page_icon="🏥"
)
st.title(Config.APP_TITLE)

@st.cache_resource
def initialize_components():
    """Initialize and cache AI components for better performance."""
    try:
        # Set up MongoDB client
        mongodb_client = MongoClient(Config.MONGODB_URI)
        
        # Load Google Gemini embedding model
        embed_model = GeminiEmbedding(model_name=Config.EMBEDDING_MODEL)
        
        # Load Gemini model to be used as the LLM
        llm = Gemini(model=Config.GEMINI_MODEL)
        
        # Create llama_index service context
        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
        
        # Set up the vector store
        vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=mongodb_client,
            db_name=Config.DB_NAME,
            collection_name=Config.COLLECTION_NAME,
            index_name=Config.INDEX_NAME
        )
        
        # Create the vector store index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=service_context
        )
        
        # Set up the query engine
        query_engine = index.as_query_engine()
        
        return query_engine
        
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None

# Initialize query engine
query_engine = initialize_components()

if query_engine is None:
    st.error("Failed to initialize the system. Please check your configuration.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message, kind in st.session_state.messages:
    with st.chat_message(kind):
        st.markdown(message)

# Chat input
prompt = st.chat_input("Ask your questions about pharmacy problems...")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append([prompt, "user"])
    
    # Generate and display AI response
    with st.spinner("Generating response..."):
        try:
            answer = query_engine.query(prompt)
            if answer:
                st.chat_message("ai").markdown(str(answer))
                st.session_state.messages.append([str(answer), "ai"])
            else:
                st.chat_message("ai").markdown("I'm sorry, I couldn't generate a response to your question.")
                st.session_state.messages.append(["I'm sorry, I couldn't generate a response to your question.", "ai"])
        except Exception as e:
            error_msg = f"An error occurred while processing your question: {e}"
            st.chat_message("ai").markdown(error_msg)
            st.session_state.messages.append([error_msg, "ai"])

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI-powered pharmacy assistant helps analyze and understand common pharmacy problems by:
    
    - 🔍 **Vector Search**: Uses semantic search to find relevant information
    - 🤖 **AI Analysis**: Leverages Google Gemini for intelligent responses
    - 📚 **Knowledge Base**: Draws from uploaded pharmacy documents and reports
    - 💬 **Interactive Chat**: Provides conversational interface for easy querying
    """)
    
    st.header("How to Use")
    st.markdown("""
    1. Type your question about pharmacy problems in the chat input
    2. The system will search through the knowledge base
    3. AI will provide a comprehensive answer based on relevant documents
    4. Continue the conversation for follow-up questions
    """)