import streamlit as st
import os
from pymongo import MongoClient
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import ServiceContext
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex


# Check if 'key' already exists in session_state
# If not, then initialize it
# web app config
DB_NAME = "langchain_demo"
COLLECTION_NAME = 'collection_of_text_blobs'
INDEX_NAME = 'Indexx'

dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
# web app config, configure the streamlit app as you'd like
st.set_page_config(page_title= "Page title",layout="wide", page_icon="📙")
# App title to be displayed at the top of the app, could be same as page title if you like
st.title("App title")
API_KEY = 'AIzaSyA3xX1ZaJGwFO2KNwS3wR6oj4ZCMM9xkX0'
uri = "mongodb+srv://geminiuser:1234@cluster0.nurmebz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)

#genai.configure(api_key=API_KEY)

#from langchain_google_genai import GoogleGenerativeAIEmbeddings

google_api_key = API_KEY
os.environ["GOOGLE_API_KEY"] = google_api_key


# set up mongodb client
mongodb_client = MongoClient(uri)

# load google gemini embedding model
embed_model = GeminiEmbedding(model_name="models/embedding-001")

# load gemini model to be used as the LLM
llm = Gemini(model="models/gemini-pro")

# create llama_index service context
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

# set up the vectore store with the details, to have access to the specific data
vector_store = MongoDBAtlasVectorSearch(mongodb_client = mongodb_client, db_name = DB_NAME, collection_name = COLLECTION_NAME, index_name  = INDEX_NAME)

# Create the vector store and index and the pipeline for vector search will be created
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

# set up the created index as a query engine
query_llm = index.as_query_engine()

# chat interface for consistent queries
if "messages" not in st.session_state: # chats for each session will be displayed
    st.session_state.messages = []

# Display for all the messages
for message, kind in st.session_state.messages:
        with st.chat_message(kind):
            st.markdown(message)
            
prompt = st.chat_input("Ask your questions ...")

if prompt:
    # Handling prompts and rendering to the chat interface
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append([prompt, "user"]) # updating the list of prompts 

    # using the query engine to get response, rendering the answer and adding to conversation history
    with st.spinner("Generating response"):
        answer = query_llm.query(prompt)
        if answer:
            st.chat_message("ai").markdown(answer)
            st.session_state.messages.append([answer, "ai"])