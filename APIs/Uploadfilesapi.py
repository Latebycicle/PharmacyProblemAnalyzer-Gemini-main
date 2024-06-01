#Api to embed and upload files to mongoDB 
from flask import Flask, request, jsonify
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pymongo
import os
import shutil

app = Flask(__name__)
uri = "mongodb+srv://geminiuser:1234@cluster0.nurmebz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


DB_NAME = "langchain_demo"
COLLECTION_NAME = 'collection_of_text_blobs'
INDEX_NAME = 'Indexx'
mongodb_client = pymongo.MongoClient(uri)
db = mongodb_client[DB_NAME]
collection = db[COLLECTION_NAME]

print("Atlas client initialized")


embed_model = HuggingFaceEmbedding(model_name="./sentence-transformers")

vector = embed_model.get_text_embedding("Vector Search with MongoDB")
print(len(vector))


service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)

from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import StorageContext

vector_store = MongoDBAtlasVectorSearch(mongodb_client = mongodb_client,
                                 db_name = DB_NAME, collection_name = COLLECTION_NAME,
                                 index_name  = INDEX_NAME)

storage_context = StorageContext.from_defaults(vector_store=vector_store)


@app.route('/upload', methods=['POST'])
def upload_documents():
    temp_dir = "./temp_files"
    os.makedirs(temp_dir, exist_ok=True)

    files = request.files.getlist("files")
    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

    docs = SimpleDirectoryReader(input_dir=temp_dir).load_data()
    print(f"Loaded {len(docs)} chunks from uploaded files.")

    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, service_context=service_context)

    for doc in docs:
        embedding = embed_model.get_text_embedding(doc.text)
        collection.insert_one({"text": doc.text, "embedding": embedding})
    shutil.rmtree(temp_dir)

    return jsonify({"message": f"Successfully loaded {len(docs)} documents into MongoDB."}), 200

def generate_embedding(text):
    return embed_model.get_text_embedding(text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)