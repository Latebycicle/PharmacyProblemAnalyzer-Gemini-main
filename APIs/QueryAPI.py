#Api to query mongoDB and use files as context
from flask import Flask, request, jsonify
import torch
from transformers import pipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pymongo
from IPython.display import Markdown, clear_output, display

embed_model = HuggingFaceEmbedding(model_name="./sentence-transformers")
app = Flask(__name__)
uri = "mongodb+srv://geminiuser:1234@cluster0.nurmebz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

@app.route('/query', methods=['POST'])
def query_model():
    query_data = request.json
    query = query_data.get("query")


    client = pymongo.MongoClient(uri)
    db = client.langchain_demo
    collection = db.collection_of_text_blobs

    def generate_embedding(quer):
        temp = embed_model.get_text_embedding(quer)
        return temp

    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": generate_embedding(query),
                "path": "embedding",
                "numCandidates": 50,
                "limit": 1,
                "index": "RAGIndexing",
            }
        }
    ])

    context = ""
    for document in results:
        print(type(document))
        print(f'Text present: {document["text"]}\n')
        context += document["text"]

    print("Query worked")
    print(context)

    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

    print("Pipeline code worked")


    def prompt_tinyllama(prompt, system_prompt=""):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return outputs[0]["generated_text"].split("<|assistant|>")[1]


    prompt = f"With the following context- {context}\nAnswer the following query {query}"
    print("Querying: "+ prompt)
    system_prompt = "You are an expert in this field and always provide detailed and accurate answers"
    response = prompt_tinyllama(prompt, system_prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)