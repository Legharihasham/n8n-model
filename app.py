from flask import Flask, request, jsonify
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load model and files
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight and free
with open("university_combined_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index("university_combined_index.faiss")

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data["query"]
    query_embedding = model.encode([question])
    D, I = index.search(np.array(query_embedding).astype("float32"), k=5)
    results = [chunks[i] for i in I[0]]
    return jsonify({"context": " ".join(results)})

@app.route("/", methods=["GET"])
def home():
    return "Flask server is running!"

if __name__ == "__main__":
    app.run(port=5000)
