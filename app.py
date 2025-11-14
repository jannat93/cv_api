import os
import re
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import faiss
from pdfminer.high_level import extract_text
import nltk

# Use pre-downloaded NLTK stopwords
nltk.data.path.append('./nltk_data')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# ------------------- Init -------------------
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load small embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

# FAISS index and job storage
faiss_index = None
job_data = []

# ------------------- Utils -------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in stop_words]
    return " ".join(tokens)

def embed_text(texts):
    vecs = model.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs

def extract_text_from_pdf(path):
    try:
        return extract_text(path)
    except:
        return ""

def rebuild_index():
    global faiss_index
    if len(job_data) == 0:
        faiss_index = faiss.IndexFlatIP(384)
        return
    mat = np.vstack([x["embedding"] for x in job_data])
    faiss.normalize_L2(mat)
    faiss_index = faiss.IndexFlatIP(384)
    faiss_index.add(mat)

# ------------------- Routes -------------------
@app.route("/")
def home():
    return "CV Analyzer API is running."

@app.route("/jobs", methods=["POST"])
def add_job():
    data = request.json
    title = data.get("title")
    description = data.get("description")
    if not title or not description:
        return jsonify({"error": "title and description required"}), 400
    vec = embed_text([preprocess(description)])[0]
    job_data.append({"title": title, "description": description, "embedding": vec})
    rebuild_index()
    return jsonify({"status": "ok", "job_id": len(job_data)-1})

@app.route("/analyze", methods=["POST"])
def analyze():
    # Check if PDF file uploaded
    if "file" in request.files:
        f = request.files["file"]
        filename = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        raw_text = extract_text_from_pdf(path)
    else:
        data = request.get_json(silent=True) or {}
        raw_text = data.get("cv", "")
        if not raw_text:
            return jsonify({"error": "No CV provided"}), 400

    vec = embed_text([preprocess(raw_text)])[0]
    top_k = int(request.args.get("k", 5))
    if faiss_index is None or faiss_index.ntotal == 0:
        return jsonify({"matches": []})
    D, I = faiss_index.search(np.expand_dims(vec, axis=0), top_k)
    matches = []
    for idx, score in zip(I[0], D[0]):
        job = job_data[idx]
        matches.append({
            "job_id": idx,
            "title": job["title"],
            "description": job["description"][:200],
            "score": float(score)
        })
    return jsonify({"cv_text_snippet": raw_text[:500], "matches": matches})

# ------------------- Start -------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
