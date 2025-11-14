import os
import re
import numpy as np
from typing import List

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import faiss
from pdfminer.high_level import extract_text

from sqlalchemy import Column, Integer, String, LargeBinary, Text, create_engine, select
from sqlalchemy.orm import declarative_base, Session

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# -----------------------------------------------------
# INIT
# -----------------------------------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_NAME = "all-MiniLM-L6-v2"

# ❗ FIXED: DO NOT wrap DB URL inside os.getenv()
# You must set your actual Neon DB URL here:
DB_URL = "postgresql://neondb_owner:npg_NmF0Mj7tTJcE@ep-cold-king-ad1rg2u9-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"

# -----------------------------------------------------
# Load Transformer model
# -----------------------------------------------------
model = SentenceTransformer(MODEL_NAME)

# -----------------------------------------------------
# SQLAlchemy Setup
# -----------------------------------------------------
Base = declarative_base()
engine = create_engine(DB_URL, echo=False, future=True)

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)

Base.metadata.create_all(engine)

# -----------------------------------------------------
# FAISS Index Globals
# -----------------------------------------------------
FAISS_INDEX_FILE = "faiss_index.bin"
faiss_index = None
job_id_to_index = []


# -----------------------------------------------------
# FAISS Build
# -----------------------------------------------------
def rebuild_faiss_index():
    global faiss_index, job_id_to_index

    with Session(engine) as session:
        jobs = session.execute(select(Job)).scalars().all()

    embeddings = []
    job_id_to_index = []

    for job in jobs:
        vec = np.frombuffer(job.embedding, dtype=np.float32)
        embeddings.append(vec)
        job_id_to_index.append(job.id)

    if len(embeddings) == 0:
        faiss_index = faiss.IndexFlatIP(384)
        return

    mat = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(mat)

    faiss_index = faiss.IndexFlatIP(384)
    faiss_index.add(mat)

    faiss.write_index(faiss_index, FAISS_INDEX_FILE)


def load_or_build_index():
    global faiss_index, job_id_to_index

    if os.path.exists(FAISS_INDEX_FILE):
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_FILE)

            with Session(engine) as session:
                jobs = session.execute(select(Job)).scalars().all()
                job_id_to_index = [job.id for job in jobs]

            return
        except:
            pass

    rebuild_faiss_index()


load_or_build_index()

# -----------------------------------------------------
# UTILITIES
# -----------------------------------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in stop_words]
    return " ".join(tokens)


def extract_text_from_pdf(path: str) -> str:
    try:
        return extract_text(path)
    except:
        return ""


def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, convert_to_numpy=True)
    vecs = vecs.astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


# -----------------------------------------------------
# ROUTES
# -----------------------------------------------------
@app.route("/")
def home():
    return "CV Analyzer API (PDF + Transformers + Neon DB) is running."


# ------------------- ADD JOB -------------------------
@app.route("/jobs", methods=["POST"])
def add_job():
    data = request.json
    title = data.get("title")
    description = data.get("description")

    if not title or not description:
        return jsonify({"error": "title and description required"}), 400

    text = preprocess_text(description)
    vec = embed_texts([text])[0]

    with Session(engine) as session:
        job = Job(
            title=title,
            description=description,
            embedding=vec.tobytes()
        )
        session.add(job)
        session.commit()
        new_id = job.id

    rebuild_faiss_index()

    return jsonify({"status": "ok", "job_id": new_id})


# ------------------- LIST JOBS -------------------------
@app.route("/jobs", methods=["GET"])
def list_jobs():
    with Session(engine) as session:
        jobs = session.execute(select(Job)).scalars().all()

    return jsonify([
        {"id": j.id, "title": j.title, "description": j.description[:400]}
        for j in jobs
    ])


# ------------------- ANALYZE CV -------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    # Accept File
    if "file" in request.files:
        f = request.files["file"]

        filename = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)

        if filename.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(path)
        else:
            raw_text = open(path, "r", encoding="utf-8", errors="ignore").read()
    else:
        data = request.get_json(silent=True) or {}
        raw_text = data.get("cv", "")
        if not raw_text:
            return jsonify({"error": "No CV provided"}), 400

    pre = preprocess_text(raw_text)
    vec = embed_texts([pre])[0]

    top_k = int(request.args.get("k", 5))

    if faiss_index is None or faiss_index.ntotal == 0:
        return jsonify({"matches": []})

    D, I = faiss_index.search(np.expand_dims(vec, axis=0), top_k)
    scores = D[0].tolist()
    indices = I[0].tolist()

    response = {"cv_text_snippet": raw_text[:800], "matches": []}

    with Session(engine) as session:
        for idx, score in zip(indices, scores):
            job = session.execute(select(Job).filter(Job.id == idx)).scalar_one_or_none()
            if job:
                response["matches"].append({
                    "job_id": job.id,
                    "title": job.title,
                    "score": score,
                    "description": job.description[:400]
                })

    return jsonify(response)


# -----------------------------------------------------
# START SERVER
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
