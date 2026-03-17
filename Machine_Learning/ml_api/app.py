from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import faiss
import os
import pandas as pd
import random

app = FastAPI()

# -----------------------------
# Request model
# -----------------------------
class ThoughtRequest(BaseModel):
    text: str


# -----------------------------
# Path setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_training", "models")
DATASET_DIR = os.path.join(BASE_DIR, "Datasets")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load ML models
# -----------------------------
emotion_model = joblib.load(os.path.join(MODEL_DIR, "emotion_model_svm.pkl"))
context_model = joblib.load(os.path.join(MODEL_DIR, "context_model_svm.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))


# -----------------------------
# Load FAISS index (optional)
# -----------------------------
try:
    index = faiss.read_index(os.path.join(MODEL_DIR, "gita_index.faiss"))
    faiss_enabled = True
    print("FAISS index loaded")
except:
    faiss_enabled = False
    print("FAISS index not available")


# -----------------------------
# Load Gita dataset
# -----------------------------
gita_df = pd.read_csv(os.path.join(DATASET_DIR, "gita_cleaned.csv"))
print("Dataset columns:", gita_df.columns)

verses = gita_df["text"].tolist()


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {"message": "Krishnas Lens ML API running"}


# -----------------------------
# Analyze Thought
# -----------------------------
@app.post("/analyze")
def analyze(data: ThoughtRequest):

    text = data.text

    # TF-IDF vector
    vec = vectorizer.transform([text])

    # Emotion prediction
    emotion = emotion_model.predict(vec)[0]

    # Context prediction
    context = context_model.predict(vec)[0]

    retrieved_verses = []

    # ---------------------------------
    # Try FAISS search
    # ---------------------------------
    if faiss_enabled:
        try:
            query_vector = vec.toarray().astype("float32")

            print("Vector shape:", query_vector.shape)
            print("Index dimension:", index.d)

            D, I = index.search(query_vector, k=3)

            retrieved_verses = [verses[i] for i in I[0] if i < len(verses)]

        except Exception as e:
            print("FAISS error:", str(e))

    # ---------------------------------
    # Fallback if FAISS fails
    # ---------------------------------
    if not retrieved_verses:
        retrieved_verses = random.sample(verses, 2)

    return {
        "emotion": emotion,
        "context": context,
        "verses": retrieved_verses
    }