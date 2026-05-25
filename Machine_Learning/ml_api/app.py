from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
import joblib
import numpy as np
import faiss
import os
import pandas as pd
import random
import json

from dotenv import load_dotenv
from groq import Groq

# This finds the directory where app.py lives
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment. Check your .env file.")
client = Groq(api_key=api_key)

app = FastAPI()

# sentence transformer for embedding the query to vector
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
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

verses = gita_df.to_dict(orient="records")


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
    text = data.text

    vec = vectorizer.transform([text])

    emotion = emotion_model.predict(vec)[0]
    context = context_model.predict(vec)[0]

    retrieved_verses = []

    if faiss_enabled:
        try:
            # query_vector = vec.toarray().astype("float32")
            query_vector = embedder.encode([text]).astype("float32") # embedding using sentence transformer
            D, I = index.search(query_vector, k=3)
            retrieved_verses = [verses[i] for i in I[0] if i < len(verses)]
        except Exception as e:
            print("FAISS error:", str(e))

    if not retrieved_verses:
        retrieved_verses = random.sample(verses, min(3, len(verses)))

    perspectives = generate_perspectives(
        text, emotion, context, retrieved_verses
    )

    return {
        "emotion": emotion,
        "context": context,
        "perspectives": perspectives
    }
def build_verse_context(verses):

    formatted_verses = []

    # Handle both DataFrame and list
    if hasattr(verses, "iterrows"):
        iterable = verses.iterrows()
    else:
        iterable = enumerate(verses)

    for i, item in enumerate(iterable, start=1):

        if hasattr(verses, "iterrows"):
            _, v = item
        else:
            v = item[1]

        formatted_verses.append(
            f"[V{i}] (Ch {v['chapter']}, Verse {v['verse']}): "
            f"{v['text']} | Meaning: {v['meaning']}"
        )

    return "\n\n".join(formatted_verses)


def build_prompt(user_text, emotion, context, verse_context):

    prompt = f"""
        A teenager is facing the following situation:

        User Situation:
        {user_text}

        Detected Emotion:
        {emotion}

        Life Context:
        {context}

        Relevant Bhagavad Gita verses:
        {verse_context}

        ---

        STRICT RULES:
        - Use ONLY the provided verses.
        - Do NOT introduce any external knowledge.
        - MUST cite verses using IDs like [V1], [V2]
        - Stay grounded in the given verses only.

        ---

        OUTPUT FORMAT (STRICT JSON, NO EXTRA TEXT):

        {{
        "emotional": "...",
        "strategic": "...",
        "spiritual": "..."
        }}

        ---

        INSTRUCTIONS:

        Emotional Perspective:
        - Acknowledge the feeling briefly
        - Do NOT over-sympathize
        - Maintain composure and clarity

        Strategic Perspective:
        - Focus on right action (dharma)
        - What should be done, not what feels good

        Spiritual Perspective:
        - Explain detachment, control, and inner stability
        - Show how perspective changes the problem

        ---

        Tone:
        - calm
        - grounded
        - mentor-like
        - NOT preachy
        - NOT generic
        """
    return prompt

def generate_perspectives(user_text, emotion, context, verses):

    verse_context = build_verse_context(verses)
    prompt = build_prompt(user_text, emotion, context, verse_context)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            # {
            #     "role": "system",
            #     "content": "You are Krishna's Lens, a rational and reflective guide. You strictly follow instructions and always return valid JSON."
            # },
            {
                "role": "system",
                "content": """
                You are Krishna's Lens.

                You do NOT behave like a therapist or motivational speaker.

                You speak with:
                - clarity
                - calm authority
                - philosophical depth

                Your guidance is:
                - detached, not emotional
                - insightful, not comforting
                - rooted in wisdom, not sympathy

                You do NOT:
                - over-validate emotions
                - use clichés
                - sound like modern self-help advice

                You interpret situations through dharma, detachment, and clarity.

                Always stay grounded in the provided verses.
                Always return valid JSON.
                """
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    raw_output = response.choices[0].message.content.strip()

    # Safe parsing
    try:
        parsed = json.loads(raw_output)
    except:
        # fallback if LLM messes up
        parsed = {
            "emotional": raw_output,
            "strategic": "",
            "spiritual": ""
        }

    return parsed