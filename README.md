# Krishnas-Lens

Krishna’s Lens is a hybrid ML + LLM system designed to help teenagers reframe personal challenges through structured perspectives. The platform uses classical machine learning models (TF-IDF + SVM) to detect a user’s dominant emotion and life context from free-text input. Based on these predictions, it retrieves relevant Bhagavad Gita principles and generates three distinct viewpoints — emotional, strategic, and spiritual. The system combines interpretable ML classification with controlled language generation to deliver reflective, non-prescriptive guidance.


## 🧠 System Architecture

```mermaid
flowchart TD

A[User Input<br>Student Problem] --> B[Emotion Detection Model<br>SVM + TF-IDF]

A --> C[Context Detection Model<br>SVM + TF-IDF]

B --> D[Query Builder]
C --> D

D --> E[Query Enrichment<br>Concept Mapping]

E --> F[Embedding Model<br>SentenceTransformer]

F --> G[Vector Search<br>FAISS Index]

G --> H[Top 50 Candidate Verses]

H --> I[Re-Ranking Model<br>CrossEncoder]

I --> J[Top 3 Relevant Bhagavad Gita Verses]

J --> K[Perspective Generator<br>LLM]

K --> L[Emotional Perspective]

K --> M[Strategic Perspective]

K --> N[Spiritual Perspective]

L --> O[Final Response to User]
M --> O
N --> O
```

