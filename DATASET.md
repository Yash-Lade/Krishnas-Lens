# 📊 Dataset Documentation – Krishna’s Lens

This document provides a complete overview of all datasets used in the **Krishna’s Lens** project, including their structure, purpose, preprocessing, and usage within the ML + RAG pipeline.

---

## Overview

The project uses **three primary datasets**:

1. **Emotion Dataset** → for emotion classification
2. **Context Dataset** → for life-context classification
3. **Bhagavad Gita Dataset** → for RAG-based knowledge retrieval

---

# Emotion Dataset

## Purpose

Used to train the **Emotion Classification Model** that detects the emotional state of the user from input text.

## File

```
data/emotion_dataset_final.csv
```

## 📂 Structure

| Column  | Description     |
| ------- | --------------- |
| text    | User input text |
| emotion | Emotion label   |

## 🏷️ Emotion Classes

```
anger
comparison
confusion
fear
pressure
sadness
self_doubt
```

## 🛠️ Data Preparation

* Combined multiple open-source datasets (e.g., Twitter emotion datasets)
* Normalized labels into unified categories
* Removed noisy/duplicate entries
* Balanced dataset across classes (~equal samples per class)

## ⚙️ Usage

* Model: **SVM (LinearSVC)**
* Vectorizer: **TF-IDF**
* Output used in:

  * Query enrichment
  * RAG retrieval guidance

---

# Context Dataset

## Purpose

Used to classify the **life domain/context** of the user's problem.

## File

```
data/context_dataset.csv
```

## 📂 Structure

| Column  | Description     |
| ------- | --------------- |
| text    | User input text |
| context | Context label   |

## 🏷️ Context Classes

```
academics
family
peer_pressure
future_career
self_identity
```

## 🛠️ Data Preparation

* Manually curated dataset based on real-life student scenarios
* Designed to reflect common teenage challenges
* Ensured balanced representation across all classes

## ⚙️ Usage

* Model: **SVM (LinearSVC)**
* Vectorizer: **TF-IDF**
* Output used in:

  * Query construction
  * Retrieval relevance improvement

---

# Bhagavad Gita Dataset

## Purpose

Serves as the **knowledge base** for the RAG (Retrieval-Augmented Generation) system.

## File

```
data/gita_verses.csv
data/gita_cleaned.csv
```

## 📂 Structure

| Column  | Description            |
| ------- | ---------------------- |
| chapter | Chapter number         |
| verse   | Verse number           |
| text    | Original verse text    |
| meaning | Simplified explanation |

## 🛠️ Preprocessing

* Removed null values
* Cleaned and normalized text (lowercasing, trimming)
* Combined fields for embedding:

  ```
  text + meaning
  ```
* Generated embeddings using:

  * **SentenceTransformer (all-MiniLM-L6-v2)**

## 📦 Derived Artifacts

| File                | Description        |
| ------------------- | ------------------ |
| gita_embeddings.npy | Vector embeddings  |
| gita_index.faiss    | FAISS vector index |

---

## 🔍 Usage in RAG Pipeline

1. User query is constructed and enriched
2. Converted into embedding
3. FAISS retrieves top **K candidates (≈50)**
4. Cross-Encoder re-ranks candidates
5. Top 3 verses are selected

---

# 🔗 Dataset Flow in System

```text
User Input
   ↓
Emotion Dataset → Emotion Model
   ↓
Context Dataset → Context Model
   ↓
Query Enrichment
   ↓
Gita Dataset → Embeddings → FAISS → Retrieval
```

---

# 📊 Dataset Quality Considerations

### Emotion Dataset

* Some overlap between classes (e.g., sadness vs self_doubt)
* Realistic ambiguity preserved intentionally

### Context Dataset

* High accuracy due to clear domain separation
* Limited classes ensure strong classification performance

### Gita Dataset

* Philosophical language differs from modern user queries
* Query enrichment required to bridge semantic gap

---

# ⚠️ Limitations

* Emotion detection limited to predefined categories
* Context classification may not capture multi-domain problems
* Gita interpretations depend on embedding quality
* Small dataset size (~700 verses) limits retrieval diversity

---

# 🚀 Future Improvements

* Expand emotion classes using deep learning (BERT)
* Add multi-label context classification
* Use domain-specific embeddings for spiritual texts
* Include commentary from multiple Gita interpretations

---

# 🙏 Acknowledgment

The Bhagavad Gita dataset is used as a **philosophical knowledge base** to provide reflective guidance. Interpretations are simplified for accessibility and educational purposes.

---
