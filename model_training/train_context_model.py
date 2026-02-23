import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("data/context_dataset_final.csv")

X = data["text"]
y = data["context"]

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 3. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# 4. Train Linear SVM
# -----------------------------
model = LinearSVC(class_weight="balanced")

model.fit(X_train_tfidf, y_train)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 6. Save Model
# -----------------------------
joblib.dump(model, "models/context_model_svm.pkl")
joblib.dump(vectorizer, "models/context_vectorizer.pkl")

print("\nContext model and vectorizer saved successfully.")