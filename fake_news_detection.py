# Fake News Detection using TF-IDF + Logistic Regression
# Loads dataset directly from CSV file

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------------------
# 1. Load Dataset
# ---------------------------
# Replace with your actual dataset path
df = pd.read_csv("fake_news_dataset.csv")  # Example: 'news.csv'

# Ensure required columns exist
if not {"text", "label"}.issubset(df.columns):
    raise ValueError("Dataset must contain 'text' and 'label' columns.")

# ---------------------------
# 2. Split Data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ---------------------------
# 3. Build & Train Model
# ---------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
])

model.fit(X_train, y_train)

# ---------------------------
# 4. Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# 5. Predict New Samples
# ---------------------------
sample_news = [
    "Scientists publish breakthrough research on climate change.",
    "Miracle cure claims to reverse aging overnight."
]
predictions = model.predict(sample_news)

for text, label in zip(sample_news, predictions):
    print(f"'{text}' -> {'Fake' if label == 1 else 'Real'}")