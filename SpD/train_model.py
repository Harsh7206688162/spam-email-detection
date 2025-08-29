import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import pickle

# =====================
# Load Dataset
# =====================
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

# =====================
# Text Cleaning Function
# =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)   # replace links
    text = re.sub(r"\d+", " number ", text)                   # replace numbers
    text = re.sub(r"[^\w\s]", " ", text)                      # remove punctuation
    return text.strip()

df["text"] = df["text"].apply(clean_text)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# =====================
# Balance Dataset (Upsample Spam)
# =====================
ham = df[df.label == 0]
spam = df[df.label == 1]

spam_upsampled = resample(spam,
                          replace=True,
                          n_samples=len(ham),
                          random_state=42)

df_balanced = pd.concat([ham, spam_upsampled])

print("✅ Dataset balanced:")
print(df_balanced["label"].value_counts())

# =====================
# Train-Test Split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["text"], df_balanced["label"], test_size=0.2, random_state=42
)

# =====================
# TF-IDF Vectorizer with unigrams + bigrams
# =====================
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =====================
# Train Logistic Regression Model
# =====================
model = LogisticRegression(max_iter=300, class_weight="balanced")
model.fit(X_train_vec, y_train)

# =====================
# Evaluate Model with custom threshold
# =====================
y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
y_pred = (y_pred_proba >= 0.4).astype(int)  # mark as spam if prob ≥ 40%

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =====================
# Save Model and Vectorizer
# =====================
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n🎉 Model & Vectorizer saved successfully as spam_model.pkl and vectorizer.pkl!")
