import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Load
data = pd.read_csv("mental_health_dataset.csv")
data = data.dropna(subset=["text", "label"])

label_map = {0: "Non-Stress", 1: "Stress", 2: "Depression", 3: "Suicidal"}
data["label"] = data["label"].map(label_map)

print("Class Distribution:")
print(data["label"].value_counts())

# Split
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes":         MultinomialNB(),
    "SVM":                 LinearSVC(max_iter=2000),
    "Random Forest":       RandomForestClassifier(n_estimators=100),
}

# Train & Evaluate
print("\nModel Accuracy:")
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)
    acc  = accuracy_score(y_test, pred)
    print(f"{name}: {acc:.2f}")

# Save best (Logistic Regression)
pickle.dump(vectorizer,           open("vectorizer.pkl", "wb"))
pickle.dump(models["Logistic Regression"], open("model.pkl", "wb"))

print("\nModels saved!")











