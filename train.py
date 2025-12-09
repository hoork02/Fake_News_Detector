import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from preprocess import clean_text

# Load dataset
df = pd.read_csv("News.csv")

# Drop useless column
df = df.drop(columns=["Unnamed: 0"])

# Combine title + text
df["content"] = df["title"] + " " + df["text"]

# Rename 'class' to 'label' for consistency
df["label"] = df["class"]

# Clean text
df["content"] = df["content"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ------------------ MODELS ------------------

models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    #"SVM": LinearSVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# ------------------ SAVE BEST MODEL ------------------

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

joblib.dump(best_model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print(f"\n Best Model Saved: {best_model_name}")

# ------------------ CONFUSION MATRIX ------------------

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Best Model")
plt.show()

plt.pause(5)
plt.close()
