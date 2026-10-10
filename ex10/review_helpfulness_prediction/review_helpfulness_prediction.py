import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Sample dataset (demonstration only)
data = {
    "review": [
        "This product is very good and useful",
        "Excellent quality and worth the money",
        "Very helpful review",
        "Amazing product and great quality",
        "This product is bad and useless",
        "Poor quality and not worth buying",
        "Terrible product and waste of money",
        "Not useful at all",
        "Good product with excellent features",
        "Bad experience with this product",
        "The review explains the product clearly",
        "This product is disappointing",
    ],
    "helpful": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)
X = df["review"]
y = df["helpful"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression()),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

new_review = ["This review clearly explains the product quality"]
prediction = model.predict(new_review)
print("Review Prediction:", "Helpful" if prediction[0] == 1 else "Not Helpful")
