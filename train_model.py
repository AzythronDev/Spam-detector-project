import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data.rename(columns={'v1': 'label', 'v2': 'text'})  # Rename if v1, v2 exist

data.columns = ['label', 'text']

# Encode labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features and labels
X = data['text']
y = data['label']

# Vectorize text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved.")
