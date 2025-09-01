import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load main dataset
main_file = "fake_or_real_news.csv"
df_main = pd.read_csv(main_file)

# Load custom dataset if available
custom_file = "my_data.csv"
if os.path.exists(custom_file):
    df_custom = pd.read_csv(custom_file)
    print(f"✅ Loaded custom dataset with {len(df_custom)} rows")
    df = pd.concat([df_main, df_custom], ignore_index=True)
else:
    print("⚠️ No custom dataset found, training only on main dataset")
    df = df_main

# Features & labels
X = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Accuracy check
accuracy = model.score(X_test_vec, y_test)
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# Save model and vectorizer
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
