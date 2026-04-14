import pandas as pd
from preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pickle

# =========================
# 1. Load Dataset
# =========================
print("🚀 Loading dataset...")
df = pd.read_csv("data/IMDB Dataset.csv")
print("✅ Dataset loaded")

# OPTIONAL: speed up training
# df = df.sample(5000)

# =========================
# 2. Encode Labels
# =========================
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# =========================
# 3. Clean Text
# =========================
print("🧹 Cleaning text...")
df['clean_text'] = df['review'].apply(clean_text)
print("✅ Cleaning done")

# =========================
# 4. Feature Extraction
# =========================
print("🔢 Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['sentiment']
print("✅ Vectorization done")

# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. Build Model Function
# =========================
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

# =========================
# 7. K-Fold Validation
# =========================
print("🔁 K-Fold Validation...")
kf = KFold(n_splits=5)
accuracies = []

for train_index, val_index in kf.split(X):
    X_tr, X_val = X[train_index], X[val_index]
    y_tr, y_val = y.iloc[train_index], y.iloc[val_index]

    model = create_model(X.shape[1])
    model.fit(X_tr, y_tr, epochs=3, batch_size=32, verbose=0)

    preds = (model.predict(X_val) > 0.5).astype("int32")
    acc = accuracy_score(y_val, preds)
    accuracies.append(acc)

print("✅ Average K-Fold Accuracy:", np.mean(accuracies))

# =========================
# 8. Final Training
# =========================
print("🏋️ Training final model...")
model = create_model(X.shape[1])
model.fit(X_train, y_train, epochs=5, batch_size=32)

# =========================
# 9. Evaluation
# =========================
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)

print("🎯 Test Accuracy:", accuracy)

# =========================
# 10. Save Model
# =========================
model.save("model/model.h5")
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("💾 Model saved!")
print("✅ Training Complete!")