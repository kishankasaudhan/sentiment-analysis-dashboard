import pickle
from tensorflow.keras.models import load_model
from src.preprocessing import clean_text

# =========================
# Load model and vectorizer
# =========================
model = load_model("model/model.h5")
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# =========================
# Prediction function
# =========================
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(vector)[0][0]

    if pred > 0.5:
        return "Positive 😊"
    else:
        return "Negative 😞"

# =========================
# Manual testing
# =========================
if __name__ == "__main__":
    print("🎬 Sentiment Analysis Tester")
    while True:
        text = input("\nEnter a review (or type 'exit'): ")
        
        if text.lower() == "exit":
            break
        
        result = predict_sentiment(text)
        print("Prediction:", result)