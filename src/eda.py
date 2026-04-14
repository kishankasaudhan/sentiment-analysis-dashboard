import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# =========================
# 1. Load Dataset
# =========================
print("🚀 Loading dataset...")
df = pd.read_csv("data/IMDB Dataset.csv")
print("✅ Dataset loaded\n")

# =========================
# 2. Basic Info
# =========================
print("🔍 Dataset Info:")
print(df.info())

print("\n📊 First 5 rows:")
print(df.head())

# =========================
# 3. Missing Values
# =========================
print("\n❓ Missing Values:")
print(df.isnull().sum())

# =========================
# 4. Class Distribution
# =========================
print("\n📊 Sentiment Distribution:")
print(df['sentiment'].value_counts())

df['sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# =========================
# 5. Review Length Analysis
# =========================
df['review_length'] = df['review'].apply(len)

print("\n📏 Review Length Stats:")
print(df['review_length'].describe())

df['review_length'].hist(bins=50)
plt.title("Review Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()

# =========================
# 6. Most Common Words
# =========================
print("\n🔤 Most Common Words:")

all_words = " ".join(df['review']).split()
common_words = Counter(all_words).most_common(10)

for word, count in common_words:
    print(f"{word}: {count}")

print("\n✅ EDA Completed!")