# ================= TRAINING =================
import re
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "I love this product",
    "This product is very bad",
    "Amazing quality and good service",
    "Worst experience ever"
]
labels = [1, 0, 1, 0]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

cleaned_texts = []

for sentence in texts:
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z\s]', '', sentence)
    tokens = word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    cleaned_texts.append(" ".join(words))  

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

joblib.dump(model, "my_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Ready for Prediction")
