#import fastapi
from fastapi import FastAPI
# optional --> on top we will build api's
from pydantic import BaseModel
# load pkl files
import joblib
# import regular expression
import re
# import nlp library
import nltk
# remove stopwords like is an i am
from nltk.corpus import stopwords
# statement --> words
from nltk.tokenize import word_tokenize
# words --> dictionary level words
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

model = joblib.load("my_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text:str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]','',text)
    tokens = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(words)

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message":"NLP FastAPI is running"}

@app.post("/predict")
def predict_sentiment(data:TextInput):
    cleaned_text = preprocess_text(data.text)
    vectorized_text =  vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    print(prediction)
    result = "Positive " if prediction[0] == 1 else "Negative " 
    return {
        "input_text" : data.text,
        "sentiment" : result
    }





