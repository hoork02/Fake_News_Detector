import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)     # Remove punctuation & numbers
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(text)
