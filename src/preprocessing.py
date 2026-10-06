import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.data.path.append("./nltk_data")

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    filtered = []
    for token in tokens:
        if token.isalnum():
            filtered.append(token)

    filtered = [t for t in filtered if t not in STOP_WORDS and t not in string.punctuation]

    stemmed = [STEMMER.stem(t) for t in filtered]

    return " ".join(stemmed)