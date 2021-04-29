import pandas as pd
import nltk
import string

nltk.download("punkt")
nltk.download('wordnet')


def read_crew_data():
    df = pd.read_csv("data/crew-data.csv")
    return df


def preprocess_data():
    df = read_crew_data()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df['lemas'] = (
        df["Message"]
        .str.lower()
        .apply(
            lambda s: s.translate(
                str.maketrans({key: None for key in string.punctuation})
            )
        )
        .apply(nltk.word_tokenize)
        .apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
    )
    return df


preprocess_data()
