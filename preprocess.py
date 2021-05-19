import pandas as pd
import nltk
import string
import contractions
from nltk.corpus import stopwords

from constants import emoticons

nltk.download("punkt")
nltk.download("wordnet")


def read_crew_data():
    df = pd.read_csv("data/crew_data_discussion_only.csv")
    return df


def preprocess_data():
    df = read_crew_data()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # df['lemmas'] = (
    #     df["Message"]
    #     .str.lower()
    #     .apply(contractions.fix)
    #     .apply(
    #         lambda s: s.translate(
    #             str.maketrans({key: None for key in string.punctuation})
    #         )
    #     )
    #     .apply(nltk.word_tokenize)
    #     .apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
    # )

    def check_for_dots(tokens):
        if len(tokens) > 1:
            return list(map(lambda token: token.replace(".", ""), tokens))
        return tokens

    df["Message"] = (
        df["Message"]
        .str.strip()
        .str.replace("\n", " ")
        .str.replace("\r", " ")
        .str.replace("’", "'")
        .str.replace("…", "...")
        .str.replace("“", "'")
        .str.replace("—", " ")
    )
    df["tokens"] = (
        df["Message"]
        .str.replace("/", " ")
        .str.replace("-", " ")
        .apply(contractions.fix)
        .str.lower()
        .str.split(" ")
    )

    df["tokens"] = (
        df["tokens"]
        .apply(
            lambda tokens: [
                x.translate(str.maketrans({key: None for key in string.punctuation}))
                if x not in emoticons
                else x
                for x in tokens
            ]
        )
        .apply(check_for_dots)
        .apply(lambda tokens: list(filter(lambda token: token != "", tokens)))
    )

    df["lemmas"] = df["tokens"].apply(
        lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
    )
    stop = stopwords.words("english")
    df["no_stopwords"] = df["lemmas"].apply(
        lambda lema: [item for item in lema if item not in stop]
    )
    df["joined_lemmas"] = df["lemmas"].apply(" ".join)

    return df
