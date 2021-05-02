import pandas as pd
import nltk
import string
import contractions

nltk.download("punkt")
nltk.download("wordnet")


def read_crew_data():
    df = pd.read_csv("data/crew_data_discussion_only.csv")
    return df


def preprocess_data():
    df = read_crew_data()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    emoticons = [":)", ":D", ":("]
    # df['lemas'] = (
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

    df["tokens"] = df["Message"].str.lower().apply(contractions.fix).str.split(" ")
    df["tokens"] = df["tokens"].apply(
        lambda tokens: [
            x.translate(str.maketrans({key: None for key in string.punctuation}))
            if x not in emoticons
            else x
            for x in tokens
        ]
    )
    df["lemas"] = df["tokens"].apply(
        lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
    )

    df["joined_lemas"] = df["lemas"].apply(" ".join)
    df["joined_lemas"] = df["joined_lemas"].str.strip()
    return df
