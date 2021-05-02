import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from feature_extraction import *
from preprocess import preprocess_data
from utils import split_train_test
import scipy.sparse as sp


def bag_of_words_features(train_data, test_data, max_features=2000, binary=False):
    """Return features using bag of words"""
    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=3, stop_words='english', binary=binary)

    joined_train_data = train_data["lemas"].apply(" ".join)
    joined_test_data = test_data["lemas"].apply(" ".join)

    X_train = vectorizer.fit_transform(joined_train_data)

    X_train = X_train.astype("float16")
    X_test = vectorizer.transform(joined_test_data)

    X_test = X_test.astype("float16")
    return X_train, X_test


def tfidf_features(train_data, test_data):
    """Return features using TFIDF"""
    joined_train_data = train_data["lemas"].apply(" ".join)
    joined_test_data = test_data["lemas"].apply(" ".join)
    vectorizer = TfidfVectorizer(
        token_pattern=r"\w{1,}",
        min_df=0.2,
        max_df=0.8,
        use_idf=True,
        binary=True,
        ngram_range=(1, 3)
    )
    X_train = vectorizer.fit_transform(joined_train_data)
    X_test = vectorizer.transform(joined_test_data)
    return X_train, X_test


def count_words(tokens):
    return len(tokens)


def longest_word(tokens):
    return max(list(map(len, tokens)))


def shortest_word(tokens):
    return min(list(map(len, tokens)))


def custom_features(train_data, test_data):
    v = DictVectorizer()
    features = []
    first = True
    for i, tokens in enumerate(train_data):
        if len(tokens) > 0:
            item = {"count_words": count_words(tokens), "longest_word": longest_word(tokens),
                    "shortest_word": shortest_word(tokens)}
            features.append(item)

    # joined_train_data = train_data["lemas"].apply(" ".join)
    # joined_test_data = test_data["lemas"].apply(" ".join)
    #
    # vectorizer1 = CountVectorizer(min_df=2, ngram_range=(1, 2))
    # bow_count = vectorizer1.fit_transform(joined_train_data)
    # bow_transform = vectorizer1.transform(joined_test_data)
    #
    # vectorizer2 = TfidfVectorizer()
    # pos_tfidf = vectorizer2.fit_transform(joined_train_data)
    # tfidf_transform = vectorizer1.transform(joined_test_data)
    #
    # X_train = sp.vstack((bow_count, pos_tfidf, v.fit_transform(features)))
    # X_test = sp.vstack((bow_transform, tfidf_transform, v.transform(features)))
    X_train = v.fit_transform(features)
    X_test = v.transform(features)
    return X_train, X_test


if __name__ == "__main__":
    data = preprocess_data()
    X_train, X_test, y_train, y_test = split_train_test(data, x_col='lemas')

    custom_features(X_train, X_test)
