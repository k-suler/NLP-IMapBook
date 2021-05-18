import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from constants import emoticons
from feature_extraction import *
from preprocess import preprocess_data
from utils import split_train_test
import scipy.sparse as sp
import functools
import re

def bag_of_words_features(train_data, test_data, max_features=2000, binary=False):
    """Return features using bag of words"""
    vectorizer = CountVectorizer(
        ngram_range=(1, 3), min_df=3, stop_words='english', binary=binary
    )

    joined_train_data = train_data["lemas"].apply(" ".join)
    joined_test_data = test_data["lemas"].apply(" ".join)

    X_train = vectorizer.fit_transform(joined_train_data)

    X_train = X_train.astype("float16")
    X_test = vectorizer.transform(joined_test_data)

    X_test = X_test.astype("float16")
    return X_train, X_test


def tfidf_features(train_data, test_data, binary=False):
    """Return features using TFIDF"""
    joined_train_data = train_data["lemas"].apply(" ".join)
    joined_test_data = test_data["lemas"].apply(" ".join)
    vectorizer = TfidfVectorizer(
        token_pattern=r"\w{1,}",
        min_df=0.2,
        max_df=0.8,
        use_idf=True,
        binary=binary,
        ngram_range=(1, 3),
    )
    X_train = vectorizer.fit_transform(joined_train_data)
    X_train = X_train.astype("float16")
    X_test = vectorizer.transform(joined_test_data)
    X_test = X_test.astype("float16")
    return X_train, X_test


def bag_of_words_features_1(train_data, test_data, max_features=2000, binary=False, kfold=False):
    """Return features using bag of words"""
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english', binary=binary)

    if not kfold:
        joined_train_data = train_data["lemas"].apply(" ".join)
        joined_test_data = test_data["lemas"].apply(" ".join)
    else:
        joined_train_data = train_data
        joined_test_data = test_data

    X_train = vectorizer.fit_transform(joined_train_data)

    X_train = X_train.astype("float16")
    X_test = vectorizer.transform(joined_test_data)

    X_test = X_test.astype("float16")
    return X_train, X_test


def tfidf_features_1(train_data, test_data, kfold):
    """Return features using TFIDF"""
    if not kfold:
        joined_train_data = train_data["lemas"].apply(" ".join)
        joined_test_data = test_data["lemas"].apply(" ".join)
    else:
        joined_train_data = train_data
        joined_test_data = test_data
    vectorizer = TfidfVectorizer(
        analyzer='word',
        max_features=200000,
        token_pattern=r"\w{1,}",
        use_idf=True,
        sublinear_tf=True,
        ngram_range=(1, 2)
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


def count_emoticons(lemas):
    number_of_emoticons = 0
    for emoticon in emoticons:
        if emoticon in lemas:
            number_of_emoticons += 1
    return number_of_emoticons


from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tag import untag, str2tuple, tuple2str
from nltk.chunk import tree2conllstr, conllstr2tree, conlltags2tree, tree2conlltags


def read_book(filename):
    import pdb
    with open(filename, "r", encoding="UTF-8") as f:
        pdb.set_trace()
        rl = f.readlines()
        rl = " ".join(rl)
        rl = rl.replace('\n', '')
        tokens = word_tokenize(rl)
        tagged_tokens = pos_tag(tokens)
        ner_tree = ne_chunk(tagged_tokens)
        iob_tagged = tree2conlltags(ner_tree)
        print(iob_tagged)
        f.close()

def is_url(s):
    return len(re.findall(r'(https?://[^\s]+)', s)) > 0

def custom_features(data):
    v = DictVectorizer()
    features = []

    data["message_length"] = data['Message'].apply(len)
    data["longest_word"] = data["lemas"].apply(max).apply(len)
    data["shortest_word"] = data["lemas"].apply(min).apply(len)
    data["num_of_words"] = data["lemas"].apply(len)
    data["contains_question_marks"] = data["Message"].str.contains("\?").apply(int)
    data["num_of_question_marks"] = data["Message"].str.count("\?")
    data["contains_exclamation_point"] = data["Message"].str.contains("\!").apply(int)
    data["num_of_exclamation_point"] = data["Message"].str.count("\!")
    data["num_of_emoticons"] = data["lemas"].apply(count_emoticons)
    data["is_url"] = data["Message"].apply(is_url)

    read_book("data/ID260 and ID261 - The Lady or the Tiger.txt")
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
