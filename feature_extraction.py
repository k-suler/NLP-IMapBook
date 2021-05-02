from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bag_of_words_features(train_data, test_data, max_features=2000, binary=False):
    """Return features using bag of words"""
    vectorizer = CountVectorizer(
        binary=binary, max_df=0.6, min_df=2, stop_words="english"
    )
    X_train = vectorizer.fit_transform(train_data)

    X_train = X_train.astype("float16")
    X_test = vectorizer.transform(test_data)

    X_test = X_test.astype("float16")
    return X_train, X_test


def tfidf_features(train_data, test_data, max_df=0.5):
    """Return features using TFIDF"""
    vectorizer = TfidfVectorizer(analyzer='word',
                                 max_features=200000,
                                 token_pattern=r"\w{1,}",
                                 use_idf=True,
                                 sublinear_tf=True,
                                 ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_test