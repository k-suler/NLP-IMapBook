from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bag_of_words_features(train_data, test_data, max_features=2000):
    """Return features using bag of words"""
    vectorizer = CountVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_test


def tfidf_features(train_data, test_data, max_df=0.5):
    """Return features using TFIDF"""
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=max_df, stop_words='english')
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_test
