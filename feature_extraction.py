import string

import nltk
import textblob as textblob
from nltk import word_tokenize, pos_tag, ne_chunk, tree2conlltags
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from constants import emoticons
from feature_extraction import *
from preprocess import preprocess_data
import scipy.sparse as sp
import re


def bag_of_words_features(data, binary=False):
    """Return features using bag of words"""
    vectorizer = CountVectorizer(
        ngram_range=(1, 3), min_df=3, stop_words="english", binary=binary
    )

    return vectorizer.fit_transform(
        data["joined_lemmas"]
    )


def tfidf_features(data, binary=False):
    """Return features using TFIDF"""
    vectorizer = TfidfVectorizer(
        token_pattern=r"\w{1,}",
        min_df=0.2,
        max_df=0.8,
        use_idf=True,
        binary=binary,
        ngram_range=(1, 3),
    )
    return vectorizer.fit_transform(
        data["joined_lemmas"]
    )


def bag_of_words_features_1(
        train_data, test_data, max_features=2000, binary=False, kfold=False
):
    """Return features using bag of words"""
    vectorizer = CountVectorizer(
        ngram_range=(1, 3), stop_words="english", binary=binary
    )

    if not kfold:
        joined_train_data = train_data["lemmas"].apply(" ".join)
        joined_test_data = test_data["lemmas"].apply(" ".join)
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
        joined_train_data = train_data["lemmas"].apply(" ".join)
        joined_test_data = test_data["lemmas"].apply(" ".join)
    else:
        joined_train_data = train_data
        joined_test_data = test_data
    vectorizer = TfidfVectorizer(
        analyzer="word",
        max_features=200000,
        token_pattern=r"\w{1,}",
        use_idf=True,
        sublinear_tf=True,
        ngram_range=(1, 2),
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


def count_emoticons(lemmas):
    number_of_emoticons = 0
    for emoticon in emoticons:
        if emoticon in lemmas:
            number_of_emoticons += 1
    return number_of_emoticons


def read_book(filename):
    with open(filename, "r", encoding="UTF-8") as f:
        rl = f.readlines()
        rl = " ".join(rl)
        rl = rl.replace("\n", "")
        rl = rl.replace("\ufeff", "")
        f.close()
        return rl


def get_iob(rl):
    tokens = list(filter(lambda token: token not in string.punctuation, map(lambda token: token.lower(), word_tokenize(rl))))

    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    stop = stopwords.words("english")
    no_stopwords = [item for item in lemmas if item not in stop]

    tagged_tokens = pos_tag(tokens)
    ner_tree = ne_chunk(tagged_tokens)
    iob_tagged = tree2conlltags(ner_tree)
    persons = list(filter(lambda x: "PERSON" in x[2], iob_tagged))
    return persons, no_stopwords


def is_url(s):
    return int(len(re.findall(r"(https?://[^\s]+)", s)) > 0)


def is_person_in_book(message, persons):
    return any(list([word in persons for word in message]))

def is_word_in_book(message, words):
    return any(list([word in words for word in message]))


def person_mentioned(data, iob):
    persons = list(map(lambda person: str(person[0]).lower(), iob))
    return data["lemmas"].apply(lambda message: is_word_in_book(message, persons))

def book_words(data, book_no_stopwords):
    return data["no_stopwords"].apply(lambda message: is_person_in_book(message, book_no_stopwords))


def count_tag_types(message, type):
    pos_tags = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }
    cnt = 0
    try:
        wiki = textblob.TextBlob(message)
        # print(wiki.tags)
        cnt = sum([1 if list(t)[1] in pos_tags[type] else 0 for t in wiki.tags])
    except:
        pass
    return cnt


def custom_features_extractor(data):
    data["message_length"] = data["Message"].apply(len)
    data["longest_word"] = data["lemmas"].apply(max).apply(len)
    data["shortest_word"] = data["lemmas"].apply(min).apply(len)
    data["num_of_words"] = data["lemmas"].apply(len)
    data["contains_question_marks"] = data["Message"].str.contains("\?").apply(int)
    data["num_of_question_marks"] = data["Message"].str.count("\?")
    data["contains_exclamation_point"] = data["Message"].str.contains("\!").apply(int)
    data["num_of_exclamation_point"] = data["Message"].str.count("\!")
    data["num_of_emoticons"] = data["lemmas"].apply(count_emoticons)
    data["is_url"] = data["Message"].apply(is_url)
    data['num_nouns'] = data['Message'].apply(lambda x: count_tag_types(x, 'noun'))
    data['num_verbs'] = data['Message'].apply(lambda x: count_tag_types(x, 'verb'))
    data['num_adjs'] = data['Message'].apply(lambda x: count_tag_types(x, 'adj'))
    data['num_advs'] = data['Message'].apply(lambda x: count_tag_types(x, 'adv'))
    data['num_prons'] = data['Message'].apply(lambda x: count_tag_types(x, 'pron'))

    book1 = read_book("data/ID260 and ID261 - The Lady or the Tiger.txt")
    book2 = read_book("data/ID264 and ID265 - Just Have Less.txt")
    book3 = read_book("data/ID266 and ID267 - Design for the Future When the Future Is Bleak.txt")

    book1_persons, book1_no_stopwords = get_iob(book1)
    book2_persons, book2_no_stopwords = get_iob(book2)
    book3_persons, book3_no_stopwords = get_iob(book3)

    # data["book1_persons_mentioned"] = person_mentioned(data, book1_persons)
    # data["book2_persons_mentioned"] = person_mentioned(data, book2_persons)
    # data["book3_persons_mentioned"] = person_mentioned(data, book3_persons)

    data["words_in_book1"] = book_words(data, book1_no_stopwords)
    data["words_in_book2"] = book_words(data, book2_no_stopwords)
    data["words_in_book3"] = book_words(data, book3_no_stopwords)

    tfidf = tfidf_features(data)
    bow = bag_of_words_features(data)

    scaler = MinMaxScaler()
    X_cols = scaler.fit_transform(data[data.columns[14:]])

    X_sparse = sp.hstack([sp.csr_matrix(tfidf), sp.csr_matrix(bow), sp.csr_matrix(X_cols)])

    return X_sparse


if __name__ == "__main__":
    print("Start")
    data = preprocess_data()
    custom_features_extractor(data)
