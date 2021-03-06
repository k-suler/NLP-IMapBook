import os
import warnings

import numpy as np
from bcolors import BLUE, ENDC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import keras
import utils
from baseline_models import LinearSVM, Model, NaiveBayes, PopularityModel, RandomModel
from basic_nn import NN
from feature_extraction import (
    bag_of_words_features_1,
    tfidf_features_1,
    custom_features_extractor,
    bag_of_words_features2,
    tfidf_features2,
)
from preprocess import preprocess_data
from utils import get_classes, preprocess_labels, split_train_test

warnings.filterwarnings("ignore")

pm = PopularityModel()
rm = RandomModel()


def custom_features(features, data):
    X_train, X_test, y_train, y_test = split_train_test(
        features, x_col="features", y=data[["CodePreliminary"]], stratify=True
    )
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    print(f"{BLUE} Starting with Naive Bayes classifier {ENDC}")
    nb = Model("Naive Bayes", MultinomialNB())
    nb.evaluate(X_train, y_train, X_test, y_test)

    nb.kfold(features, data, 'custom')
    # nb.loocv(features, data, 'custom')

    print(f"{BLUE} Starting with Support Vector Machine classifier {ENDC}")
    svm = Model("Support Vector Machine", SVC(kernel="rbf", gamma="auto", C=1000))
    svm.evaluate(X_train, y_train, X_test, y_test)

    svm.kfold(features, data, 'custom')
    # svm.loocv(features, data, 'custom')

    print(f"{BLUE} Starting with Logistic Regression classifier {ENDC}")
    lg = Model(
        "Support Vector Machine",
        LogisticRegression(multi_class="multinomial", max_iter=100, C=1.0 / 0.01),
    )
    lg.evaluate(X_train, y_train, X_test, y_test)

    lg.kfold(features, data, 'custom')
    # lg.loocv(features, data, 'custom')

    print(f"{BLUE} Starting with Popularity classifier {ENDC}")
    pm.evaluate(y_train, y_test)

    print(f"{BLUE} Starting with Random classifier {ENDC}")
    rm.evaluate(y_train, y_test)

    print(f"{BLUE} Starting with NN {ENDC}")
    lb = LabelEncoder()
    lb.fit(get_classes(data).tolist())

    Y_test_classes = y_test
    Y_train = lb.transform(y_train["CodePreliminary"].tolist())
    Y_train = keras.utils.to_categorical(Y_train)
    Y_test = lb.transform(y_test["CodePreliminary"].tolist())
    Y_test = keras.utils.to_categorical(Y_test)

    nn = NN("basic", lb)

    nn.train(
        X_train,
        Y_train,
        None,
        None,
        validation_split=True,
        filename="model-custom",
        nb_classes=len(get_classes(data).tolist()),
    )
    nn.evaluate(X_test, Y_test, Y_test_classes)

    print(f"{BLUE} Starting with MLP {ENDC}")
    mlp = NN("mlp", lb)
    mlp.train(
        X_train,
        Y_train,
        None,
        None,
        validation_split=True,
        filename="model-custom",
        nb_classes=len(get_classes(data).tolist()),
    )
    mlp.evaluate(X_test, Y_test, Y_test_classes)

    print(f"{BLUE} Starting with Deep NN {ENDC}")
    deep = NN("deep", lb)
    deep.train(
        X_train,
        Y_train,
        None,
        None,
        validation_split=True,
        filename="model-custom",
        nb_classes=len(get_classes(data).tolist()),
    )
    deep.evaluate(X_test, Y_test, Y_test_classes)


def bow(data):
    _X_train, _X_test, y_train, y_test = split_train_test(
        data, x_col="features", y=data[["CodePreliminary"]]
    )
    X_train, X_test = bag_of_words_features_1(_X_train, _X_test, kfold=False)

    print(f"{BLUE} Starting with Naive Bayes classifier {ENDC}")
    nb = Model("Naive Bayes", MultinomialNB())
    nb.evaluate(X_train, y_train, X_test, y_test)

    nb.kfold(None, data, 'bow')
    # nb.loocv(data, False)

    print(f"{BLUE} Starting with Support Vector Machine classifier {ENDC}")
    params_grid = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    ]
    # svm_find_best = Model(
    #     "Support Vector Machine", GridSearchCV(SVC(), params_grid, cv=10)
    # )
    # bets_score, best_C, best_kernel, best_gamma = svm_find_best.find_best_parameters(
    #     X_train, y_train
    # )
    svm = Model("Support Vector Machine", SVC(kernel="rbf", gamma="auto", C=1000))
    svm.evaluate(X_train, y_train, X_test, y_test)

    svm.kfold(None, data, 'bow')
    # svm.loocv(data, False)

    print(f"{BLUE} Starting with Logistic Regression classifier {ENDC}")
    lg = Model(
        "Support Vector Machine",
        LogisticRegression(multi_class="multinomial", max_iter=1000, C=1.0 / 0.01),
    )
    lg.evaluate(X_train, y_train, X_test, y_test)

    lg.kfold(None, data, 'bow')
    # lg.loocv(data, False)

    print(f"{BLUE} Starting with Popularity classifier {ENDC}")
    pm.evaluate(y_train, y_test)

    print(f"{BLUE} Starting with Random classifier {ENDC}")
    rm.evaluate(y_train, y_test)

    print(f"{BLUE} Starting with NN {ENDC}")
    X_train, X_test, Y_train, Y_test = split_train_test(
        data, x_col="features", y=data[["CodePreliminary"]]
    )
    X_train, X_val, Y_train, Y_val = split_train_test(
        X_train, x_col="features", y=Y_train[["CodePreliminary"]]
    )

    X_train, X_test, X_val = bag_of_words_features2(
        X_train,
        X_test,
        X_val,
        binary=True,
    )
    Y_test_classes = Y_test

    lb = LabelEncoder()
    lb.fit(get_classes(data).tolist())
    Y_train = lb.transform(Y_train["CodePreliminary"].tolist())
    Y_train = keras.utils.to_categorical(Y_train)

    Y_val = lb.transform(Y_val["CodePreliminary"].tolist())
    Y_val = keras.utils.to_categorical(Y_val)

    Y_test = lb.transform(Y_test["CodePreliminary"].tolist())
    Y_test = keras.utils.to_categorical(Y_test)

    nn = NN("basic", lb)
    nn.train(
        X_train,
        Y_train,
        X_val,
        Y_val,
        filename="model-bow",
        nb_classes=len(get_classes(data).tolist()),
    )
    nn.evaluate(X_test, Y_test, Y_test_classes)

    print(f"{BLUE} Starting with MLP {ENDC}")
    mlp = NN("mlp", lb)
    mlp.train(
        X_train,
        Y_train,
        X_val,
        Y_val,
        filename="model-bow",
        nb_classes=len(get_classes(data).tolist()),
    )
    mlp.evaluate(X_test, Y_test, Y_test_classes)

    print(f"{BLUE} Starting with Deep NN {ENDC}")
    deep = NN("deep", lb)
    deep.train(
        X_train,
        Y_train,
        X_val,
        Y_val,
        filename="model-bow",
        nb_classes=len(get_classes(data).tolist()),
    )
    deep.evaluate(X_test, Y_test, Y_test_classes)


def tfidf(data):
    _X_train, _X_test, y_train, y_test = split_train_test(
        data, x_col="features", y=data[["CodePreliminary"]]
    )
    X_train, X_test = tfidf_features_1(_X_train, _X_test, kfold=False)

    print(f"{BLUE} Starting with Naive Bayes classifier {ENDC}")
    nb = Model("Naive Bayes", MultinomialNB())
    nb.evaluate(X_train, y_train, X_test, y_test)

    nb.kfold(None, data, 'tfidf')
    # nb.loocv(data, False)

    print(f"{BLUE} Starting with Support Vector Machine classifier {ENDC}")
    params_grid = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    ]
    svm_find_best = Model(
        "Support Vector Machine", GridSearchCV(SVC(), params_grid, cv=10)
    )
    bets_score, best_C, best_kernel, best_gamma = svm_find_best.find_best_parameters(
        X_train, y_train
    )
    svm = Model(
        "Support Vector Machine", SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
    )
    svm.evaluate(X_train, y_train, X_test, y_test)

    svm.kfold(None, data, 'tfidf')
    # svm.loocv(data, False)

    print(f"{BLUE} Starting with Logistic Regression classifier {ENDC}")
    svm = Model(
        "Logistic Regression classifier",
        LogisticRegression(multi_class="multinomial", max_iter=1000, C=1.0 / 0.01),
    )
    svm.evaluate(X_train, y_train, X_test, y_test)

    svm.kfold(None, data, 'tfidf')
    # svm.loocv(data, False)

    print(f"{BLUE} Starting with Popularity classifier {ENDC}")
    pm.evaluate(y_train, y_test)

    print(f"{BLUE} Starting with Random classifier {ENDC}")
    rm.evaluate(y_train, y_test)

    print(f"{BLUE} Starting with NN {ENDC}")
    X_train, X_test, Y_train, Y_test = split_train_test(
        data, x_col="features", y=data[["CodePreliminary"]]
    )
    X_train, X_val, Y_train, Y_val = split_train_test(
        X_train, x_col="features", y=Y_train[["CodePreliminary"]]
    )

    X_train, X_test, X_val = tfidf_features2(
        X_train,
        X_test,
        X_val,
        binary=True,
    )
    Y_test_classes = Y_test

    lb = LabelEncoder()
    lb.fit(get_classes(data).tolist())
    Y_train = lb.transform(Y_train["CodePreliminary"].tolist())
    Y_train = keras.utils.to_categorical(Y_train)

    Y_val = lb.transform(Y_val["CodePreliminary"].tolist())
    Y_val = keras.utils.to_categorical(Y_val)

    Y_test = lb.transform(Y_test["CodePreliminary"].tolist())
    Y_test = keras.utils.to_categorical(Y_test)

    nn = NN("basic", lb)
    nn.train(
        X_train,
        Y_train,
        X_val,
        Y_val,
        filename="model-tfidf",
        nb_classes=len(get_classes(data).tolist()),
    )
    nn.evaluate(X_test, Y_test, Y_test_classes)

    print(f"{BLUE} Starting with MLP {ENDC}")
    mlp = NN("mlp", lb)
    mlp.train(
        X_train,
        Y_train,
        X_val,
        Y_val,
        filename="model-tfidf",
        nb_classes=len(get_classes(data).tolist()),
    )
    mlp.evaluate(X_test, Y_test, Y_test_classes)

    print(f"{BLUE} Starting with Deep NN {ENDC}")
    deep = NN("deep", lb)
    deep.train(
        X_train,
        Y_train,
        X_val,
        Y_val,
        filename="model-tfidf",
        nb_classes=len(get_classes(data).tolist()),
    )
    deep.evaluate(X_test, Y_test, Y_test_classes)


if __name__ == "__main__":
    data = preprocess_data()
    features = custom_features_extractor(data)

    print(f"{BLUE} ############################ {ENDC}")
    print(f"{BLUE} #          TF-IDF          # {ENDC}")
    print(f"{BLUE} ############################ {ENDC}")
    tfidf(data)

    print(f"{BLUE} ############################ {ENDC}")
    print(f"{BLUE} #            BOW           # {ENDC}")
    print(f"{BLUE} ############################ {ENDC}")
    bow(data)

    print(f"{BLUE} ############################ {ENDC}")
    print(f"{BLUE} #          CUSTOM          # {ENDC}")
    print(f"{BLUE} ############################ {ENDC}")
    custom_features(features, data)
