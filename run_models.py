import os

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import utils
from preprocess import preprocess_data
from utils import split_train_test
from feature_extraction import tfidf_features_1, bag_of_words_features_1
from baseline_models import NaiveBayes, LinearSVM, PopularityModel, RandomModel, Model
import warnings
from bcolors import BLUE, ENDC


warnings.filterwarnings("ignore")

pm = PopularityModel()
rm = RandomModel()


def bow(data, X_train, X_test, y_train, y_test):
    # word level TF-IDF
    X_train, X_test = bag_of_words_features_1(
        X_train, X_test, kfold=False
    )

    print(f"{BLUE} Starting with Naive Bayes classifier {ENDC}")
    nb = Model("Naive Bayes", MultinomialNB())
    nb.evaluate(X_train, y_train, X_test, y_test)

    nb.kfold(data, False)
    nb.loocv(data, False)

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
    svm = Model(
        "Support Vector Machine", SVC(kernel='rbf', gamma='auto', C=1000)
    )
    svm.evaluate(X_train, y_train, X_test, y_test)

    svm.kfold(data, False)
    svm.loocv(data, False)

    print(f"{BLUE} Starting with Logistic Regression classifier {ENDC}")
    svm = Model(
        "Support Vector Machine",
        LogisticRegression(multi_class="multinomial", max_iter=1000, C=1.0 / 0.01),
    )
    svm.evaluate(X_train, y_train, X_test, y_test)

    svm.kfold(data, False)
    # svm.loocv(data, False)

    print(f"{BLUE} Starting with Popularity classifier {ENDC}")
    pm.evaluate(X_train, y_train, X_test, y_test)

    print(f"{BLUE} Starting with Random classifier {ENDC}")
    rm.evaluate(X_train, y_train, X_test, y_test)


def tfidf(data, X_train, X_test, y_train, y_test):
    # word level TF-IDF
    X_train, X_test = tfidf_features_1(
        X_train, X_test, kfold=False
    )

    print(f"{BLUE} Starting with Naive Bayes classifier {ENDC}")
    nb = Model("Naive Bayes", MultinomialNB())
    nb.evaluate(X_train, y_train, X_test, y_test)

    nb.kfold(data, False)
    nb.loocv(data, False)

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

    svm.kfold(data, False)
    svm.loocv(data, False)

    print(f"{BLUE} Starting with Logistic Regression classifier {ENDC}")
    svm = Model(
        "Support Vector Machine",
        LogisticRegression(multi_class="multinomial", max_iter=1000, C=1.0 / 0.01),
    )
    svm.evaluate(X_train, y_train, X_test, y_test)

    svm.kfold(data, False)
    svm.loocv(data, False)

    print(f"{BLUE} Starting with Popularity classifier {ENDC}")
    pm.evaluate(X_train, y_train, X_test, y_test)

    print(f"{BLUE} Starting with Random classifier {ENDC}")
    rm.evaluate(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    data = preprocess_data()
    X_train, X_test, y_train, y_test = split_train_test(data, x_col="lemas")

    tfidf(data, X_train, X_test, y_train, y_test)

    # bow(data, X_train, X_test, y_train, y_test)
