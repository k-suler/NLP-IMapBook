import os
import warnings

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
from feature_extraction import bag_of_words_features_1, tfidf_features_1, custom_features
from preprocess import preprocess_data
from utils import get_classes, preprocess_labels, split_train_test

warnings.filterwarnings("ignore")

pm = PopularityModel()
rm = RandomModel()


def bow(data, X_train, X_test, y_train, y_test):
    # word level TF-IDF
    X_train, X_test = bag_of_words_features_1(X_train, X_test, kfold=False)

    print(f"{BLUE} Starting with Naive Bayes classifier {ENDC}")
    nb = Model("Naive Bayes", MultinomialNB())
    nb.evaluate(X_train, y_train, X_test, y_test)

    # nb.kfold(data, False)
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
    svm = Model("Support Vector Machine", SVC(kernel='rbf', gamma='auto', C=1000))
    svm.evaluate(X_train, y_train, X_test, y_test)

    # svm.kfold(data, False)
    # svm.loocv(data, False)

    print(f"{BLUE} Starting with Logistic Regression classifier {ENDC}")
    lg = Model(
        "Support Vector Machine",
        LogisticRegression(multi_class="multinomial", max_iter=1000, C=1.0 / 0.01),
    )
    lg.evaluate(X_train, y_train, X_test, y_test)

    # lg.kfold(data, False)
    # lg.loocv(data, False)

    print(f"{BLUE} Starting with Popularity classifier {ENDC}")
    pm.evaluate(X_train, y_train, X_test, y_test)

    print(f"{BLUE} Starting with Random classifier {ENDC}")
    rm.evaluate(X_train, y_train, X_test, y_test)

    # print(f"{BLUE} Starting with basic nn classifier {ENDC}")
    # lb = LabelEncoder()
    # lb.fit(get_classes(data).tolist())
    # Y_train = lb.transform(y_train["CodePreliminary"].tolist())
    # Y_train = keras.utils.to_categorical(Y_train)
    # Y_test = lb.transform(y_test["CodePreliminary"].tolist())
    # Y_test = keras.utils.to_categorical(Y_test)
    # nn = NN("basic", lb)
    # # nn.train(X_train, Y_train, save_model=True, filename="model-tfidf")
    # nn.load_fitted_model("./saved_models/model-bg-basic.h5")
    # nn.evaluate(X_test, Y_test)

    # print(f"{BLUE} Starting with deep nn classifier {ENDC}")
    # nn.set_type("deep")
    # nn.load_fitted_model("./saved_models/model-bg-deep.h5")
    # nn.evaluate(X_test, Y_test)

    # print(f"{BLUE} Starting with MLP nn classifier {ENDC}")
    # nn.set_type("mlp")
    # nn.load_fitted_model("./saved_models/model-bg-mlp.h5")
    # nn.evaluate(X_test, Y_test)


def tfidf(data, X_train, X_test, y_train, y_test):
    # word level TF-IDF
    X_train, X_test = tfidf_features_1(X_train, X_test, kfold=False)

    print(f"{BLUE} Starting with Naive Bayes classifier {ENDC}")
    nb = Model("Naive Bayes", MultinomialNB())
    nb.evaluate(X_train, y_train, X_test, y_test)

    # nb.kfold(data, False)
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

    # svm.kfold(data, False)
    # svm.loocv(data, False)

    print(f"{BLUE} Starting with Logistic Regression classifier {ENDC}")
    svm = Model(
        "Logistic Regression classifier",
        LogisticRegression(multi_class="multinomial", max_iter=1000, C=1.0 / 0.01),
    )
    svm.evaluate(X_train, y_train, X_test, y_test)

    # svm.kfold(data, False)
    # svm.loocv(data, False)

    print(f"{BLUE} Starting with Popularity classifier {ENDC}")
    pm.evaluate(X_train, y_train, X_test, y_test)

    print(f"{BLUE} Starting with Random classifier {ENDC}")
    rm.evaluate(X_train, y_train, X_test, y_test)

    # print(f"{BLUE} Starting with basic nn classifier {ENDC}")
    # lb = LabelEncoder()
    # lb.fit(get_classes(data).tolist())
    # Y_train = lb.transform(y_train["CodePreliminary"].tolist())
    # Y_train = keras.utils.to_categorical(Y_train)
    # Y_test = lb.transform(y_test["CodePreliminary"].tolist())
    # Y_test = keras.utils.to_categorical(Y_test)
    # nn = NN("basic", lb)
    # # nn.train(X_train, Y_train, save_model=True, filename="model-tfidf")
    # nn.load_fitted_model("./saved_models/model-tfidf-basic.h5")
    # nn.evaluate(X_test, Y_test)

    # print(f"{BLUE} Starting with deep nn classifier {ENDC}")
    # nn.set_type("deep")
    # nn.load_fitted_model("./saved_models/model-tfidf-deep.h5")
    # nn.evaluate(X_test, Y_test)

    # print(f"{BLUE} Starting with MLP nn classifier {ENDC}")
    # nn.set_type("mlp")
    # nn.load_fitted_model("./saved_models/model-tfidf-mlp.h5")
    # nn.evaluate(X_test, Y_test)


if __name__ == "__main__":
    data = preprocess_data()
    features = custom_features(data)
    X_train, X_test, y_train, y_test = split_train_test(data, x_col="lemas")


    print(f"{BLUE} ############################ {ENDC}")
    print(f"{BLUE} #          TF-IDF          # {ENDC}")
    print(f"{BLUE} ############################ {ENDC}")
    tfidf(data, X_train, X_test, y_train, y_test)

    print(f"{BLUE} ############################ {ENDC}")
    print(f"{BLUE} #            BOW           # {ENDC}")
    print(f"{BLUE} ############################ {ENDC}")
    bow(data, X_train, X_test, y_train, y_test)
