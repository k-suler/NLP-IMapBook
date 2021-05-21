import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    roc_auc_score,
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
    LeaveOneOut,
    GridSearchCV,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

import utils
import random
from evaluation import Evaluator
from feature_extraction import tfidf_features_1, bag_of_words_features_1

evaluator = Evaluator()


class PopularityModel:
    def name(self):
        return "Popularity model"

    def get_most_representative_class(self, Y_train):
        """Return most representative class"""
        item_counts = Y_train[utils.col_to_predict].value_counts()
        most_reprenetative = item_counts.idxmax()
        return most_reprenetative

    def predict(self, train, test):
        most_representative_class = self.get_most_representative_class(train)
        return [most_representative_class for _ in range(len(test))]

    def evaluate(self, Y_train, Y_test):
        preds = self.predict(Y_train, Y_test)
        preds = np.array(preds)
        evaluator.accuracy(Y_test, preds)
        # evaluator.classification_report(Y_test, preds)
        evaluator.confusion_matrix(Y_test, preds)


class RandomModel:
    def name(self):
        return "Random model"

    def get_random_class(self, X_train):
        """Return random class"""
        classes = utils.get_classes(X_train)
        return random.choice(classes)

    def predict(self, train, test):
        return [self.get_random_class(train) for _ in range(len(test))]

    def evaluate(self, Y_train, Y_test):
        preds = self.predict(Y_train, Y_test)
        preds = np.array(preds)
        evaluator.accuracy(Y_test, preds)
        # evaluator.classification_report(Y_test, preds)
        evaluator.confusion_matrix(Y_test, preds)


class NaiveBayes:
    def name(self):
        return "Naive Bayes"

    def train(self, X_train, Y_train):
        nb = MultinomialNB()
        model = nb.fit(X_train, Y_train)
        return model

    def predict(self, model, X_test):
        preds = model.predict(X_test)
        return preds

    def predict_proba(self, model, X_test):
        preds_prob = model.predict_proba(X_test)
        return preds_prob

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        model = self.train(X_train, Y_train)
        preds = self.predict(model, X_test)
        auc = metrics.accuracy_score(preds, Y_test)
        class_report = metrics.classification_report(Y_test, preds)
        return auc, class_report


class LinearSVM:
    def name(self):
        return "Linear Support Vector Machine"

    def train(self, X_train, Y_train):
        nb = SGDClassifier()
        model = nb.fit(X_train, Y_train)
        return model

    def predict(self, model, X_test):
        preds = model.predict(X_test)
        return preds

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        model = self.train(X_train, Y_train)
        preds = self.predict(model, X_test)
        auc = metrics.accuracy_score(preds, Y_test)
        class_report = metrics.classification_report(Y_test, preds)
        return auc, class_report


class Model:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model

    def name(self):
        return self.model_name

    def train(self, X_train, Y_train):
        nb = self.model
        model = nb.fit(X_train, Y_train)
        return model

    def predict(self, model, X_test):
        preds = model.predict(X_test)
        return preds

    def kfold(self, features, data, type):
        """K-fold cross validation - train the model k times"""
        model = self.model

        f1, prec, recall, acc = [], [], [], []
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        if type == 'custom':
            input_data = features
        else:
            input_data = data
        for train_index, test_index in kf.split(input_data):
            if type == "tfidf" or type == "bow":
                xtrain, xtest = data.iloc[train_index, 13], data.iloc[test_index, 13]
            ytrain, ytest = data.iloc[train_index, 6], data.iloc[test_index, 6]

            if type == 'tfidf':
                X_train, X_test = tfidf_features_1(
                    xtrain.tolist(), xtest.tolist(), True
                )
            elif type == 'bow':
                X_train, X_test = bag_of_words_features_1(
                    xtrain.tolist(), xtest.tolist(), kfold=True
                )
            elif type == 'custom':
                X_train = features.tocsr()[train_index]
                X_test = features.tocsr()[test_index]

            model.fit(X_train, ytrain)
            y_predicted = model.predict(X_test)

            f1.append(metrics.f1_score(ytest, y_predicted, average="weighted"))
            recall.append(metrics.recall_score(ytest, y_predicted, average="weighted"))
            acc.append(metrics.accuracy_score(ytest, y_predicted))
            prec.append(metrics.precision_score(ytest, y_predicted, average="weighted"))

        print(
            f"10-FOLD - Accuracy: {round(np.mean(acc), 3)}, Precision: {round(np.mean(prec), 3)} "
            f"Recall: {round(np.mean(recall), 3)}, F1-score: {round(np.mean(f1), 3)}"
        )

    def loocv(self, features, data, type):
        """Leave one out cross validation - train the model n times (n = number of values in the data)"""
        model = self.model

        f1, prec, recall, acc = [], [], [], []
        kf = KFold(n_splits=data.shape[0], shuffle=True, random_state=0)
        for train_index, test_index in kf.split(features):
            if type == "tfidf" or type == "bow":
                xtrain, xtest = data.iloc[train_index, 12], data.iloc[test_index, 12]
            ytrain, ytest = data.iloc[train_index, 6], data.iloc[test_index, 6]

            if type == 'tfidf':
                X_train, X_test = tfidf_features_1(
                    xtrain.tolist(), xtest.tolist(), True
                )
            elif type == 'bow':
                X_train, X_test = bag_of_words_features_1(
                    xtrain.tolist(), xtest.tolist(), kfold=True
                )
            elif type == 'custom':
                X_train = features.tocsr()[train_index]
                X_test = features.tocsr()[test_index]

            model.fit(X_train, ytrain)
            y_predicted = model.predict(X_test)

            f1.append(metrics.f1_score(ytest, y_predicted, average="weighted"))
            recall.append(metrics.recall_score(ytest, y_predicted, average="weighted"))
            acc.append(metrics.accuracy_score(ytest, y_predicted))
            prec.append(metrics.precision_score(ytest, y_predicted, average="weighted"))

        print(
            f"LOOCV - Accuracy: {round(np.mean(acc), 3)}, Precision: {round(np.mean(prec), 3)}"
            f"Recall: {round(np.mean(recall), 3)}, F1-score: {round(np.mean(f1), 3)}"
        )

    def find_best_parameters(self, X_train, Y_train):
        model = self.model
        model.fit(X_train, Y_train)
        print("Best score for training data:", model.best_score_, "\n")

        # View the best parameters for the model found using grid search
        print("Best C:", model.best_estimator_.C, "\n")
        print("Best Kernel:", model.best_estimator_.kernel, "\n")
        print("Best Gamma:", model.best_estimator_.gamma, "\n")

        return (
            model.best_score_,
            model.best_estimator_.C,
            model.best_estimator_.kernel,
            model.best_estimator_.gamma,
        )

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        model = self.train(X_train, Y_train)
        preds = self.predict(model, X_test)
        evaluator.accuracy(Y_test, preds)
        # evaluator.classification_report(Y_test, preds)
        evaluator.confusion_matrix(Y_test, preds)
        return model
