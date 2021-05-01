import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, LeaveOneOut
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import utils
import random
from evaluation import Evaluator

evaluator = Evaluator()


class PopularityModel():

    def name(self):
        return "Popularity model"

    def get_most_representative_class(self, Y_train):
        """ Return most representative class """
        item_counts = Y_train[utils.col_to_predict].value_counts()
        most_reprenetative = item_counts.idxmax()
        return most_reprenetative

    def predict(self, df):
        most_representative_class = self.get_most_representative_class(df)
        return [most_representative_class for _ in range(len(df))]

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        preds = self.predict(Y_test)
        preds = np.array(preds)
        evaluator.accuracy(Y_test, preds)
        evaluator.classification_report(Y_test, preds)


class RandomModel():

    def name(self):
        return "Random model"

    def get_random_class(self, X_train):
        """ Return random class """
        classes = utils.get_classes(X_train)
        return random.choice(classes)

    def predict(self, df):
        return [self.get_random_class(df) for _ in range(len(df))]

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        preds = self.predict(Y_test)
        preds = np.array(preds)
        evaluator.accuracy(Y_test, preds)
        evaluator.classification_report(Y_test, preds)


class NaiveBayes():

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


class LinearSVM():

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


class Model():

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

    def kfold(self, data, target):
        """ K-fold cross validation - train the model k times"""
        model = self.model
        model.fit(data, target)
        preds = cross_val_predict(model, data, target, cv=10)
        scores = cross_val_score(model, data, target, cv=10)
        print(f"10-FOLD - accuracy is {round(np.mean(scores), 2)}")

    def loocv(self, data, target):
        """Leave one out cross validation - train the model n times (n = number of values in the data)"""
        cv = LeaveOneOut()
        model = self.model
        scores = cross_val_score(model, data, target, scoring='accuracy', cv=cv, n_jobs=-1)
        print(f"LOOCV - accuracy is {round(np.mean(scores), 2)}")

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        model = self.train(X_train, Y_train)
        preds = self.predict(model, X_test)
        evaluator.accuracy(Y_test, preds)
        evaluator.classification_report(Y_test, preds)
        evaluator.confusion_matrix(Y_test, preds)

