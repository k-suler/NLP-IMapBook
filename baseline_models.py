import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import utils
import random


class PopularityModel():

    def name(self):
        return "Popularity model"

    def get_most_representative_class(self, X_train):
        """ Return most representative class """
        item_counts = X_train[utils.col_to_predict].value_counts()
        most_reprenetative = item_counts.idxmax()
        return most_reprenetative

    def predict(self, X_train):
        most_representative_class = self.get_most_representative_class(X_train)
        return [most_representative_class * len(X_train)]


class RandomModel():

    def name(self):
        return "Random model"

    def get_random_class(self, X_train):
        """ Return random class """
        classes = utils.get_classes(X_train)
        return random.choice(classes)

    def predict(self, X_train):
        return [self.get_random_class(X_train) * len(X_train)]


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

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        model = self.train(X_train, Y_train)
        preds = self.predict(model, X_test)
        return metrics.accuracy_score(preds, Y_test)


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
        return metrics.accuracy_score(preds, Y_test)




