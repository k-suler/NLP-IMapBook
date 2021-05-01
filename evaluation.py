from sklearn import metrics
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

class Evaluator():

    def auc_roc_score(self, model, Y_test, preds):
        """Return AUC value"""
        roc_auc_value = roc_auc_score(Y_test, preds)
        print(f"AUC value is {roc_auc_value}")

    def accuracy(self, Y_test, preds):
        auc = metrics.accuracy_score(preds, Y_test)
        print(f"Classification accuracy is {auc}")
        return auc

    def classification_report(self, Y_test, preds):
        cr = metrics.classification_report(Y_test, preds)
        print(f"Classification report:\n {cr}")
        return cr

    def confusion_matrix(self, Y_test, preds):
        cm = metrics.confusion_matrix(Y_test, preds)
        print(f"Confusion matrix:\n {cm}")
        precision = metrics.precision_score(Y_test, preds, average='weighted')
        recall = metrics.recall_score(Y_test, preds, average='weighted')
        f1_score = metrics.f1_score(Y_test, preds, average='weighted')
        print(f"Precision: {round(precision, 2)}, "
              f"recall: {round(recall, 2)}, "
              f"F1-score: {round(f1_score, 2)}")
        return cm

    def feature_importance(self, model, classes):
        """Return feature importance - value below zero means that the feature is not important"""
        imp = model.coef_[0]
        imp, names = zip(*sorted(zip(imp, classes)))
        plt.rcParams["figure.figsize"] = (15, 10)
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        plt.show()
