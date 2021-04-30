from preprocess import preprocess_data
from utils import split_train_test
from feature_extraction import tfidf_features, bag_of_words_features
from baseline_models import NaiveBayes, LinearSVM, PopularityModel, RandomModel
import warnings

warnings.filterwarnings("ignore")

nb = NaiveBayes()
svm = LinearSVM()
pm = PopularityModel()
rm = RandomModel()


def tfidf(X_train, X_test, y_train, y_test):
    # word level TF-IDF
    X_train, X_test = tfidf_features(X_train['joined_lemas'].tolist(), X_test['joined_lemas'].tolist())

    # Naive Bayes model
    accuracy_nb, report_nb = nb.evaluate(X_train, y_train, X_test, y_test)
    print(accuracy_nb)
    print(report_nb)

    accuracy_svm, report_svm = svm.evaluate(X_train, y_train, X_test, y_test)
    print(accuracy_svm)
    print(report_svm)


if __name__ == "__main__":
    data = preprocess_data()
    X_train, X_test, y_train, y_test = split_train_test(data)

    tfidf(X_train, X_test, y_train, y_test)




