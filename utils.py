from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

col_to_predict = "CodePreliminary"


def get_classes(df):
    """Return all distinct classes"""
    classes = df.CodePreliminary.unique()
    return classes


def split_train_test(X, x_col="lemmas", y=None, stratify=True, test_size=0.2):
    """Split the data to train and test set"""

    if y is None:
        y = X[[col_to_predict]]

    if stratify:
        stratify = y
    else:
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True, stratify=stratify
    )
    return X_train, X_test, y_train, y_test


def preprocess_data(X, scaler=None):
    """Preprocess input data by standardise features
    by removing the mean and scaling to unit variance"""
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    """Encode labels with values among 0 and `n-classes-1`"""
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder
