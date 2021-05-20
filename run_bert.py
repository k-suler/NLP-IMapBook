import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from evaluation import Evaluator

print(f"Tensorflow version: {tf.__version__}")
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import TFBertForSequenceClassification, BertTokenizer

from tqdm import tqdm
from preprocess import preprocess_data
from utils import get_classes, preprocess_labels, split_train_test

data = preprocess_data()

X = np.array(data['Message'].astype(str).values.tolist())
y = np.array(data['CodePreliminary'].astype(str).values.tolist())

lb = LabelEncoder()

lb.fit(get_classes(data).tolist())
y = lb.transform(y)
# y = np.array(keras.utils.to_categorical(y))

# y = lb.transform(y)
# y_test = keras.utils.to_categorical(y_test)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=13
)
# X_val, X_test, y_val, y_test = train_test_split(
#     X_test, y_test, test_size=0.5, random_state=42
# )
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def get_token_ids(texts):
    return bert_tokenizer.batch_encode_plus(
        texts, add_special_tokens=True, max_length=128, pad_to_max_length=True
    )["input_ids"]


train_token_ids = get_token_ids(X_train)
test_token_ids = get_token_ids(X_test)
train_data = tf.data.Dataset.from_tensor_slices(
    (tf.constant(train_token_ids), tf.constant(y_train))
).batch(12)
test_data = tf.data.Dataset.from_tensor_slices(
    (tf.constant(test_token_ids), tf.constant(y_test))
).batch(12)


class CustomIMapModel(tf.keras.Model):
    def __init__(
        self,
        vocabulary_size,
        embedding_dimensions=128,
        cnn_filters=50,
        dnn_units=512,
        model_output_classes=2,
        dropout_rate=0.1,
        training=False,
        name="custom_imdb_model",
    ):
        super(CustomIMapModel, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size, embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(
            filters=cnn_filters, kernel_size=2, padding="valid", activation="relu"
        )
        self.cnn_layer2 = layers.Conv1D(
            filters=cnn_filters, kernel_size=3, padding="valid", activation="relu"
        )
        self.cnn_layer3 = layers.Conv1D(
            filters=cnn_filters, kernel_size=4, padding="valid", activation="relu"
        )
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1, activation="sigmoid")
        else:
            self.last_dense = layers.Dense(
                units=model_output_classes, activation="softmax"
            )

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat(
            [l_1, l_2, l_3], axis=-1
        )  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


VOCAB_LENGTH = len(bert_tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = len(get_classes(data))
DROPOUT_RATE = 0.2
NB_EPOCHS = 7

custom_model = CustomIMapModel(
    vocabulary_size=VOCAB_LENGTH,
    embedding_dimensions=EMB_DIM,
    cnn_filters=CNN_FILTERS,
    dnn_units=DNN_UNITS,
    model_output_classes=OUTPUT_CLASSES,
    dropout_rate=DROPOUT_RATE,
)


custom_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["sparse_categorical_accuracy"],
)

custom_model.fit(train_data, epochs=NB_EPOCHS)


results_predicted = custom_model.predict(test_data)
evaluator = Evaluator()
predictions = lb.inverse_transform([np.argmax(pred) for pred in results_predicted])
test_classes = lb.inverse_transform(y_test)
evaluator.accuracy(test_classes, predictions)
evaluator.classification_report(test_classes, predictions)
evaluator.confusion_matrix(test_classes, predictions)
