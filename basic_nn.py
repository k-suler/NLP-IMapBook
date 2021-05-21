import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dense, Dropout
from keras.models import load_model
from keras.utils import plot_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from evaluation import Evaluator
from feature_extraction import custom_features_extractor
from preprocess import preprocess_data
from utils import get_classes, preprocess_labels, split_train_test


class NN:
    def __init__(self, type, labelEncoder):
        self.type = type
        self.model = None
        self.lb = labelEncoder

    def set_train_model(self, type, nb_classes, dims):
        self.type = type

        if self.type == "basic":
            model = keras.Sequential()
            model.add(Dense(nb_classes, input_shape=(dims,)))
            model.add(Activation("softmax"))
            model.compile(
                optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
            )
        elif self.type == "mlp":
            model = keras.Sequential()
            # model.add(Dense(nb_classes * 7, activation="relu", input_shape=(dims,)))
            model.add(Dense(512, input_shape=(dims,)))
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))
            model.add(Dense(nb_classes, activation="relu"))
            model.add(Activation("softmax"))
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
        elif self.type == "deep":
            model = keras.Sequential()
            model.add(Dense(512, input_shape=(dims,)))
            model.add(Activation('tanh'))
            # model.add(
            #     Dense(
            #         nb_classes * 3,
            #         input_shape=(dims,),
            #         # kernel_initializer=keras.initializers.he_normal(seed=1),
            #         activation="relu",
            #     )
            # )
            model.add(keras.layers.Dropout(0.71))
            model.add(
                keras.layers.Dense(
                    400,
                    kernel_initializer=keras.initializers.he_normal(seed=2),
                    activation="relu",
                )
            )
            model.add(keras.layers.Dropout(0.71))
            model.add(
                keras.layers.Dense(
                    nb_classes,
                    kernel_initializer=keras.initializers.RandomNormal(
                        mean=0.0, stddev=0.05, seed=5
                    ),
                    activation="softmax",
                )
            )
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
        else:
            raise Exception("Unsupported model type")
        self.model = model

    def set_type(self, type):
        self.type

    def train(
        self,
        X_train,
        Y_train,
        type="deep",
        X_validation=None,
        Y_validation=None,
        save_model=False,
        filename="model",
        nb_classes=None,
    ):

        if not nb_classes:
            nb_classes = Y_train.shape[1]

        dims = X_train.shape[1]
        self.set_train_model(self.type, nb_classes, dims)

        fBestModel = f"./saved_models/{filename}-{self.type}.h5"
        early_stop = EarlyStopping(monitor="val_loss", patience=2, verbose=1)
        best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

        history = self.model.fit(
            X_train,
            Y_train,
            epochs=500,
            batch_size=1024,
            validation_split=0.1,
            callbacks=[best_model, early_stop],
            verbose=True,
        )

        if save_model:
            self.model.save("./saved_models/" + fBestModel)

        self.model.summary()

        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()
        # summarize history for loss
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()

    def load_fitted_model(self, filename):
        self.model = load_model(filename)

    def evaluate(self, X_test, Y_test):
        if not self.model:
            raise Exception("Load or fit new model first")

        score, acc = self.model.evaluate(X_test, Y_test, batch_size=3)
        print("Test accuracy:", acc)

        evaluator = Evaluator()
        predictions_encoded = self.model.predict(X_test)
        predictions = self.lb.inverse_transform(
            [np.argmax(pred) for pred in predictions_encoded]
        )
        evaluator.accuracy(Y_test_classes, predictions)
        evaluator.classification_report(Y_test_classes, predictions)
        evaluator.confusion_matrix(Y_test_classes, predictions)

    def plot_model(self, show_shapes=False, show_dtype=False):
        plot_model(
            self.model,
            to_file=f"model-{self.type}.png",
            show_shapes=show_shapes,
            show_dtype=show_dtype,
        )

    def plot_model(self, show_shapes=False, show_dtype=False):
        plot_model(
            self.model,
            to_file=f"model-{self.type}.png",
            show_shapes=show_shapes,
            show_dtype=show_dtype,
        )


tfidf = TfidfVectorizer(binary=True, stop_words="english", max_df=0.5, min_df=2)


def tfidf_features_my(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    x = x.astype("float16")
    return x


if __name__ == "__main__":

    data = preprocess_data()
    features = custom_features_extractor(data)
    X_train, X_test, Y_train, Y_test = split_train_test(
        features, x_col="features", y=data[["CodePreliminary"]]
    )
    Y_test_classes = Y_test
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    lb = LabelEncoder()
    lb.fit(get_classes(data).tolist())
    Y_train = lb.transform(Y_train["CodePreliminary"].tolist())
    Y_train = keras.utils.to_categorical(Y_train)
    Y_test = lb.transform(Y_test["CodePreliminary"].tolist())
    Y_test = keras.utils.to_categorical(Y_test)

    nn = NN("basic", lb)
    nn.train(
        X_train,
        Y_train,
        save_model=True,
        filename="model-tfidf",
        nb_classes=len(get_classes(data).tolist()),
    )
    # nn.load_fitted_model("./saved_models/model-bg-basic.h5")
    nn.evaluate(X_test, Y_test)
    # nn.plot_model()

# firstTestExample = X_test[:1]
# prediction = model.predict(firstTestExample)
# predictedClass = np.argmax(prediction)

# print(f"Test example: \n\t{firstTestExample}")
# print(f"Predicted vector: \n\t{prediction}")
# print(f"Predicted class index: \n\t{predictedClass}")
