import json
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from preprocess import preprocess_data
from utils import get_classes, preprocess_labels, split_train_test

tfidf = TfidfVectorizer(binary=True)


def tfidf_features(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    x = x.astype('float16')
    return x


data = preprocess_data()
X_train, X_test, Y_train, Y_test = split_train_test(data, x_col="joined_lemas")

X_train = tfidf_features(X_train['joined_lemas'].tolist(), flag="train")
X_test = tfidf_features(X_test['joined_lemas'].tolist(), flag="test")


lb = LabelEncoder()
lb.fit(get_classes(data).tolist())
Y_train = lb.transform(Y_train['CodePreliminary'].tolist())
Y_train = keras.utils.to_categorical(Y_train)
Y_test = lb.transform(Y_test['CodePreliminary'].tolist())
Y_test = keras.utils.to_categorical(Y_test)


nb_classes = Y_train.shape[1]
dims = X_train.shape[1]

# print("Building simple model...")

# model = keras.Sequential()
# model.add(Dense(nb_classes, input_shape=(dims,)))
# model.add(Activation('softmax'))

# print("Compile structure ...")
# model.compile(optimizer='sgd', loss='categorical_crossentropy')

# print("Fit model ...")
# history = model.fit(X_train, Y_train, epochs=500, batch_size=512)

# # TODO: make learning and testing set represent each class

# score, acc = model.evaluate(X_test, Y_test, batch_size=3)
# print('Test accuracy:', acc)

# print(f"The dataset contains {nb_classes} classes and {dims} features.")
# print(f"Labels are one-hot encoded for training: \n{Y_train}")

# print(f"\nTraining data: \n{X_train}")


# print("Building model ...")
# model = Sequential()
# model.add(Dense(nb_classes, input_shape=(dims,)))
# model.add(Activation('softmax'))

# print("Compile structure ...")
# model.compile(optimizer='sgd', loss='categorical_crossentropy')


# fBestModel = 'best_model.h5'
# early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
# best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

# print("Fit model ...")
# model.fit(
#     X_train,
#     Y_train,
#     validation_data=(X_test, Y_test),
#     epochs=50,
#     batch_size=2,
#     verbose=True,
#     callbacks=[best_model, early_stop],
# )
# model.fit(X_train, Y_train)


model = keras.Sequential()

model.add(Dense(nb_classes, input_shape=(dims,)))
# model.add(
#     keras.layers.Dense(
#         nb_classes,
#         kernel_initializer=keras.initializers.he_normal(seed=1),
#         activation='relu',
#         input_shape=(dims,)
#         #         input_dim=3010,
#     )
# )
model.add(keras.layers.Dropout(0.81))
model.add(
    keras.layers.Dense(
        1000, kernel_initializer=keras.initializers.he_normal(seed=2), activation='relu'
    )
)
model.add(keras.layers.Dropout(0.81))
model.add(
    keras.layers.Dense(
        nb_classes,
        kernel_initializer=keras.initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=4
        ),
        activation='softmax',
    )
)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=700, batch_size=512)
model.save_weights("model.h5")
print("Saved model to disk")

score, acc = model.evaluate(X_test, Y_test, batch_size=3)
print('Test accuracy:', acc)

print(history.history.keys())
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


print("\nModel description:")
model.summary()

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# Graphical
SVG(model_to_dot(model).create(prog='dot', format='svg'))


firstTestExample = X_test[:1]
prediction = model.predict(firstTestExample)
predictedClass = np.argmax(prediction)

print(f"Test example: \n\t{firstTestExample}")
print(f"Predicted vector: \n\t{prediction}")
print(f"Predicted class index: \n\t{predictedClass}")
