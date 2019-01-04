import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D
import src.neural.dataLoader.loadKaggle as bdd
from keras.callbacks import TensorBoard

def buildModel():
    dataset = bdd.loadAll()

    # the tensorboard callBack 
    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    x_train = np.array([x for x in dataset["imgNF"].values])
    sampleSize, n, m = x_train.shape
    x_train = x_train.reshape(sampleSize, n, m, 1)

    y_train = dataset["sparse_lettre"].values

    model = Sequential([
        Conv2D(filters=128,kernel_size=[3,3],activation="relu", input_shape=(28, 28, 1)),
        Conv2D(filters=128,kernel_size=[3,3],activation="relu"),
        Conv2D(filters=128,kernel_size=[3,3],activation="relu"),
        Flatten(),
        Dense(128, activation='sigmoid'),
        Dense(128, activation='sigmoid'),
        Dropout(0.05),
        Dense(26, activation='softmax'),
    ])

    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    model.fit(x_train, y_train, validation_split=0.05, epochs=10, callbacks=[tbCallBack])
    return model

def saveModel(model):
    model.save("src/neural/RNN/model/hand_written_recognition_Conv.model")

