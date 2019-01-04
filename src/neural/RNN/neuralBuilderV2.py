import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import src.neural.dataLoader.loadKaggle as bdd
from keras.callbacks import TensorBoard

def buildModel():
    dataset = bdd.loadAll()

    # the tensorboard callBack 
    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    x_train = np.array([x for x in dataset["imgNF"].values])

    y_train = dataset["sparse_lettre"].values

    model = Sequential([
        Dense(784, activation='sigmoid'),
        Dense(256, activation='sigmoid'),
        Dense(256, activation='sigmoid'),
        Dense(256, activation='sigmoid'),
        Dense(256, activation='sigmoid'),
        Dense(256, activation='sigmoid'),
        Dense(26, activation='softmax'),
    ])


    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    model.fit(x_train, y_train, validation_split=0.05, epochs=10, callbacks=[tbCallBack])
    return model

def saveModel(model):
    model.save("src/neural/RNN/model/hand_written_recognition_kaggle.model")

