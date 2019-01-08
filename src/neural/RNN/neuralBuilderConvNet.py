import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from src.neural.dataLoader.loadKaggleV3 import trainingGenerator, testGenerator
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session
# à utiliser avec la base de donnée kaggleV3 (src.neural.dataBuilder.buildKaggleV3)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement= True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def buildModel():
    # the tensorboard callBack
    nbChunksTraining = 370
    nbChunksTest = 8

    testGene = testGenerator(nbChunksTest)
    trainingGene = trainingGenerator(nbChunksTraining)
    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model = Sequential([
        Conv2D(filters=128,kernel_size=[3,3],activation="relu", input_shape=(28, 28, 1)),
        Conv2D(filters=128,kernel_size=[3,3],activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(filters=64,kernel_size=[3,3],activation="relu"),
        Conv2D(filters=64,kernel_size=[3,3],activation="relu"),
        Conv2D(filters=32,kernel_size=[3,3],activation="relu"),
        Flatten(),
        Dense(128, activation='sigmoid'),
        Dense(128, activation='sigmoid'),
        Dropout(0.5),
        Dense(26, activation='softmax'),
    ])

    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    model.fit_generator(trainingGene, epochs=10, steps_per_epoch=nbChunksTraining, validation_data=testGene, validation_steps=nbChunksTest, callbacks=[tbCallBack])

    return model

def saveModel(model):
    model.save("src/neural/RNN/model/hand_written_recognition_Conv.model")
