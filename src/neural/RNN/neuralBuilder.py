import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


dataset = pd.read_json("src/neural/trainingData/processed.json")

dataset = dataset.sample(frac=1)

dataset["im_norm_flat"] = dataset["im_norm_flat"].apply(lambda x: np.asarray(x))
dataset["image"] = dataset["image"].apply(lambda x: np.asarray(x)) 

x_train = dataset["im_norm_flat"].values

x_train = np.asarray([x for x in x_train])

y_train = dataset["normLettre"].values

model = Sequential([
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(36, activation='softmax'),
])


model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=18)

model.save("src/neural/RNN/model/hand_written_recognition.model")

