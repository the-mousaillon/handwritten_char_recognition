import pandas as pd
import numpy as np


# lis la base de données par chunks, du fait de sa taille trop importante pour la ram
def read_part(n):
    return pd.read_json("src/neural/trainingData/kaggleV3/train/kagglePart" + str(n) + ".json")

def loadTestData():
    df = pd.read_json("src/neural/trainingData/kaggleV3/test/test.json")
    x_train = np.array([x for x in df["imgNF"].values])
    sampleSize, n, m = x_train.shape
    x_train = x_train.reshape(sampleSize, n, m, 1)
    y_train = df["sparse_lettre"].values
    return (x_train, y_train)

def trainingGenerator():
    for i in range(30):
        df = read_part(i)
        x_train = np.array([x for x in df["imgNF"].values])
        sampleSize, n, m = x_train.shape
        x_train = x_train.reshape(sampleSize, n, m, 1)
        y_train = df["sparse_lettre"].values
        yield (x_train, y_train)

