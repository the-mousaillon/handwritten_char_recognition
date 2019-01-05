import pandas as pd
import numpy as np

# lis la base de données par chunks, du fait de sa taille trop importante pour la ram
def read_part(n):
    return pd.read_json("src/neural/trainingData/kaggleV3/train/kagglePart" + str(n) + ".json")

def makeChunk(df, n):
    fractile = int(len(df)/n)
    chunkList = []
    a = 0
    b = 0
    for _ in range(n-1):
        b = a + fractile
        chunkList += [df.loc[a:b]]
        a = b+1
    chunkList += [df.loc[a:]]
    return chunkList

# !! infinite generator
def testGenerator(nbChunks):
    df = pd.read_json("src/neural/trainingData/kaggleV3/test/test.json")
    chunks = makeChunk(df, nbChunks//30)
    while 1:
        for c in chunks:
            x_train = np.array([x for x in c["imgNF"].values])
            sampleSize, n, m = x_train.shape
            x_train = x_train.reshape(sampleSize, n, m, 1)
            y_train = df["sparse_lettre"].values
            yield (x_train, y_train)
    return (x_train, y_train)

# !! infinite generator
def trainingGenerator(nbChunks):
    while 1:
        for i in range(30):
            df = read_part(i)
            chunks = makeChunk(df, nbChunks//30)
            for c in chunks:
                x_train = np.array([x for x in c["imgNF"].values])
                sampleSize, n, m = x_train.shape
                x_train = x_train.reshape(sampleSize, n, m, 1)
                y_train = df["sparse_lettre"].values
                yield (x_train, y_train)

