import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def lettreGenerator():
    iniLettre = 64
    for i in range(1,63):
        if i == 37:
            iniLettre = 65
            yield {"lettre": chr(iniLettre), "sample": i}
        elif i < 11:
            yield {"lettre": str(i-1), "sample": i}
        else:
            iniLettre+=1
            yield {"lettre": chr(iniLettre), "sample": i}


def inverseLettre(l):
    gene = lettreGenerator()
    i = 0
    for w in gene:
        if w["lettre"] == l:
            return i
        i+=1
    return i


def loadAllPath():
    pathDf = pd.DataFrame(columns=["path", "sample"])
    f = open("src/neural/trainingData/goodSample/Hnd/all.txt")
    for l in f:
        sample = str(int(l.split("/")[1].strip("Sample")))
        pathDf = pathDf.append({"path": l.strip("\n"), "sample": int(sample)}, ignore_index=True)
    return pathDf


plt.show()
def loadAllImg():
    df = pd.DataFrame(columns=["image", "im_norm_flat", "normLettre", "lettre"])
    pathDf = loadAllPath()
    gene = lettreGenerator()
    for ind in gene:
        pathList = pathDf.loc[pathDf["sample"] == ind["sample"]]["path"].values
        for path in pathList:
            img = cv2.imread("src/neural/trainingData/goodSample/Hnd/" + path, 0)
            img = cv2.resize(img,(28,28))
            img = cv2.bitwise_not(img)
            imNF = normalize(img, axis=1)
            plt.imshow(imNF, cmap=plt.cm.binary)
            plt.pause(0.001)
            plt.clf()
            imNF = imNF.flatten()
            df = df.append({"image": img, "im_norm_flat": imNF, "normLettre": inverseLettre(str(ind["lettre"])), "lettre": ind["lettre"]}, ignore_index=True)
    return df


def makeData():
    df = loadAllImg()
    df.to_json("src/neural/trainingData/processed.json")

